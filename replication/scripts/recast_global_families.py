"""Global selection of forget-specific directions over several weight matrices (extension, user 2026-10-29).

For every target W_m (out x in; down_proj and o_proj of every decoder block) a family of
``recast_direction_families`` gives orthonormal directions in the order of the generalised eigenvalue lambda:
output directions for the output and the gradient contrast, input directions for the input contrast. All directions
of all targets enter one order, by lambda (``score``, highest first) or random (``random``, the control: the same
directions in an order drawn by a generator) or by the absolute output energies of each direction alone,
``e_f,j / e_r,j^alpha`` (``energy_ratio``: alpha 1 is the forget / retain ratio, alpha 0 the forget energy alone; the
directions stay the family's, only their order changes, and the energies are not normalised per target, so the
ranking compares targets in the units of the budget). Taken in that order, each chosen direction keeps the share ``floor`` of
its part, beta = 1 - floor (output side ``Delta W = -sum_j beta_j u_j u_j^T W``, input side
``Delta W = -W sum_j beta_j v_j v_j^T``), until the forget output energy removed over all targets,
``sum_m tr(Delta W_m C_f,m Delta W_m^T)``, reaches the budget; the last chosen direction gets the beta that meets it.
down_proj and o_proj both write into the residual stream, so their output energies share one unit. With the budget unit
``share`` (user, 2026-10-31: up_proj and v_proj join, whose outputs live in other spaces) each target's removed forget
energy counts as a share of its own forget output energy, ``tr(Delta W_m C_f,m Delta W_m^T) / tr(W_m C_f,m W_m^T)``,
and the budget is a sum of shares over the targets. With ``s_gamma`` (user, 2026-10-31) each direction keeps its own
share ``s_j = clip(lambda_j^-gamma, s_min, 1)`` instead of one floor (lambda <= 1: untouched); the coefficients
``c_j = 1 - s_j`` are folded into the energies, and the selection then runs at beta 1. With ``s_rule="retain_share"``
(user, 2026-10-31) ``s_j = clip(e_r,j / (e_f,j + e_r,j), s_min, 1)`` from the absolute output energies of removing the
direction alone, comparable across projection types. ``calibrated_selection`` (user, 2026-10-31) instead contracts
every direction by ``s_j = clip((lambda* / lambda_j)^gamma, s_min, 1)`` with one threshold lambda* for all targets,
found by bisection so that the removed forget energy meets the budget: directions with lambda <= lambda* stay, the more
specific ones are contracted more (gamma -> infinity: a hard threshold at s_min). Input-side energies then need the
whole matrices H (``keep_h``), since the contracted set is no longer a prefix of the order.

Within a target the chosen directions are always a prefix of the target's own order (the lambda order, or the random
keys sorted within the target), so the energy a target loses is a prefix sum: additive over output directions
(``u_j^T W C W^T u_j`` each), the leading block of ``H = (V^T W^T W V) * (V^T C V)`` over input directions (C and
W^T W mix the input directions, so a direction adds ``H_kk + 2 sum_{i<k} H_ik``, which can be negative). With every
direction taken at beta = 1 - floor a target loses ``(1 - floor)^2`` of its forget energy, the most the budget can
reach.
"""

import math
from typing import Dict, Optional

import torch

from recast_direction_families import FAMILIES, ordered_directions

GLOBAL_ORDERS = ("score", "random", "energy_ratio")
BUDGET_UNITS = ("energy", "share")
S_RULES = ("lambda_power", "retain_share")
OUTPUT_SIDE = ("output_contrast", "gradient_contrast")


def target_directions(weight: torch.Tensor, cov_forget: torch.Tensor, cov_retain: torch.Tensor,
                      g_forget: torch.Tensor, g_retain: torch.Tensor, family: str, rho: float,
                      keys_generator: Optional[torch.Generator] = None, key_alpha: Optional[float] = None,
                      s_gamma: Optional[float] = None, s_min: float = 0.0, s_rule: Optional[str] = None,
                      keep_h: bool = False) -> dict:
    """One target's family (float64): its directions as columns in the target's order (on the CPU), their sort keys
    (``-lambda``, uniform random keys drawn from ``keys_generator``, or with ``key_alpha``
    ``-(log e_f,j - key_alpha log e_r,j)`` from the energies of each direction alone) and, per position of that order, the forget and
    retain output energy the direction adds at beta 1 after the ones before it (``gain_f``, ``gain_r``), its own
    energy (``self_f``, ``self_r``, the diagonal of H; the gain itself on the output side) and its displacement
    energy ``disp2``; ``energy_f`` and ``energy_r`` are ``tr(W C W^T)``. With ``s_gamma`` the gains, own energies and
    displacements are those of direction j contracted by ``coef`` c_j = 1 - clip(lambda_j^-s_gamma, s_min, 1) after the
    ones before it at theirs."""
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    if keys_generator is not None and key_alpha is not None:
        raise ValueError("give random keys or energy-ratio keys, not both")
    if key_alpha is not None and key_alpha < 0:
        raise ValueError(f"key_alpha must be >= 0, got {key_alpha}")
    if s_rule is None and s_gamma is not None:
        s_rule = "lambda_power"
    if s_rule is not None and (s_rule not in S_RULES or (s_rule == "lambda_power") != (s_gamma is not None)
                               or (s_gamma is not None and s_gamma <= 0) or not 0.0 <= s_min <= 1.0):
        raise ValueError(f"s_rule must be one of {S_RULES} (lambda_power with, and only with, s_gamma > 0) and s_min "
                         f"in [0, 1], got {s_rule}, {s_gamma}, {s_min}")
    w = weight.reshape(weight.shape[0], -1).double()
    cf, cr = cov_forget.to(w.device).double(), cov_retain.to(w.device).double()
    a_f, a_r = w @ cf @ w.transpose(0, 1), w @ cr @ w.transpose(0, 1)
    if family == "input_contrast":
        d, lam = ordered_directions(cf, cr, rho, return_values=True)
    elif family == "output_contrast":
        d, lam = ordered_directions(a_f, a_r, rho, return_values=True)
    else:
        gf = g_forget.reshape(w.shape).to(w.device).double()
        gr = g_retain.reshape(w.shape).to(w.device).double()
        d, lam = ordered_directions(gf @ gf.transpose(0, 1), gr @ gr.transpose(0, 1), rho, return_values=True)
    n = d.shape[1]
    alone_f = alone_r = None
    if key_alpha is not None or s_rule == "retain_share":
        # each direction alone: output side u^T W C W^T u, input side ||W v||^2 v^T C v
        if family == "input_contrast":
            col2 = (w @ d).square().sum(0)
            alone_f, alone_r = col2 * ((cf @ d) * d).sum(0), col2 * ((cr @ d) * d).sum(0)
        else:
            alone_f, alone_r = ((a_f @ d) * d).sum(0), ((a_r @ d) * d).sum(0)
    if key_alpha is not None:
        tiny = torch.finfo(torch.float64).tiny
        keys = -(alone_f.clamp_min(tiny).log() - key_alpha * alone_r.clamp_min(tiny).log()).cpu()
    elif keys_generator is not None:
        keys = torch.rand(n, generator=keys_generator, dtype=torch.float64)
    else:
        keys = -lam.cpu()
    order = torch.argsort(keys, stable=True)
    keys, lam = keys[order], lam.cpu()[order]
    d = d[:, order.to(d.device)]
    coef = None
    if s_rule == "lambda_power":
        s = lam.clamp_min(torch.finfo(torch.float64).tiny).pow(-s_gamma).clamp(min=s_min, max=1.0)
        coef = 1.0 - s
    elif s_rule == "retain_share":
        ef, er = alone_f.cpu()[order], alone_r.cpu()[order]
        s = (er / (ef + er).clamp_min(torch.finfo(torch.float64).tiny)).clamp(min=s_min, max=1.0)
        coef = 1.0 - s
    cw = coef.to(d.device) if coef is not None else None
    if family == "input_contrast":
        wd = w @ d
        gram = wd.transpose(0, 1) @ wd
        if cw is not None:
            gram = gram * torch.outer(cw, cw)
        out = {}
        for side, c in (("f", cf), ("r", cr)):
            h = gram * (d.transpose(0, 1) @ c @ d)
            if keep_h:
                out[f"h_{side}"] = h.float().cpu()
            prefix = h.cumsum(0).cumsum(1).diagonal()
            out[f"gain_{side}"] = torch.diff(prefix, prepend=prefix.new_zeros(1)).cpu()
            out[f"self_{side}"] = h.diagonal().cpu()
            del h, prefix
        disp2 = gram.diagonal().cpu()
        del wd, gram
    else:
        dt = d.transpose(0, 1)
        out = {}
        for side, a in (("f", a_f), ("r", a_r)):
            e = ((dt @ a) * dt).sum(1)
            if cw is not None:
                e = e * cw * cw
            out[f"gain_{side}"], out[f"self_{side}"] = e.cpu(), e.cpu()
        disp2 = (dt @ w).square().sum(1)
        disp2 = (disp2 * cw * cw if cw is not None else disp2).cpu()
    return {"side": "input" if family == "input_contrast" else "output", "directions": d.cpu(), "keys": keys,
            "lam": lam, **out, "disp2": disp2, **({"coef": coef} if coef is not None else {}),
            "energy_f": float(torch.trace(a_f)), "energy_r": float(torch.trace(a_r))}


def _last_beta(part: dict, k: int, side: str, beta: float, need: float) -> float:
    """The beta in [0, beta] of position k after k directions at ``beta`` that adds ``need`` energy on ``side``:
    ``self c^2 + 2 beta cross c = need`` with ``cross = (gain - self) / 2``."""
    a = float(part[f"self_{side}"][k])
    cross = 0.5 * (float(part[f"gain_{side}"][k]) - a)
    if a <= 0:
        return beta
    c = (-beta * cross + math.sqrt(max((beta * cross) ** 2 + a * need, 0.0))) / a
    return min(max(c, 0.0), beta)


def _removed(part: dict, k: int, beta: float, last: Optional[float], side: str) -> float:
    """The energy a target loses on ``side`` with its first k directions at ``beta`` and, when ``last`` is given,
    the (k+1)-th at ``last``."""
    e = beta * beta * float(part[f"gain_{side}"][:k].sum())
    if last is not None:
        a = float(part[f"self_{side}"][k])
        cross = 0.5 * (float(part[f"gain_{side}"][k]) - a)
        e += 2.0 * beta * last * cross + last * last * a
    return e


def global_selection(parts: Dict[str, dict], floor: float, budget: float, unit: str = "energy") -> dict:
    """Merge every target's directions by their keys and take them at beta = 1 - floor until the forget output energy
    removed over all targets reaches ``budget`` (``energy``: absolute, summed; ``share``: each target's removed energy
    divided by its own forget output energy, summed), the last one partly. Per target: ``directions`` (the length
    of its chosen prefix, the partly contracted one included), ``beta_last`` (that one's beta, else None) and the
    energies and shares it loses; ``global``: the totals, ``reached`` False when even every direction at 1 - floor
    stays below the budget (then all are taken)."""
    if not 0.0 <= floor < 1.0:
        raise ValueError(f"floor must be in [0, 1), got {floor}")
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}")
    if unit not in BUDGET_UNITS:
        raise ValueError(f"unit must be one of {BUDGET_UNITS}, got {unit!r}")
    names = list(parts)
    beta = 1.0 - floor
    weight = [1.0 / parts[n]["energy_f"] if unit == "share" and parts[n]["energy_f"] > 0 else 1.0 for n in names]
    sizes = [int(parts[n]["keys"].numel()) for n in names]
    keys = torch.cat([parts[n]["keys"] for n in names])
    label = torch.cat([torch.full((s,), i, dtype=torch.long) for i, s in enumerate(sizes)])
    gains = torch.cat([parts[n]["gain_f"] * w for n, w in zip(names, weight)]) * (beta * beta)
    order = torch.argsort(keys, stable=True)
    cum = torch.cumsum(gains[order], 0)
    hit = (cum >= budget).nonzero()
    counts = [0] * len(names)
    last_of: Dict[int, float] = {}
    reached = budget == 0.0 or hit.numel() > 0
    if budget == 0.0:
        t = 0
    elif hit.numel() == 0:
        counts = sizes[:]
        t = len(order)
    else:
        t = int(hit[0])
        chosen = torch.bincount(label[order[:t]], minlength=len(names))
        counts = [int(c) for c in chosen]
        m = int(label[order[t]])
        before = float(cum[t - 1]) if t > 0 else 0.0
        last_of[m] = _last_beta(parts[names[m]], counts[m], "f", beta, (budget - before) / weight[m])
    per_target, tot = {}, {"forget": 0.0, "retain": 0.0, "energy_f": 0.0, "energy_r": 0.0, "disp2": 0.0,
                           "share_sum": 0.0}
    for i, n in enumerate(names):
        p, k, last = parts[n], counts[i], last_of.get(i)
        rf, rr = _removed(p, k, beta, last, "f"), _removed(p, k, beta, last, "r")
        disp2 = beta * beta * float(p["disp2"][:k].sum()) + (last * last * float(p["disp2"][k]) if last is not None else 0.0)
        kk = k + (1 if last is not None else 0)
        s_mean = float(1.0 - p["coef"][:kk].mean()) if "coef" in p and kk > 0 else None
        per_target[n] = {"directions": kk, "of": sizes[i], "beta_last": last, "s_mean": s_mean,
                         "forget_energy_removed": rf, "retain_energy_removed": rr,
                         "forget_share": rf / p["energy_f"] if p["energy_f"] > 0 else None,
                         "retain_share": rr / p["energy_r"] if p["energy_r"] > 0 else None,
                         "displacement": math.sqrt(max(disp2, 0.0))}
        tot["forget"] += rf
        tot["share_sum"] += rf / p["energy_f"] if p["energy_f"] > 0 else 0.0
        tot["retain"] += rr
        tot["energy_f"] += p["energy_f"]
        tot["energy_r"] += p["energy_r"]
        tot["disp2"] += disp2
    return {"per_target": per_target, "global": {
        "floor": floor, "beta": beta, "budget": budget, "unit": unit, "reached": bool(reached),
        "forget_share_sum": tot["share_sum"],
        "directions": sum(r["directions"] for r in per_target.values()), "of": sum(sizes),
        "forget_energy_removed": tot["forget"], "retain_energy_removed": tot["retain"],
        "forget_share": tot["forget"] / tot["energy_f"] if tot["energy_f"] > 0 else None,
        "retain_share": tot["retain"] / tot["energy_r"] if tot["energy_r"] > 0 else None,
        "displacement": math.sqrt(max(tot["disp2"], 0.0))}}


def _energy_of(part: dict, coef: torch.Tensor, side: str) -> float:
    """The energy a target loses on ``side`` with every direction contracted by its ``coef`` (in the target's order)."""
    if part["side"] == "output":
        return float((coef * coef * part[f"self_{side}"]).sum())
    c = coef.float()
    return float(c @ (part[f"h_{side}"] @ c))


def calibrated_selection(parts: Dict[str, dict], budget: float, gamma: float, s_min: float = 0.0,
                         unit: str = "energy", iters: int = 60) -> dict:
    """Contract every direction of every target by ``c_j = 1 - clip((lambda* / lambda_j)^gamma, s_min, 1)``, with the
    one threshold lambda* found by bisection on log lambda* so that the forget energy removed over all targets (in
    ``unit``) meets ``budget``. Parts come from ``target_directions`` in the lambda order, with ``keep_h`` on the input
    side. Returns ``coef`` per target and the totals as ``global_selection`` does; ``reached`` False when even
    lambda* -> 0 (every direction at s_min) stays below the budget."""
    if gamma <= 0 or not 0.0 <= s_min < 1.0 or budget < 0 or unit not in BUDGET_UNITS:
        raise ValueError(f"need gamma > 0, s_min in [0, 1), budget >= 0 and a unit in {BUDGET_UNITS}")
    names = list(parts)
    weight = {n: 1.0 / parts[n]["energy_f"] if unit == "share" and parts[n]["energy_f"] > 0 else 1.0 for n in names}
    tiny = torch.finfo(torch.float64).tiny
    def coefs(log_t):
        return {n: 1.0 - (math.exp(log_t) / parts[n]["lam"].clamp_min(tiny)).pow(gamma).clamp(min=s_min, max=1.0)
                for n in names}
    def removed(log_t):
        c = coefs(log_t)
        return sum(weight[n] * _energy_of(parts[n], c[n], "f") for n in names), c
    lams = torch.cat([parts[n]["lam"] for n in names]).clamp_min(tiny)
    lo, hi = math.log(float(lams.min())) - 30.0 / gamma, math.log(float(lams.max())) + 1.0
    most, _ = removed(lo)
    reached = most >= budget
    if not reached:
        log_t = lo
    else:
        for _ in range(iters):  # removal falls as lambda* rises
            mid = 0.5 * (lo + hi)
            if removed(mid)[0] >= budget:
                lo = mid
            else:
                hi = mid
        log_t = lo
    got, c = removed(log_t)
    per_target, tot = {}, {"forget": 0.0, "retain": 0.0, "energy_f": 0.0, "energy_r": 0.0, "disp2": 0.0, "share_sum": 0.0}
    for n in names:
        p, cn = parts[n], c[n]
        rf, rr = _energy_of(p, cn, "f"), _energy_of(p, cn, "r")
        disp2 = float((cn * cn * p["disp2"]).sum())  # additive on both sides: the directions are orthonormal
        moved = cn > 0
        per_target[n] = {"directions": int(moved.sum()), "of": int(cn.numel()), "coef": cn,
                         "s_mean": float(1.0 - cn[moved].mean()) if bool(moved.any()) else None,
                         "forget_energy_removed": rf, "retain_energy_removed": rr,
                         "forget_share": rf / p["energy_f"] if p["energy_f"] > 0 else None,
                         "retain_share": rr / p["energy_r"] if p["energy_r"] > 0 else None,
                         "displacement": math.sqrt(max(disp2, 0.0))}
        tot["forget"] += rf
        tot["share_sum"] += rf / p["energy_f"] if p["energy_f"] > 0 else 0.0
        tot["retain"] += rr
        tot["energy_f"] += p["energy_f"]
        tot["energy_r"] += p["energy_r"]
        tot["disp2"] += disp2
    return {"per_target": per_target, "global": {
        "gamma": gamma, "s_min": s_min, "floor": s_min, "beta": None, "threshold": math.exp(log_t), "budget": budget,
        "unit": unit,
        "reached": bool(reached), "removed_in_unit": got, "forget_share_sum": tot["share_sum"],
        "directions": sum(r["directions"] for r in per_target.values()), "of": sum(r["of"] for r in per_target.values()),
        "forget_energy_removed": tot["forget"], "retain_energy_removed": tot["retain"],
        "forget_share": tot["forget"] / tot["energy_f"] if tot["energy_f"] > 0 else None,
        "retain_share": tot["retain"] / tot["energy_r"] if tot["energy_r"] > 0 else None,
        "displacement": math.sqrt(max(tot["disp2"], 0.0))}}


def delta_from_coef(part: dict, weight: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    """The Delta W (float64) of contracting every direction of one target by its ``coef`` (zero leaves it)."""
    w = weight.reshape(weight.shape[0], -1).double()
    keep = (coef > 0).nonzero().flatten()
    c = coef[keep].to(w.device, w.dtype)
    d = part["directions"][:, keep].to(w.device)
    if part["side"] == "output":
        delta = -(d * c) @ (d.transpose(0, 1) @ w)
    else:
        delta = -((w @ d) * c) @ d.transpose(0, 1)
    return delta.reshape(weight.shape)


def global_delta(part: dict, weight: torch.Tensor, directions: int, beta: float,
                 beta_last: Optional[float]) -> torch.Tensor:
    """The Delta W (float64, on the device of ``weight``) of one target's chosen prefix: ``directions`` directions at
    ``beta``, the last at ``beta_last`` when given."""
    w = weight.reshape(weight.shape[0], -1).double()
    if "coef" in part:  # per-direction coefficients folded into the selection, which ran at beta 1
        base = part["coef"][:directions].to(w.device, w.dtype)
        coef = base * beta
        if beta_last is not None and directions > 0:
            coef[-1] = base[-1] * beta_last
    else:
        coef = torch.full((directions,), beta, dtype=w.dtype, device=w.device)
        if beta_last is not None and directions > 0:
            coef[-1] = beta_last
    d = part["directions"][:, :directions].to(w.device)
    if part["side"] == "output":
        delta = -(d * coef) @ (d.transpose(0, 1) @ w)
    else:
        delta = -((w @ d) * coef) @ d.transpose(0, 1)
    return delta.reshape(weight.shape)
