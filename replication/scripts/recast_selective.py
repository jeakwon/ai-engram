"""Selective input contrast with retain protection (extension, user 2026-11-03).

The contrast of trace-normalised forget and retain statistics ranks an input direction by a ratio,
``lambda = v^T A_f v / v^T (A_r + rho / n I) v``, so a direction strong in both sets (shares f 100, r 25) ranks like a
weak one (4, 1), and a budget-driven selection keeps taking directions in a target whose selectivity has ended. Here,
per target W (out x in), the removed input directions V (in x k) are chosen by

1. retain protection, a protected span B (orthonormal columns) that V stays orthogonal to:

   * ``prefix``: the inputs of the constant prompt prefix. The chat template's leading tokens (BOS, the system prompt,
     the user header) are the same in every sequence and, the attention being causal, so are their inputs to every
     target; with their span protected, the edit leaves the prefix's computation in this target exactly unchanged (the
     BOS token's massive activations among it), which the answer-token statistics cannot see;
   * ``kappa``: the eigenvectors of ``R = C_r / tr C_r`` (C_r the retain input covariance, by default over every
     non-prefix token the edit acts on) with eigenvalue above ``kappa / n``; every unit v orthogonal to them has
     ``v^T R v <= kappa / n``, kappa times the retain share of an average direction (1 / n);

2. selectivity: with ``P = I - B B^T`` the pencil ``(P A_f P) v = lambda (P A_r P + rho / n I) v``, A = C / tr C of the
   full statistics (lambda keeps the units of the unprotected contrast; eigenvectors with lambda > 0 are orthogonal to
   B), and the directions with ``lambda >= lambda_star``: one threshold for every target and no budget, so a target
   without selective directions loses none. Columns orthonormalised in lambda order.

The edit is ``W <- W - (1 - s) (W V) V^T`` (input side, floor s). The driver ``recast_selective_tofu.py`` adds the loss
guard (each target's edit measured by inference on forget and retain answers) and saves the edited model for phase two.
``nonprefix_labels`` (user 2026-11-05) puts the retain loss of the gradient statistics on the question and answer
tokens. ``leave_one_out`` (user 2026-11-04) measures each target's contribution within the joint edit instead of
alone, and can guard on it.

The same protection applies to the contract's plain restricted warm start (``plain_restricted_update``, user
2026-11-04): the forget gradient's input side is projected off the protected span before its SVD, so the restricted
update leaves the prefix computation (and, with kappa, the strongest retain input directions) untouched. The same
projection applies to the gradient-contrast basis at the matched rank (``contrast_restricted_update``: both mean
gradients projected) and to the per-example moment basis (``moment_restricted_update``: every per-example gradient
projected, its moments from ``collect_example_moments``), user 2026-11-04.
"""

from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch import nn

POSITION_CLASSES = ("prefix", "prompt", "answer", "nonprefix")


def nonprefix_labels(input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor,
                     prefix_len: int) -> torch.Tensor:
    """Labels with the loss on every non-prefix token, question and answer (the token ids there, -100 elsewhere): the
    retain loss of the gradient statistics over S2's retain tokens (user 2026-11-05)."""
    m = position_masks(attention_mask, labels, prefix_len)["nonprefix"]
    return torch.where(m, input_ids, torch.full_like(input_ids, -100))


def common_prefix_length(sequences: Iterable[Sequence[int]]) -> int:
    """Length of the longest prefix of token ids shared by every sequence."""
    first, n = None, 0
    for s in sequences:
        s = list(s)
        if first is None:
            first, n = s, len(s)
            continue
        i, m = 0, min(n, len(s))
        while i < m and s[i] == first[i]:
            i += 1
        n = i
    if first is None:
        raise ValueError("no sequences")
    return n


def position_masks(attention_mask: torch.Tensor, labels: torch.Tensor, prefix_len: int) -> Dict[str, torch.Tensor]:
    """Boolean (batch, seq) masks of right-padded sequences: ``prefix`` (the first prefix_len positions, whatever their
    attention mask: the OpenUnlearning collator pads with the EOS id, so it also masks the template's own <|eot_id|>
    tokens), ``answer`` (labels != -100), ``nonprefix`` (attended or answer, not prefix) and ``prompt`` (nonprefix,
    not answer)."""
    if prefix_len < 0 or prefix_len > attention_mask.shape[1]:
        raise ValueError(f"prefix_len must be in [0, {attention_mask.shape[1]}], got {prefix_len}")
    pos = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0)
    prefix = (pos < prefix_len).expand(attention_mask.shape)
    answer = labels != -100
    if bool((prefix & answer).any()):
        raise ValueError("answer tokens inside the prefix")
    nonprefix = (attention_mask.bool() | answer) & ~prefix
    return {"prefix": prefix, "prompt": nonprefix & ~answer, "answer": answer, "nonprefix": nonprefix}


@torch.no_grad()
def collect_position_statistics(
    model: nn.Module,
    loader: Iterable,
    targets: Dict[str, nn.Linear],
    batch_fn,
    prefix_len: int,
    classes: Sequence[str] = ("answer", "nonprefix"),
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, int], Dict[str, torch.Tensor], Dict[str, float]]:
    """Per target: the uncentred input covariance ``E[x x^T]`` over the tokens of each position class (float64 sums,
    returned as float32 on the CPU), the token count per class, the inputs at the prefix positions (prefix_len x in,
    from the first sequence) and the largest relative deviation of any sequence's prefix inputs from those (the check
    that the prefix inputs are one fixed set). Training mode is restored afterwards."""
    unknown = set(classes) - set(POSITION_CLASSES)
    if unknown:
        raise ValueError(f"unknown position classes {sorted(unknown)}")
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    sums = {n: {c: torch.zeros(t.in_features, t.in_features, dtype=torch.float64, device=t.weight.device)
                for c in classes} for n, t in targets.items()}
    prefix_inputs: Dict[str, torch.Tensor] = {}
    deviation = {n: 0.0 for n in targets}
    state = {"masks": None}

    def hook(name):
        def fn(module, args):
            x = args[0]
            for c in classes:
                xc = x[state["masks"][c]].double()
                sums[name][c].addmm_(xc.transpose(0, 1), xc)
            if prefix_len > 0:
                p = x[:, :prefix_len].float()
                if name not in prefix_inputs:
                    prefix_inputs[name] = p[0].clone()
                ref = prefix_inputs[name]
                dev = float((p - ref).abs().max() / ref.abs().max().clamp_min(1e-30))
                deviation[name] = max(deviation[name], dev)
        return fn

    handles = [t.register_forward_pre_hook(hook(n)) for n, t in targets.items()]
    counts = {c: 0 for c in classes}
    try:
        for batch in loader:
            inputs = batch_fn(batch)
            state["masks"] = position_masks(inputs["attention_mask"], inputs["labels"], prefix_len)
            for c in classes:
                counts[c] += int(state["masks"][c].sum())
            model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    finally:
        for h in handles:
            h.remove()
        for m, mode in modes:
            m.train(mode)
    if any(v == 0 for v in counts.values()):
        raise ValueError(f"a position class has no tokens: {counts}")
    covs = {n: {c: (s / float(counts[c])).float().cpu() for c, s in per.items()} for n, per in sums.items()}
    return covs, counts, {n: p.cpu() for n, p in prefix_inputs.items()}, deviation


def retain_spectrum(retain_cov: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eigenvalues and eigenvectors of ``retain_cov / tr retain_cov`` (float64), for ``protected_span``."""
    r = retain_cov.double()
    r = r / torch.trace(r)
    return torch.linalg.eigh(0.5 * (r + r.transpose(0, 1)))


def protected_span(n: int, retain_cov: Optional[torch.Tensor] = None, kappa: Optional[float] = None,
                   prefix_inputs: Optional[torch.Tensor] = None, rtol: float = 1e-6,
                   spectrum: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, dict]:
    """Orthonormal columns (n x p, float64) spanning the protected directions: the eigenvectors of
    ``retain_cov / tr retain_cov`` with eigenvalue above ``kappa / n`` (with ``kappa``; ``spectrum`` from
    ``retain_spectrum`` instead of ``retain_cov`` saves the decomposition) and the row span of ``prefix_inputs``
    (singular values above ``rtol`` times the largest), with the count each part protects and the share of the retain
    input energy the kappa part holds."""
    parts, info = [], {"kappa": kappa, "kappa_directions": 0, "kappa_retain_share": 0.0, "prefix_rank": 0}
    device = None
    if kappa is not None:
        if (retain_cov is None and spectrum is None) or kappa <= 0:
            raise ValueError("kappa needs a retain covariance and must be positive")
        lam, vec = spectrum if spectrum is not None else retain_spectrum(retain_cov)
        keep = lam > kappa / n
        parts.append(vec[:, keep])
        info["kappa_directions"] = int(keep.sum())
        info["kappa_retain_share"] = float(lam[keep].sum())
        device = vec.device
    if prefix_inputs is not None and prefix_inputs.shape[0] > 0:
        x = prefix_inputs.double().to(device) if device is not None else prefix_inputs.double()
        _, s, vh = torch.linalg.svd(x, full_matrices=False)
        rank = int((s > rtol * s[0]).sum()) if float(s[0]) > 0 else 0
        parts.append(vh[:rank].transpose(0, 1))
        info["prefix_rank"] = rank
        device = x.device
    if not parts:
        info["protected"] = 0
        return torch.zeros(n, 0, dtype=torch.float64), info
    m = torch.cat([p.to(device) for p in parts], 1)
    if m.shape[1] == 0:
        info["protected"] = 0
        return m, info
    u, s, _ = torch.linalg.svd(m, full_matrices=False)
    rank = int((s > rtol * s.max()).sum())
    info["protected"] = rank
    return u[:, :rank], info


def selective_directions(cov_forget: torch.Tensor, cov_retain: torch.Tensor, protected: torch.Tensor, rho: float,
                         lambda_star: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Orthonormal columns (n x k, float64) spanning, in order, the leading eigenvectors of
    ``(P A_f P) v = lambda (P A_r P + rho / n I) v`` with ``A = C / tr C`` and ``P = I - B B^T`` (B = ``protected``),
    k the number of eigenvalues >= ``lambda_star`` (all n without it), and every eigenvalue, descending. Without
    protection this is ``recast_direction_families.ordered_directions``."""
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")
    a, b = cov_forget.double(), cov_retain.double()
    a, b = a / torch.trace(a), b / torch.trace(b)
    n = a.shape[0]
    eye = torch.eye(n, dtype=a.dtype, device=a.device)
    bp = protected.to(a.device, a.dtype)
    if bp.shape[1] > 0:
        proj = eye - bp @ bp.transpose(0, 1)
        a, b = proj @ a @ proj, proj @ b @ proj
    chol = torch.linalg.cholesky(0.5 * (b + b.transpose(0, 1)) + (rho / n) * eye)
    linv = torch.linalg.solve_triangular(chol, eye, upper=False)
    m = linv @ a @ linv.transpose(0, 1)
    lam, q = torch.linalg.eigh(0.5 * (m + m.transpose(0, 1)))
    idx = torch.argsort(lam, descending=True)
    lam = lam[idx]
    k = n if lambda_star is None else int((lam >= lambda_star).sum())
    if k == 0:
        return torch.zeros(n, 0, dtype=a.dtype, device=a.device), lam
    vecs = torch.linalg.solve_triangular(chol.transpose(0, 1), q[:, idx[:k]], upper=True)
    if bp.shape[1] > 0:  # exact in theory (lambda > 0); removes round-off along the protected span
        vecs = vecs - bp @ (bp.transpose(0, 1) @ vecs)
    u, r = torch.linalg.qr(vecs)
    d = torch.diagonal(r)
    return u * torch.where(d < 0, -torch.ones_like(d), torch.ones_like(d)), lam


def _off_span(m: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``m (I - B B^T)`` in float64, cast back to m's dtype: the input side of m projected off the span of B."""
    m64 = m.double()
    return (m64 - (m64 @ b) @ b.transpose(0, 1)).to(m.dtype)


def _protected_source(weight: torch.Tensor, grad_forget: torch.Tensor, protected: Optional[torch.Tensor]):
    """float32 weight, float32 forget gradient, B (float64 on the gradient's device, None without protection) and the
    forget gradient with its input side projected off B (the source of every protected restricted basis)."""
    w32 = weight.detach().float()
    g = grad_forget.detach().float()
    if protected is None or protected.shape[1] == 0:
        return w32, g, None, g
    b = protected.to(g.device, torch.float64)
    return w32, g, b, _off_span(g, b)


def _restricted_delta(w32: torch.Tensor, g: torch.Tensor, grad_retain: torch.Tensor, basis, floor: float,
                      b: Optional[torch.Tensor], rep: dict) -> Tuple[torch.Tensor, dict]:
    """The restricted_shrink update of ``recast_core.warm_start`` for a given basis (full-space alignment): the rows
    shrunk to ``s_i = max(cos(W_i, -G_r,i), floor)``, the update ``U U^T ((S - I) W) Vh^T Vh``; with B the round-off of
    Vh along B removed first (exact in theory; a row with less than half its norm outside B is an error)."""
    from recast_core import COS_EPS, project, row_alignment, shrink_factors

    if not basis.active:
        rep.update({"kept_forget_gradient_energy": 0.0, "shrink_at_floor": None, "weight_rel_change": 0.0})
        return torch.zeros_like(w32), rep
    vh = basis.vh
    if b is not None:
        vh = _off_span(vh.to(b.device), b)
        if float(vh.norm(dim=1).min()) < 0.5:
            raise ValueError("a basis direction lies in the protected span")
    a = row_alignment(w32, grad_retain)
    s = shrink_factors(a, floor)
    delta = project((s - 1.0).unsqueeze(1) * w32, basis.u, vh)
    rep.update({"kept_forget_gradient_energy": float(project(g, basis.u, vh).square().sum() / g.square().sum()),
                "shrink_at_floor": float((a <= floor).float().mean()),
                "weight_rel_change": float(delta.norm() / (w32.norm() + COS_EPS))})
    return delta, rep


def _basis_report(basis, g: torch.Tensor, src: torch.Tensor) -> dict:
    return {"rank": basis.rank, "rank_in": basis.rank_in if basis.rank_in is not None else basis.rank,
            "total_rank": basis.total_rank, "evr_of_source": basis.evr,
            "source_share_after_projection": float(src.square().sum() / g.square().sum().clamp_min(1e-30))}


def plain_restricted_update(weight: torch.Tensor, grad_forget: torch.Tensor, grad_retain: torch.Tensor, evr: float,
                            floor: float, protected: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
    """The contract's restricted warm start with the plain basis, as ``recast_core.warm_start`` computes it for
    ``restricted_shrink`` (forget_only source, full-space alignment): U, Vh from ``svd_basis`` of the mean forget
    gradient at the EVR, the rows shrunk to ``s_i = max(cos(W_i, -G_r,i), floor)``, the update
    ``U U^T ((S - I) W) Vh^T Vh``. With ``protected`` (B, orthonormal columns: the prefix inputs' span, the kappa-protected
    retain directions) the forget gradient's input side is projected off B first, ``G_f (I - B B^T)``, so that Vh is
    orthogonal to B and the update leaves ``W x`` unchanged for every x in B (the rank comes from the EVR of the
    projected gradient). Returns the update and its report."""
    from recast_core import svd_basis

    w32, g, b, src = _protected_source(weight, grad_forget, protected)
    basis = svd_basis(src, evr)
    return _restricted_delta(w32, g, grad_retain, basis, floor, b, _basis_report(basis, g, src))


def contrast_restricted_update(weight: torch.Tensor, grad_forget: torch.Tensor, grad_retain: torch.Tensor, evr: float,
                               floor: float, rho: float, sides: str = "both",
                               protected: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
    """The restricted warm start with the forget-vs-retain gradient-contrast basis at the matched rank, as
    ``recast_core.warm_start`` computes it for ``recast_tofu.py --restriction_gradient_contrast <sides>
    --restriction_rank_multiplier 1`` (user 2026-11-01): both sides get the rank of the plain basis at the EVR, the
    bases from ``gradient_contrast_basis`` at ``rho``. With ``protected`` (user 2026-11-04) the input sides of both
    mean gradients are projected off B first (``G (I - B B^T)``), and the rank is that of the plain basis of the
    projected forget gradient: the pencil's directions with lambda > 0 then lie outside B, so the update leaves
    ``W x`` unchanged for every x in B. Returns the update and its report."""
    from recast_core import gradient_contrast_basis, svd_basis

    w32, g, b, src = _protected_source(weight, grad_forget, protected)
    src_r = grad_retain.detach().float() if b is None else _off_span(grad_retain.detach().float(), b)
    rank = svd_basis(src, evr).rank
    basis = gradient_contrast_basis(src, src_r, rank, rho, sides)
    rep = _basis_report(basis, g, src)
    rep["reference_rank"] = rank
    return _restricted_delta(w32, g, grad_retain, basis, floor, b, rep)


def moment_restricted_update(weight: torch.Tensor, grad_forget: torch.Tensor, grad_retain: torch.Tensor,
                             moments: Tuple[torch.Tensor, torch.Tensor], evr: float, floor: float,
                             protected: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
    """The restricted warm start with the per-example forget-gradient moment basis and its own-spectrum ranks, as
    ``recast_core.warm_start`` computes it for ``recast_tofu.py --restriction_gradient_moment per_example`` without a
    rank multiplier (user 2026-11-02): U and V the leading eigenvectors of ``E[g g^T]`` and ``E[g^T g]``, each side at
    its own EVR rank. With ``protected`` (user 2026-11-04) the per-example gradients are projected off B, g_e (I - B
    B^T): ``moments`` must then be the output moment of the projected gradients, ``E[g P g^T]``
    (``collect_example_moments``), with the plain input moment ``E[g^T g]`` (projected here, ``P E[g^T g] P``), so V
    lies outside B and the update leaves ``W x`` unchanged for every x in B. Returns the update and its report."""
    from recast_core import gradient_moment_basis

    w32, g, b, src = _protected_source(weight, grad_forget, protected)
    m_out, m_in = moments
    if b is not None:
        m_in = m_in.to(b.device, torch.float64)
        m_in = _off_span(_off_span(m_in, b).transpose(0, 1), b)
    basis = gradient_moment_basis(src, m_out, m_in, energy_threshold=evr)
    return _restricted_delta(w32, g, grad_retain, basis, floor, b, _basis_report(basis, g, src))


def collect_example_moments(
    model: nn.Module,
    loader: Iterable,
    targets: Dict[str, nn.Linear],
    batch_fn,
    protected: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]], int]:
    """Per target the per-example gradient moments ``(E[g g^T], E[g^T g])`` exactly as
    ``recast_core.collect_example_gradient_moments`` computes them, and for every target with a protected span B in
    ``protected`` (orthonormal columns) also ``E[g P g^T]``, P = I - B B^T, the output moment of the per-example
    gradients projected off B (g_e P = sum_t delta_t (P x_t)^T: the inputs projected in float64, the products in float32
    as for the plain moments); None for the others. Modes and ``requires_grad`` flags are restored."""
    protected = protected or {}
    modes = [(m, m.training) for m in model.modules()]
    flags = [(p, p.requires_grad) for p in model.parameters()]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    names = list(targets)
    for n in names:
        targets[n].weight.requires_grad_(True)
    spans = {n: protected[n].to(targets[n].weight.device, torch.float64) for n in names
             if n in protected and protected[n].shape[1] > 0}
    saved: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    hooks = [targets[n].register_forward_hook(lambda mod, inp, out, n=n: saved.__setitem__(n, (inp[0].detach(), out)))
             for n in names]

    def zeros(k, n):
        return torch.zeros(k, k, dtype=torch.float64, device=targets[n].weight.device)

    sums = {n: [zeros(targets[n].out_features, n), zeros(targets[n].in_features, n),
                zeros(targets[n].out_features, n) if n in spans else None] for n in names}
    count = 0
    try:
        for batch in loader:
            inputs = batch_fn(batch)
            size = int(inputs["input_ids"].shape[0])
            loss = model(**inputs).loss
            deltas = torch.autograd.grad(loss, [saved[n][1] for n in names], allow_unused=True)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            for n, d in zip(names, deltas):
                if d is None:
                    continue
                x = saved[n][0].float() * mask
                d = d.float() * mask * float(size)
                out_sum, in_sum, out_p = sums[n]
                out_sum.add_(d.flatten(0, -2).transpose(0, 1) @ (x @ x.transpose(-1, -2) @ d).flatten(0, -2))
                in_sum.add_(x.flatten(0, -2).transpose(0, 1) @ (d @ d.transpose(-1, -2) @ x).flatten(0, -2))
                if out_p is not None:
                    xp = _off_span(x, spans[n])
                    out_p.add_(d.flatten(0, -2).transpose(0, 1) @ (xp @ xp.transpose(-1, -2) @ d).flatten(0, -2))
            saved.clear()
            count += size
    finally:
        for h in hooks:
            h.remove()
        saved.clear()
        for p, flag in flags:
            p.requires_grad_(flag)
        for m, mode in modes:
            m.train(mode)
    if count == 0:
        raise ValueError("empty source loader")
    return {n: (o / float(count), i / float(count), None if op is None else op / float(count))
            for n, (o, i, op) in sums.items()}, count


def guard_passes(change: dict, ratio: float) -> bool:
    """The loss guard's test of a change {d_forget, d_retain}: forget rises, and retain falls or rises at most
    ``1 / ratio`` as much."""
    df, dr = change["d_forget"], change["d_retain"]
    return df > 0 and (dr <= 0 or df / dr >= ratio)


def leave_one_out(chosen: Dict[str, torch.Tensor], measure, ratio: float = 0.0, steps: int = 4):
    """Each target's contribution to the joint edit (user 2026-11-04): the joint change of every target's directions
    minus the change with that target left out, the others as chosen. ``measure`` maps {name: V} to a dict with
    ``d_forget`` and ``d_retain`` (changes against the unedited model). With ``ratio > 0`` a target whose contribution
    fails ``guard_passes`` keeps the leading half of its directions, measured again against the same others, at most
    ``steps`` times, else none; each decision is taken against the others as chosen, and all apply together.
    Returns the kept count per target and the report (the full joint change, per target the change without it and
    every trial's contribution)."""
    full = measure(chosen)
    kept, report = {}, {"full": full, "targets": {}}
    for n, v in chosen.items():
        k = int(v.shape[1])
        if k == 0:
            kept[n] = 0
            report["targets"][n] = {"without": full, "trials": [], "kept": 0}
            continue
        others = {m: w for m, w in chosen.items() if m != n}
        without = measure(others)
        trials, cur, step = [], full, 0
        while True:
            c = {"directions": k, "d_forget": cur["d_forget"] - without["d_forget"],
                 "d_retain": cur["d_retain"] - without["d_retain"]}
            c["ratio"] = c["d_forget"] / c["d_retain"] if c["d_retain"] > 0 else None
            trials.append(c)
            if ratio <= 0 or guard_passes(c, ratio):
                break
            if step == steps or k // 2 == 0:
                k = 0
                break
            k //= 2
            step += 1
            cur = measure({**others, n: v[:, :k]})
        kept[n] = k
        report["targets"][n] = {"without": without, "trials": trials, "kept": k}
    return kept, report


def input_shrink(weight: torch.Tensor, directions: torch.Tensor, floor: float) -> torch.Tensor:
    """``W - (1 - s) (W V) V^T``: the input directions V (orthonormal columns) keep the share s of W's response."""
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"floor must be in [0, 1], got {floor}")
    v = directions.to(weight.device, weight.dtype)
    return weight - (1.0 - floor) * (weight @ v) @ v.transpose(0, 1)


def direction_report(directions: torch.Tensor, cov_forget: torch.Tensor, cov_retain: torch.Tensor,
                     prefix_inputs: Optional[torch.Tensor] = None) -> dict:
    """Shares of the (trace-normalised) forget and retain input energy along V, the largest retain share of a single
    column in units of the average share 1 / n, and the share of the prefix inputs' energy along V."""
    n, k = directions.shape
    rep = {"directions": int(k), "forget_share": 0.0, "retain_share": 0.0, "max_retain_share_x_n": 0.0,
           "prefix_share": None}
    if prefix_inputs is not None and prefix_inputs.shape[0] > 0:
        rep["prefix_share"] = 0.0
    if k == 0:
        return rep
    v = directions.double()
    a = cov_forget.double().to(v.device)
    b = cov_retain.double().to(v.device)
    rep["forget_share"] = float(torch.trace(v.transpose(0, 1) @ a @ v) / torch.trace(a))
    per = (v * (b @ v)).sum(0) / torch.trace(b)
    rep["retain_share"] = float(per.sum())
    rep["max_retain_share_x_n"] = float(per.max() * n)
    if prefix_inputs is not None and prefix_inputs.shape[0] > 0:
        x = prefix_inputs.double().to(v.device)
        rep["prefix_share"] = float((x @ v).square().sum() / x.square().sum())
    return rep
