"""Learned gates on forget-specific input directions (extension, user 2026-11-03).

The user's Classification idea (unlearning-svd ``recast_svd_gates_fo``: a learned gate on every SVD component of a
filter instead of a solver), adapted to this port: the gates act on the input directions that the global
input-contrast selection removes whole (``recast_global_families``), and their strengths are learned on the
training objective instead of being set by the lambda order, an energy budget and one floor.

Parametrisation. For a target W0 (out x in, its Init Fine. weight) a fixed orthonormal frame V = [v_1 .. v_k] of input
directions is computed once, and each direction gets a gate s_j in [0, 1], the share of its part that stays (1 - s_j
its contraction); every gate starts at exactly 1:

    W_eff = W0 - sum_j (1 - s_j) (W0 v_j) v_j^T = W0 - A diag(1 - s) V^T,    A = W0 V.

The parts (W0 v_j) v_j^T are Frobenius-orthogonal (V^T V = I), so ``||W_eff - W0||_F^2 = sum_j e_j (1 - s_j)^2`` with
``e_j = ||W0 v_j||^2``, and ``dJ/ds_j = <dJ/dW_eff, (W0 v_j) v_j^T> = a_j^T G v_j``. The gated forward is computed on
the activation side, ``y = W0 x - A ((1 - s) * (V^T x))`` (a forward hook on the target): at s = 1 the correction is
exactly zero and the output is bitwise that of W0. Bases (``GATE_BASES``), from the uncentred input covariances C_f,
C_r over answer tokens at Init Fine.: ``input_contrast``, the lambda-leading directions of
``(C_f / tr) v = lambda (C_r / tr + rho / in I) v`` (``ordered_directions``, the family of the global selection,
orthonormalised by a QR in lambda order: the span of the first k columns is that of the k leading eigenvectors, but
column j > 1 is not eigenvector j and its own forget / retain ratio is not lambda_j); ``input_contrast_dual``
(user question 2026-11-03 on that QR) the same eigenvectors x_j without the QR, each gate removing exactly the
coefficient of x_j: with X^T B X = I (B = C_r / tr + rho / in I) every input is ``z = sum_j (B x_j)(x_j^T z)``, and the
gate j removes ``(B x_j)(x_j^T z)``, so ``W_eff = W0 - sum_j (1 - s_j) W0 B x_j x_j^T`` (an oblique part: read
direction v_j = x_j / ||x_j||, write direction u_j = ||x_j|| B x_j, A = W0 U). The coefficients x_j^T z are uncorrelated
under C_f and under B, so the forget and the B-retain output energy the gates remove are sums over the gates, gate j
with the ratio exactly lambda_j; the displacement is then ``c^T ((A^T A) * (V^T V)) c`` (c = 1 - s);
``protected_contrast`` and ``protected_contrast_dual`` (user 2026-11-04, from the selective contrast of
``recast_selective.py``): the same two with a protected span B, the inputs of the constant prompt prefix (BOS, the
system prompt, the user header; the BOS token's massive activations among them), the pencil taken in its complement,
``(P A_f P) x = lambda (P A_r P + rho / in I) x`` with P = I - B B^T, so that every read direction is orthogonal to B
and the gates leave the prefix's computation unchanged (the caller passes the retain covariance over every non-prefix
token, question and answer); ``forget_input``, the leading eigenvectors of C_f alone (no retain contrast);
``random``, a seeded random orthonormal frame (the control).

Objective (``train_gates``), every model parameter frozen, the gates (float64) trained by Adam or plain SGD with a
constant learning rate and clipped to [0, 1] after every step:

    J = L_R + alpha L_F + gamma O + beta R,

L_R the retain answer-token NLL; O = sum_j s_j, the open gates (user 2026-11-03: instead of, or with, the forget term
every gate is pushed shut at the same price gamma, and only the retain loss holds open the ones retain needs; to first
order gate j closes when closing it costs less than gamma in L_R, -dL_R/ds_j < gamma); L_F the forget answer-token
unlikelihood ``-log(1 - p(y_t | x, y_<t))`` in the stable
form ``logsumexp_c z_c - logsumexp_{c != y} z_c`` (>= 0, finite for any logits; its gradient on the true logit is p_y,
so it pushes hardest on memorised tokens and saturates as they are forgotten); each a mean over the answer tokens of
the whole accumulation window; ``R = sum_m sum_j e_mj (1 - s_mj)^2 / sum_m ||W0_m||_F^2``, the squared displacement of
the gated weights relative to their energy. Epochs are anchored on the forget split as OpenUnlearning's
ForgetRetainDataset (anchor forget): per epoch a seeded permutation of the forget items, each paired with a uniformly
random retain item. The phase runs to convergence (user 2026-11-03), capped at a number of epochs: after every epoch J
is evaluated on a fixed probe (every forget item and a seeded subset of retain items), and the phase stops once, over
the last ``convergence_epochs`` epochs, the probe J fell by at most ``tol`` (relative) and the gates no longer drift:
``||s_end - s_start|| / sum_t ||s_t - s_{t-1}||`` (1 when every update goes one way, about 1 / sqrt(steps) for SGD and
4.4 / sqrt(steps) for Adam's momentum 0.9 when the updates are noise around a fixed point) is at most ``drift_max``.
Each step logs the gradient of every term of J, the update and its reversals against the previous one (oscillation);
each epoch the probe, the net change, the path length and the drift. A fixed schedule instead (user 2026-11-05: a
short, practical gate phase; ``convergence_epochs`` 0): exactly ``max_epochs`` epochs, no convergence rule, the
learning rate constant or falling linearly to 0 over the steps (``lr_schedule``: the Adam noise around the end point
dies with the step size instead of a trailing window proving it), the probe after every ``probe_every``-th and after
the last epoch only. ``materialize`` then writes W_eff into the native weights once; phase two trains every parameter.
"""

import math
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from recast_direction_families import ordered_directions

GATE_BASES = ("input_contrast", "input_contrast_dual", "protected_contrast", "protected_contrast_dual", "forget_input",
              "random")
PROTECTED_BASES = ("protected_contrast", "protected_contrast_dual")  # the bases with a protected span
CONTRAST_BASES = ("input_contrast", "input_contrast_dual") + PROTECTED_BASES  # the bases with a ridge rho
GATE_OPTIMIZERS = ("adam", "sgd")
GATE_LR_SCHEDULES = ("constant", "linear")  # linear: lr (T - t + 1) / T at step t of T, a fixed schedule only
IGNORE_INDEX = -100
MATERIALIZE_RTOL = 1.0e-3  # max |logit difference| / max |logit| allowed between the gated and the materialized model


def contrast_eigenvectors(cov_forget: torch.Tensor, cov_retain: torch.Tensor, rho: float,
                          protected: Optional[torch.Tensor] = None):
    """The generalised eigenvectors X of ``(C_f / tr) x = lambda B x``, ``B = C_r / tr + rho / n I``, B-orthonormal
    (``X^T B X = I``), as columns with lambda descending, the lambdas and B (float64). With ``protected`` (n x p
    orthonormal columns) both traces-normalised covariances are first projected by P = I - protected protected^T, so
    the eigenvectors with lambda > 0 are orthogonal to the protected span (exactly in theory; the round-off along it is
    removed) and lambda keeps the units of the unprotected contrast."""
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")
    a, b = cov_forget.double(), cov_retain.to(cov_forget.device).double()
    a, b = a / torch.trace(a), b / torch.trace(b)
    n = a.shape[0]
    eye = torch.eye(n, dtype=a.dtype, device=a.device)
    bp = None
    if protected is not None and protected.shape[1] > 0:
        bp = protected.to(a.device, a.dtype)
        proj = eye - bp @ bp.transpose(0, 1)
        a, b = proj @ a @ proj, proj @ b @ proj
        a, b = 0.5 * (a + a.transpose(0, 1)), 0.5 * (b + b.transpose(0, 1))
    b = b + (rho / n) * eye  # unchanged for the unprotected bases (their earlier runs reproduce bitwise)
    chol = torch.linalg.cholesky(b)
    linv = torch.linalg.solve_triangular(chol, eye, upper=False)
    m = linv @ a @ linv.transpose(0, 1)
    lam, q = torch.linalg.eigh(0.5 * (m + m.transpose(0, 1)))
    idx = torch.argsort(lam, descending=True)
    x = torch.linalg.solve_triangular(chol.transpose(0, 1), q[:, idx], upper=True)
    if bp is not None:
        lam = lam[idx]
        pos = lam > 0
        x[:, pos] = x[:, pos] - bp @ (bp.transpose(0, 1) @ x[:, pos])
        return x, lam, b
    return x, lam[idx], b


def gate_directions(cov_forget: torch.Tensor, cov_retain: Optional[torch.Tensor], basis: str, k: int,
                    rho: Optional[float] = None, generator: Optional[torch.Generator] = None,
                    protected: Optional[torch.Tensor] = None, lambda_star: Optional[float] = None):
    """The k leading input directions of one target's ``basis`` (k clipped to the input width): the read directions V
    as float64 columns (orthonormal except for ``input_contrast_dual``, whose columns have unit norm), their scores
    (lambda for the contrast bases, the share of tr(C_f) for forget_input, NaN for random) and the write directions U
    of the dual basis (``W_eff = W0 - W0 U diag(1 - s) V^T``; None for the orthonormal bases, U = V)."""
    if basis not in GATE_BASES:
        raise ValueError(f"basis must be one of {GATE_BASES}, got {basis!r}")
    if (basis in CONTRAST_BASES) != (rho is not None) or (rho is not None and rho <= 0):
        raise ValueError(f"a positive rho goes with, and only with, the bases {CONTRAST_BASES}")
    if (basis == "random") != (generator is not None):
        raise ValueError("a generator goes with, and only with, the random basis")
    if (basis in PROTECTED_BASES) != (protected is not None):
        raise ValueError(f"a protected span goes with, and only with, the bases {PROTECTED_BASES}")
    if lambda_star is not None and (basis not in CONTRAST_BASES or lambda_star <= 0):
        # lambda* > 0: lambda >= 0 would admit the protected span itself (lambda = 0 there)
        raise ValueError(f"a positive lambda_star goes only with the bases {CONTRAST_BASES}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    cf = cov_forget.double()
    n = cf.shape[0]
    k = min(int(k), n)
    if basis in CONTRAST_BASES and cov_retain is None:
        raise ValueError(f"the {basis} basis needs the retain covariance")
    def cap(lam: torch.Tensor) -> int:
        """k, or with lambda_star the directions with lambda >= lambda_star (lambda descending), at most k."""
        return k if lambda_star is None else min(k, int((lam >= lambda_star).sum()))

    if basis == "input_contrast":
        d, lam = ordered_directions(cf, cov_retain.to(cf.device).double(), rho, return_values=True)
        k = cap(lam)
        return d[:, :k], lam[:k], None
    if basis in ("input_contrast_dual", "protected_contrast_dual"):
        x, lam, b = contrast_eigenvectors(cf, cov_retain, rho, protected)
        k = cap(lam)
        x = x[:, :k]
        norms = x.norm(dim=0)
        return x / norms, lam[:k], (b @ x) * norms
    if basis == "protected_contrast":
        x, lam, _ = contrast_eigenvectors(cf, cov_retain, rho, protected)
        k = cap(lam)
        if k == 0:
            return x[:, :0], lam[:0], None
        q, r = torch.linalg.qr(x[:, :k])
        d = torch.diagonal(r)
        return q * torch.where(d < 0, -torch.ones_like(d), torch.ones_like(d)), lam[:k], None
    if basis == "forget_input":
        vals, vecs = torch.linalg.eigh(0.5 * (cf + cf.transpose(0, 1)))
        idx = torch.argsort(vals, descending=True)[:k]
        return vecs[:, idx], vals[idx] / torch.trace(cf), None
    q, r = torch.linalg.qr(torch.randn(n, k, generator=generator, dtype=torch.float64))
    d = torch.diagonal(r)
    q = q * torch.where(d < 0, -torch.ones_like(d), torch.ones_like(d))
    return q.to(cf.device), torch.full((k,), float("nan"), dtype=torch.float64), None


@torch.no_grad()
def column_specificity(v: torch.Tensor, cov_forget: torch.Tensor, cov_retain: torch.Tensor, rho: float,
                       scores: Optional[torch.Tensor] = None) -> dict:
    """How well the columns of V carry the contrast's order (user question 2026-11-03): each column's own ratio
    ``r_j = v_j^T (C_f / tr) v_j / v_j^T B v_j`` (B = C_r / tr + rho / n I; lambda_j for the eigenvectors themselves);
    with the scores (lambda) the Spearman correlation of r and lambda, the largest |r_j - lambda_j| / lambda_j, and the
    share of the columns with lambda > 1 whose own ratio is <= 1; and the off-diagonal share
    ``||offdiag(V^T (C_f / tr) V)||_F / ||V^T (C_f / tr) V||_F`` (0 when the columns' forget parts do not mix)."""
    a, b = cov_forget.double(), cov_retain.to(cov_forget.device).double()
    a, b = a / torch.trace(a), b / torch.trace(b)
    b = b + (rho / a.shape[0]) * torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
    v = v.to(a.device).double()
    fa = v.transpose(0, 1) @ a @ v
    own = torch.diagonal(fa) / ((b @ v) * v).sum(0)
    off = fa - torch.diag(torch.diagonal(fa))
    out = {"own_ratio": own.cpu(), "own_ratio_median": float(own.median()), "own_ratio_above_1": int((own > 1).sum()),
           "offdiag_share": float(off.norm() / fa.norm()) if float(fa.norm()) > 0 else 0.0}
    if scores is not None and bool(torch.isfinite(scores).all()):
        lam = scores.to(own.device).double()

        def ranks(x):
            return torch.argsort(torch.argsort(x)).double()

        ro, rl = ranks(own), ranks(lam)
        ro, rl = ro - ro.mean(), rl - rl.mean()
        out["spearman_with_lambda"] = float((ro @ rl) / (ro.norm() * rl.norm())) if own.numel() > 1 else None
        out["max_rel_gap_to_lambda"] = float(((own - lam).abs() / lam.abs().clamp_min(1e-300)).max())
        out["lambda_above_1"] = int((lam > 1).sum())
        out["lambda_above_1_own_at_most_1"] = int(((lam > 1) & (own <= 1)).sum())
    return out


def _answer_logits(logits: torch.Tensor, labels: torch.Tensor):
    """Next-token logits (float32) and targets at the answer positions: position t predicts labels[t + 1]."""
    targets = labels[:, 1:]
    mask = targets != IGNORE_INDEX
    return logits[:, :-1][mask].float(), targets[mask]


def unlikelihood_sum(logits: torch.Tensor, labels: torch.Tensor):
    """Sum over the answer tokens of ``-log(1 - p(y_t))`` as ``logsumexp_c z_c - logsumexp_{c != y} z_c``, and the
    number of answer tokens."""
    z, y = _answer_logits(logits, labels)
    every = torch.logsumexp(z, dim=-1)
    others = torch.logsumexp(z.scatter(1, y.unsqueeze(1), float("-inf")), dim=-1)
    return (every - others).sum(), int(y.numel())


def nll_sum(logits: torch.Tensor, labels: torch.Tensor):
    """Sum over the answer tokens of ``-log p(y_t)``, and the number of answer tokens."""
    z, y = _answer_logits(logits, labels)
    return F.cross_entropy(z, y, reduction="sum"), int(y.numel())


class DirectionGates(nn.Module):
    """Gates in [0, 1] on k fixed input directions of each target (module docstring), applied by forward hooks; the
    targets' weights stay W0 until ``materialize``. The targets are not submodules: the gates own no model weight."""

    def __init__(self, targets: Dict[str, nn.Linear], directions: Dict[str, torch.Tensor],
                 writes: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        writes = writes or {}
        self.names = list(directions)
        self._targets = {n: targets[n] for n in self.names}
        self.gates = nn.ParameterList()
        norm2 = 0.0
        for i, n in enumerate(self.names):
            w = targets[n].weight.detach()
            v = directions[n].to(w.device, torch.float64)
            u = writes[n].to(w.device, torch.float64) if n in writes else v
            if v.dim() != 2 or v.shape[0] != w.shape[1] or u.shape != v.shape:
                raise ValueError(f"{n}: directions of shape {tuple(v.shape)} (writes {tuple(u.shape)}) for a weight "
                                 f"of shape {tuple(w.shape)}")
            a = w.double() @ u
            self.register_buffer(f"v_{i}", v.float())
            self.register_buffer(f"a_{i}", a.float())
            # e_j = ||a_j||^2 ||v_j||^2, the squared norm of part j alone (float64); the parts of an oblique basis are
            # not Frobenius-orthogonal, so its displacement needs the Gram (A^T A) * (V^T V)
            self.register_buffer(f"e_{i}", a.square().sum(0) * v.square().sum(0))
            self.register_buffer(f"gram_{i}", (a.transpose(0, 1) @ a) * (v.transpose(0, 1) @ v) if n in writes else None)
            # float64: an SGD step lr * g below 6e-8 would vanish next to 1 in float32
            self.gates.append(nn.Parameter(torch.ones(v.shape[1], dtype=torch.float64, device=w.device)))
            norm2 += float(w.double().square().sum())
        self.norm2 = norm2
        self._handles = []

    def parts(self, i: int):
        """(V, A = W0 U, e) of target i (float32, float32, float64)."""
        return getattr(self, f"v_{i}"), getattr(self, f"a_{i}"), getattr(self, f"e_{i}")

    def displacement2(self, i: int, c: torch.Tensor) -> torch.Tensor:
        """||A diag(c) V^T||_F^2 of target i (float64): sum e c^2, or c^T Gram c for an oblique basis."""
        gram = getattr(self, f"gram_{i}")
        c = c.double()
        return (self.parts(i)[2] * c * c).sum() if gram is None else c @ (gram @ c)

    def openness(self) -> torch.Tensor:
        """O = sum of every gate (float64, differentiable): the open gates, the closing term of J."""
        return sum(g.double().sum() for g in self.gates)

    @torch.no_grad()
    def reset_(self) -> None:
        """Every gate back to 1 (the hooks then leave W0 exactly)."""
        for g in self.gates:
            g.fill_(1.0)

    def _hook(self, i: int):
        def fn(module, inputs, output):
            v, a, _ = self.parts(i)
            x = inputs[0]
            h = (x @ v.to(x.dtype)) * (1.0 - self.gates[i]).to(x.dtype)
            return output - (h @ a.to(h.dtype).transpose(0, 1)).to(output.dtype)
        return fn

    def attach(self) -> None:
        if self._handles:
            raise RuntimeError("the gates are already attached")
        self._handles = [self._targets[n].register_forward_hook(self._hook(i)) for i, n in enumerate(self.names)]

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def regulariser(self) -> torch.Tensor:
        """R = ||W_eff - W0||_F^2 over all targets / sum ||W0||_F^2 (sum_j e_j (1 - s_j)^2 for an orthonormal basis;
        float64, differentiable in the gates)."""
        total = 0.0
        for i in range(len(self.names)):
            total = total + self.displacement2(i, 1.0 - self.gates[i].double())
        return total / self.norm2

    @torch.no_grad()
    def clamp_(self) -> None:
        for g in self.gates:
            g.clamp_(0.0, 1.0)

    @torch.no_grad()
    def summary(self) -> Dict[str, dict]:
        """Per target: the gate distribution, sum(1 - s) and the squared displacement ||A diag(1 - s) V^T||^2."""
        out = {}
        for i, n in enumerate(self.names):
            s = self.gates[i].detach().double().cpu()
            e = self.parts(i)[2].cpu()
            c = 1.0 - s
            q = torch.quantile(s, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))
            out[n] = {"k": int(s.numel()), "mean": float(s.mean()), "min": float(s.min()), "q10": float(q[0]),
                      "q50": float(q[1]), "q90": float(q[2]), "below_0p5": int((s < 0.5).sum()),
                      "below_0p1": int((s < 0.1).sum()), "at_0": int((s == 0).sum()), "at_1": int((s == 1).sum()),
                      "contraction_sum": float(c.sum()),
                      "displacement2": float(self.displacement2(i, c.to(self.gates[i].device))),
                      "candidate_energy": float(e.sum())}
        return out

    @torch.no_grad()
    def materialize(self) -> Dict[str, dict]:
        """Detach the hooks and write W_eff = W0 - A diag(1 - s) V^T into every target (float64 sum, cast to the
        weight's dtype); per target ||W_eff - W0|| and its ratio to ||W0||."""
        self.detach()
        out = {}
        for i, n in enumerate(self.names):
            v, a, _ = self.parts(i)
            w = self._targets[n].weight
            w0 = w.detach().to(torch.float64, copy=True)  # a copy even when the weight is float64 already
            delta = -(a.double() * (1.0 - self.gates[i]).double()) @ v.double().transpose(0, 1)
            w.copy_((w0 + delta).to(w.dtype))
            moved = float((w.detach().double() - w0).norm())
            out[n] = {"displacement": moved, "weight_rel_change": moved / float(w0.norm())}
            del w0, delta
        return out


@torch.no_grad()
def removed_energy(gates: DirectionGates, cov_forget: Dict[str, torch.Tensor],
                   cov_retain: Dict[str, torch.Tensor]) -> Dict[str, dict]:
    """Per target the forget and retain output energy the gates remove, ``tr(dW C dW^T) = sum((A^T A) * (V^T C V) *
    c c^T)`` with c = 1 - s, and ``tr(W0 C W0^T)``; call before ``materialize`` (the weights are then W0)."""
    out = {}
    for i, n in enumerate(gates.names):
        v, a, _ = gates.parts(i)
        w = gates._targets[n].weight.detach().double()
        c = (1.0 - gates.gates[i].detach()).double()
        v64, a64 = v.double(), a.double()
        gram = (a64.transpose(0, 1) @ a64) * torch.outer(c, c)
        row = {}
        for side, cov in (("forget", cov_forget[n]), ("retain", cov_retain[n])):
            cc = cov.to(w.device).double()
            removed = float((gram * (v64.transpose(0, 1) @ cc @ v64)).sum())
            total = float(((w @ cc) * w).sum())
            row.update({f"{side}_energy_removed": removed, f"{side}_energy": total,
                        f"{side}_share": removed / total if total > 0 else None})
        out[n] = row
    return out


@torch.no_grad()
def energy_forms(gates: DirectionGates, cov_forget: Dict[str, torch.Tensor],
                 cov_retain: Dict[str, torch.Tensor]) -> List[dict]:
    """Per target the removed output energy as a quadratic form in the contraction c = 1 - s,
    ``tr(dW C dW^T) = c^T M c`` with ``M = (A^T A) * (V^T C V)``, for the forget and the retain covariance, and
    ``tr(W0 C W0^T)`` (float64; call before ``materialize``, the weights are then W0)."""
    forms = []
    for i, n in enumerate(gates.names):
        v, a, _ = gates.parts(i)
        w = gates._targets[n].weight.detach().double()
        v64, a64 = v.double(), a.double()
        gram = a64.transpose(0, 1) @ a64
        row = {"name": n}
        for side, cov in (("forget", cov_forget[n]), ("retain", cov_retain[n])):
            cc = cov.to(w.device).double()
            row[side] = gram * (v64.transpose(0, 1) @ cc @ v64)
            row[f"{side}_total"] = float(((w @ cc) * w).sum())
        forms.append(row)
    return forms


@torch.no_grad()
def binarize_by_share(gates: DirectionGates, cov_forget: Dict[str, torch.Tensor],
                      cov_retain: Dict[str, torch.Tensor], share: float, s_hard: float) -> dict:
    """The gates choose the directions, the cut removes them whole (user 2026-11-04): the gates below 1 sorted by their
    value over all targets (most closed first), and the shortest such prefix whose removal at ``s_hard`` (c = 1 - s_hard
    on the chosen gates, 0 elsewhere) removes at least ``share`` of the forget output energy summed over the targets
    (as the global selection's budget), found by bisection on its length; the chosen gates are set to ``s_hard``, every
    other gate to 1. Returns the count (in all and per projection type), the largest chosen gate value (the
    threshold) and the forget and retain shares removed."""
    if not 0.0 < share < 1.0 or not 0.0 <= s_hard < 1.0:
        raise ValueError(f"need share in (0, 1) and s_hard in [0, 1), got {share}, {s_hard}")
    forms = energy_forms(gates, cov_forget, cov_retain)
    sizes = [g.numel() for g in gates.gates]
    s = torch.cat([g.detach().double().flatten() for g in gates.gates])
    order = torch.argsort(s, stable=True)
    movable = int((s < 1).sum())
    totals = {side: sum(f[f"{side}_total"] for f in forms) for side in ("forget", "retain")}

    def removed(m: int, side: str) -> float:
        c = torch.zeros_like(s)
        c[order[:m]] = 1.0 - s_hard
        return sum(float(part @ (f[side] @ part)) for part, f in zip(c.split(sizes), forms))

    target = share * totals["forget"]
    if movable == 0 or removed(movable, "forget") < target:
        raise ValueError(f"share {share} unreachable: removing all {movable} gates below 1 removes "
                         f"{removed(movable, 'forget') / totals['forget']:.4f} of the forget output energy")
    lo, hi = 0, movable  # removed(lo) < target <= removed(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if removed(mid, "forget") >= target:
            hi = mid
        else:
            lo = mid
    chosen = torch.zeros_like(s, dtype=torch.bool)
    chosen[order[:hi]] = True
    per_type = {}
    for g, mask, n in zip(gates.gates, chosen.split(sizes), gates.names):
        g.copy_(torch.where(mask.to(g.device), torch.full_like(g, s_hard), torch.ones_like(g)))
        typ = n.rsplit(".", 1)[-1]
        per_type[typ] = per_type.get(typ, 0) + int(mask.sum())
    return {"share": share, "s_hard": s_hard, "gates_removed": hi, "gates_below_1": movable,
            "threshold": float(s[order[hi - 1]]), "per_type": per_type,
            "forget_share": removed(hi, "forget") / totals["forget"],
            "retain_share": removed(hi, "retain") / totals["retain"]}


@torch.no_grad()
def prefix_change(gates: DirectionGates, prefix_inputs: Dict[str, torch.Tensor]) -> Dict[str, dict]:
    """Per target, how much the current gates change its output at the constant prefix positions (call before
    ``materialize``): for each prefix input z_t, ``||dW z_t||^2 / ||W0 z_t||^2`` with dW = -A diag(1 - s) V^T (the
    position-wise counterpart of the removed energy shares over answer tokens), at the first position (BOS), its mean
    and largest value over the prefix, and the read leakage ``max_t ||V^T z_t|| / ||z_t||`` (0 for a protected basis)."""
    out = {}
    for i, n in enumerate(gates.names):
        v, a, _ = gates.parts(i)
        z = prefix_inputs[n].to(v.device).double()
        w = gates._targets[n].weight.detach().double()
        coef = z @ v.double()  # (prefix, k)
        delta = (coef * (1.0 - gates.gates[i].detach().double())) @ a.double().transpose(0, 1)
        base = (z @ w.transpose(0, 1)).square().sum(1).clamp_min(1e-300)
        share = delta.square().sum(1) / base
        out[n] = {"bos_share": float(share[0]), "prefix_share_mean": float(share.mean()),
                  "prefix_share_max": float(share.max()),
                  "read_leak_max": float((coef.norm(dim=1) / z.norm(dim=1).clamp_min(1e-300)).max())}
    return out


GROWTH_GROUPS = (("cut", -1.0, 0.5), ("partial", 0.5, 1.0), ("open", 1.0, 1.0))  # cut: the reflection s = -1 too


@torch.no_grad()
def regrowth_report(weights: Dict[str, torch.Tensor], saved: Dict[str, dict]) -> dict:
    """Whether phase two regrows what the gates cut (user 2026-11-04, the recovery hypothesis): per gate j of a
    target, with the write direction u_j (u_j = v_j for an orthonormal basis), its Init Fine. part a_j = W0 u_j and
    the gate s_j it was materialized with (the part is then s_j a_j, by orthonormality or duality), the final weight's
    part W u_j. Per group of gates (``cut``: -1 <= s < 0.5, ``partial``: 0.5 <= s < 1, ``open``: s = 1), over all targets and
    per projection type: the count, ``start`` = sum s^2 ||a||^2 / sum ||a||^2 (where phase two began),
    ``regrowth`` = sum ||W u||^2 / sum ||a||^2 and ``back`` = sum a^T W u / sum ||a||^2 (1: back to Init Fine. along
    the original direction, s: unchanged since the cut). Rotated parts (``rotated``: the columns a'_j = Q a_j of
    ``rotate_cut``, ``rotated_index`` their gates) form their own group, with ``start`` the back at the cut
    (a^T a' / ||a||^2: about 0 for the quarter turn, c for the partial turn) and ``stay`` = sum a'^T W u / sum ||a||^2
    (1: still rotated)."""
    acc = {}
    for n, rec in saved.items():
        w = weights[n].detach().double()
        u = rec["u"].to(w.device).double()
        a = rec["a"].to(w.device).double()
        s = rec["s"].to(w.device).double()
        wu = w @ u
        e = a.square().sum(0)
        stats = {"start": s.square() * e, "regrowth": wu.square().sum(0), "back": (a * wu).sum(0), "energy": e}
        rotated = rec.get("rotated")  # columns of the rotated parts (their gate is 1, their part a'_j = Q a_j)
        if rotated is not None:
            rot = torch.zeros_like(s, dtype=torch.bool)
            rot[rec["rotated_index"].to(w.device)] = True
            ar = torch.zeros_like(a)
            ar[:, rot] = rotated.to(w.device).double()
            stats["start"] = torch.where(rot, (ar * a).sum(0), stats["start"])  # back at the cut: a^T a' / ||a||^2
            stats["stay"] = (ar * wu).sum(0)
        typ = n.rsplit(".", 1)[-1]
        for g, lo, hi in GROWTH_GROUPS + (("rotated", None, None),):
            if g == "rotated":
                if rotated is None:
                    continue
                mask = rot
            else:
                mask = (s == 1.0) if g == "open" else (s >= lo) & (s < hi)
                if rotated is not None:
                    mask = mask & ~rot
            for scope in ("all", typ):
                d = acc.setdefault(scope, {}).setdefault(g, {"count": 0, "energy": 0.0, "start": 0.0, "regrowth": 0.0,
                                                             "back": 0.0, "stay": 0.0})
                d["count"] += int(mask.sum())
                for key in ("energy", "start", "regrowth", "back") + (("stay",) if g == "rotated" else ()):
                    d[key] += float(stats[key][mask].sum())
    out = {}
    for scope, groups in acc.items():
        out[scope] = {}
        for g, d in groups.items():
            e = d["energy"]
            out[scope][g] = {"count": d["count"], "energy": e,
                             **{k: (d[k] / e if e > 0 else None) for k in ("start", "regrowth", "back")
                                + (("stay",) if g == "rotated" else ())}}
    return out


@torch.no_grad()
def rotate_cut(a: torch.Tensor, weight: torch.Tensor, cov_forget: torch.Tensor, evr: float,
               generator: torch.Generator, keep: float = 0.0) -> dict:
    """A rotation instead of a shrink for the cut parts of one target (user 2026-11-04: "obrót ortogonalny do forget"):
    the parts' outputs A (out x m, the columns a_j = W0 u_j of the cut gates) are carried by one orthogonal map Q of the
    output space onto a random subspace orthogonal both to their own span and to the target's forget output subspace F
    (the leading eigenvectors of W0 C_f W0^T holding the share ``evr`` of its trace), by a quarter turn in each plane
    (Q maps an orthonormal basis U of span(A) onto an orthonormal basis T of the new subspace): A' = T U^T A. Every
    inner product among the parts is kept (Q is orthogonal), so each input's output along the cut channel keeps its
    norm, written where neither these parts nor the forget outputs were. With ``keep`` = c in (-1, 1) a partial turn
    (user 2026-11-05, the rotation with S2's floor): A' = c A + sqrt(1 - c^2) T U^T A, by the angle arccos c in each
    plane; each part keeps c of itself along its own direction (the shrink to c's component) and its norm, the inner
    products are still kept (T is orthogonal to span(A)), and ||a' - a||^2 = 2 (1 - c) ||a||^2 (c 0: the quarter turn,
    exactly as before). Returns A' (float64) and a report (the overlaps are those of the moved component T U^T A);
    raises if the output space has no room (rank A + dim F + rank A > out)."""
    if not -1.0 < keep < 1.0:
        raise ValueError(f"keep must lie in (-1, 1), got {keep}")
    a64 = a.double()
    w = weight.detach().double()
    out = a64.shape[0]
    if a64.shape[1] == 0:
        return {"rotated": a64.clone(), "rank": 0, "forget_dims": 0}
    u, r = torch.linalg.qr(a64)
    full_rank = torch.diagonal(r).abs() > 1e-10 * torch.diagonal(r).abs().max()
    u = u[:, full_rank]
    m = u.shape[1]
    af = w @ cov_forget.to(w.device).double() @ w.transpose(0, 1)
    vals, vecs = torch.linalg.eigh(0.5 * (af + af.transpose(0, 1)))
    vals, vecs = vals.flip(0).clamp_min(0), vecs.flip(1)
    rf = int(torch.searchsorted(torch.cumsum(vals, 0) / vals.sum(), torch.tensor([evr], dtype=vals.dtype,
                                                                               device=vals.device)).item()) + 1
    taken, _ = torch.linalg.qr(torch.cat([u, vecs[:, :rf]], dim=1))
    if taken.shape[1] + m > out:
        raise ValueError(f"no room to rotate: rank {m} + forget dims {rf} + rank {m} > {out}")
    g = torch.randn(out, m, generator=generator, dtype=torch.float64).to(w.device)
    g = g - taken @ (taken.transpose(0, 1) @ g)
    t, _ = torch.linalg.qr(g)
    moved = t @ (u.transpose(0, 1) @ a64)
    rotated = moved if keep == 0.0 else keep * a64 + math.sqrt(1.0 - keep * keep) * moved
    return {"rotated": rotated, "rank": m, "forget_dims": rf, "keep": keep,
            "own_overlap": float((u.transpose(0, 1) @ moved).norm() / moved.norm().clamp_min(1e-300)),
            "forget_overlap": float((vecs[:, :rf].transpose(0, 1) @ moved).norm() / moved.norm().clamp_min(1e-300)),
            "gram_error": float(((rotated.transpose(0, 1) @ rotated) - (a64.transpose(0, 1) @ a64)).abs().max()
                                / (a64.transpose(0, 1) @ a64).abs().max().clamp_min(1e-300))}


@torch.no_grad()
def edit_energy(delta_a: torch.Tensor, v: torch.Tensor, weight: torch.Tensor, cov: torch.Tensor) -> dict:
    """For an edit dW = dA V^T of one target (dA out x m, V in x m): ``tr(dW C dW^T)`` and its share of
    ``tr(W0 C W0^T)`` (float64)."""
    da, v64, w = delta_a.double(), v.double(), weight.detach().double()
    cc = cov.to(w.device).double()
    removed = float(((da.transpose(0, 1) @ da) * (v64.transpose(0, 1) @ cc @ v64)).sum())
    total = float(((w @ cc) * w).sum())
    return {"energy": removed, "total": total, "share": removed / total if total > 0 else None}


def forget_anchored_windows(n_forget: int, n_retain: int, window: int, epochs: int, generator: torch.Generator):
    """Per epoch a permutation of the forget items cut into windows of ``window`` items (the last one shorter), each
    forget item with a uniformly random retain partner: a list of (epoch, forget indices, retain indices)."""
    if min(n_forget, n_retain, window, epochs) <= 0:
        raise ValueError("need positive split sizes, window and epochs")
    out = []
    for epoch in range(epochs):
        perm = torch.randperm(n_forget, generator=generator)
        partner = torch.randint(0, n_retain, (n_forget,), generator=generator)
        for start in range(0, n_forget, window):
            out.append((epoch, perm[start:start + window], partner[start:start + window]))
    return out


@torch.no_grad()
def logits_of(model: nn.Module, inputs: dict) -> torch.Tensor:
    return model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False).logits


@torch.no_grad()
def check_identity(model: nn.Module, gates: DirectionGates, inputs: dict, autocast_dtype=None) -> None:
    """Positive control: with every gate at 1 the gated logits are bitwise those of the model without hooks."""
    if any(bool((g != 1).any()) for g in gates.gates):
        raise RuntimeError("the identity check needs every gate at 1")
    ctx = _autocast(model, autocast_dtype)
    with ctx():
        plain = logits_of(model, inputs)
    gates.attach()
    try:
        with ctx():
            gated = logits_of(model, inputs)
    finally:
        gates.detach()
    if not torch.equal(plain, gated):
        raise RuntimeError(f"gates at 1 changed the logits (max |diff| {float((plain - gated).abs().max()):.3e})")


@torch.no_grad()
def materialize_checked(model: nn.Module, gates: DirectionGates, inputs: dict) -> dict:
    """Materialize the gates; positive control on ``inputs`` (no autocast): the logits of the materialized model match
    the gated ones within MATERIALIZE_RTOL of the largest logit. Returns the per-target report and the check."""
    gates.attach()
    try:
        gated = logits_of(model, inputs).float()
    finally:
        gates.detach()
    per_target = gates.materialize()
    native = logits_of(model, inputs).float()
    mask = inputs["attention_mask"].bool()
    diff = float((gated - native).abs().max())
    scale = float(gated.abs().max())
    agree = float((gated.argmax(-1) == native.argmax(-1))[mask].float().mean())
    check = {"max_abs_logit_diff": diff, "max_abs_logit": scale, "relative": diff / scale if scale > 0 else 0.0,
             "argmax_agreement": agree}
    if check["relative"] > MATERIALIZE_RTOL:
        raise RuntimeError(f"materialized logits differ from the gated ones: {check}")
    return {"per_target": per_target, "check": check}


@torch.no_grad()
def weight_checksums(model: nn.Module) -> Dict[str, tuple]:
    """(sum, sum of squares) in float64 of every parameter and buffer, to show the gate phase left them unchanged."""
    return {n: (float(t.double().sum()), float(t.double().square().sum()))
            for n, t in list(model.named_parameters()) + list(model.named_buffers())}


def _autocast(model: nn.Module, dtype):
    if dtype is None:
        return nullcontext
    device_type = next(model.parameters()).device.type
    return lambda: torch.autocast(device_type=device_type, dtype=dtype)


def gate_optimizer(params, optim: str, lr: float) -> torch.optim.Optimizer:
    """Adam (betas 0.9 / 0.999, eps 1e-8) or plain SGD (momentum 0, as Classification), constant lr, no decay."""
    if optim == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    if optim == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=0.0)
    raise ValueError(f"optim must be one of {GATE_OPTIMIZERS}, got {optim!r}")


def gate_stats(s: torch.Tensor) -> dict:
    return {"gate_mean": float(s.mean()), "gate_min": float(s.min()), "gates_below_0p5": int((s < 0.5).sum()),
            "gates_below_0p1": int((s < 0.1).sum()), "gates_at_0": int((s == 0).sum()), "gates_at_1": int((s == 1).sum())}


def update_stats(delta: torch.Tensor, previous: Optional[torch.Tensor]) -> dict:
    """One gate update delta = s_t - s_{t-1} (after the clip) against the previous one: its L2 and max norms, the gates
    it moved, the cosine of the two updates (< 0: the step went back) and the share of the gates moved by both whose
    direction reversed (an oscillation); the last two None without a previous update or a common moving gate."""
    moving = delta != 0
    rec = {"update_l2": float(delta.norm()), "update_max": float(delta.abs().max()), "gates_moving": int(moving.sum()),
           "update_cosine": None, "reversal_share": None}
    if previous is not None:
        norms = float(delta.norm()) * float(previous.norm())
        if norms > 0:
            rec["update_cosine"] = float(delta @ previous) / norms
        both = moving & (previous != 0)
        if bool(both.any()):
            rec["reversal_share"] = float(((delta * previous)[both] < 0).double().mean())
    return rec


def convergence_check(objectives: Sequence[float], net: float, path: float, tol: float, drift_max: float) -> dict:
    """Convergence over a trailing window of epochs (module docstring): the relative decrease of the probe objective
    from the window's first to its last value and the drift ratio net / path of the gates (0 when no gate moved)."""
    first, last = float(objectives[0]), float(objectives[-1])
    decrease = (first - last) / abs(first) if first != 0 else 0.0
    drift = net / path if path > 0 else 0.0
    return {"objective_decrease": decrease, "drift": drift, "converged": decrease <= tol and drift <= drift_max}


@torch.no_grad()
def probe_objective(model: nn.Module, gates: DirectionGates, probe: Dict[str, list], to_device: Callable[[dict], dict],
                    alpha: float, beta: float, ctx, gamma: float = 0.0) -> dict:
    """J and its parts with the current gates (attached by the caller) on the fixed probe batches: the forget
    unlikelihood and NLL and the retain NLL, each a mean over the answer tokens of its whole set."""
    sums = {"forget_unlikelihood": 0.0, "forget_nll": 0.0, "retain_nll": 0.0}
    counts = {"forget": 0, "retain": 0}
    for kind, batches in probe.items():
        for b in batches:
            x = to_device(b)
            with ctx():
                logits = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"], use_cache=False).logits
            nll, n = nll_sum(logits, x["labels"])
            sums[f"{kind}_nll"] += float(nll)
            if kind == "forget":
                sums["forget_unlikelihood"] += float(unlikelihood_sum(logits, x["labels"])[0])
            counts[kind] += n
            del logits
    out = {k: v / counts[k.split("_")[0]] for k, v in sums.items()}
    out["regulariser"] = float(gates.regulariser())
    out["openness"] = float(gates.openness())
    out["objective"] = (out["retain_nll"] + alpha * out["forget_unlikelihood"] + gamma * out["openness"]
                        + beta * out["regulariser"])
    return out


def train_gates(model: nn.Module, gates: DirectionGates, forget_items: Sequence, retain_items: Sequence,
                collate: Callable[[list], dict], to_device: Callable[[dict], dict], *, max_epochs: int,
                convergence_epochs: int, tol: float, drift_max: float, micro_batch: int, grad_accum: int, optim: str,
                lr: float, alpha: float, beta: float, probe_retain: int, seed: int, gamma: float = 0.0,
                autocast_dtype=None, log_step: Optional[Callable[[dict], None]] = None,
                log_epoch: Optional[Callable[[dict], None]] = None, lr_schedule: str = "constant",
                probe_every: int = 1) -> dict:
    """Train the gates alone on J = L_R + alpha L_F + gamma O + beta R until convergence or ``max_epochs`` (module
    docstring):
    one step of ``gate_optimizer`` per window of ``micro_batch * grad_accum`` forget items and as many random retain
    partners, then the gates are clipped to [0, 1]. One seeded generator draws first the ``probe_retain`` retain items
    of the probe, then every epoch's windows. The model runs in eval mode with every parameter frozen; modes and
    ``requires_grad`` flags are restored afterwards. Returns the step records (the window losses of the gates before
    the step, the gradient of each term, the update), the epoch records (epoch 0: the probe at the start), the stop
    reason, the contraction 1 - s after every epoch (float32, epochs x gates) and per gate the path length sum_t
    |s_t - s_{t-1}| and the number of reversals. ``convergence_epochs`` 0 is the fixed schedule (module docstring):
    exactly ``max_epochs`` epochs (stop ``fixed_schedule``), ``tol`` and ``drift_max`` None, ``lr_schedule`` constant
    or linear and the probe after every ``probe_every``-th and the last epoch (the other epoch records carry no probe
    fields); with the convergence rule the lr is constant and the probe follows every epoch."""
    rule = convergence_epochs > 0
    if (max_epochs < 0 or convergence_epochs < 0 or min(micro_batch, grad_accum, probe_retain, probe_every) <= 0
            or lr <= 0 or min(alpha, beta, gamma) < 0 or alpha + gamma <= 0 or probe_retain > len(retain_items)
            or lr_schedule not in GATE_LR_SCHEDULES
            or (rule and (tol is None or drift_max is None or tol < 0 or not 0 < drift_max <= 1
                          or (max_epochs > 0 and convergence_epochs > max_epochs) or lr_schedule != "constant"
                          or probe_every != 1))
            or (not rule and (tol is not None or drift_max is not None))):
        raise ValueError("need max_epochs >= 0 (0: only the probe at the start, no step), convergence_epochs >= 0 "
                         "(at most max_epochs; 0: the fixed schedule, without tol and drift_max), positive "
                         "micro_batch, grad_accum, lr, probe_every and probe_retain <= the retain items, alpha, beta, "
                         f"gamma >= 0 with alpha or gamma positive, lr_schedule in {GATE_LR_SCHEDULES} (with the "
                         "convergence rule: constant, the probe every epoch, tol >= 0 and drift_max in (0, 1])")
    params = list(gates.gates.parameters())
    opt = gate_optimizer(params, optim, lr)  # raises on an unknown optimizer before the model is touched
    modes = [(m, m.training) for m in model.modules()]
    flags = [(p, p.requires_grad) for p in model.parameters()]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ctx = _autocast(model, autocast_dtype)
    gen = torch.Generator().manual_seed(int(seed))
    probe_idx = torch.randperm(len(retain_items), generator=gen)[:probe_retain]
    probe = {"forget": [collate([forget_items[i] for i in range(b, min(b + micro_batch, len(forget_items)))])
                        for b in range(0, len(forget_items), micro_batch)],
             "retain": [collate([retain_items[int(j)] for j in c]) for c in probe_idx.split(micro_batch)]}

    def flat() -> torch.Tensor:
        return torch.cat([g.detach().flatten() for g in params])

    def grad_vector() -> torch.Tensor:
        return torch.cat([(g.grad if g.grad is not None else torch.zeros_like(g)).detach().flatten() for g in params])

    s = flat().clone()
    snapshots = [(1.0 - s).float().cpu()]
    trailing = [s]  # float64 gates at the end of the last convergence_epochs + 1 epochs
    gate_path = torch.zeros_like(s)
    reversals = torch.zeros(s.shape, dtype=torch.long, device=s.device)
    steps, epochs, epoch_paths = [], [], []
    previous, step = None, 0
    stop = "no_training" if max_epochs == 0 else ("max_epochs" if rule else "fixed_schedule")
    total_steps = max_epochs * math.ceil(len(forget_items) / (micro_batch * grad_accum))
    try:
        gates.attach()
        rec = {"epoch": 0, "steps": 0,
               **{f"probe_{k}": v for k, v in
                  probe_objective(model, gates, probe, to_device, alpha, beta, ctx, gamma).items()},
               **gate_stats(s)}
        epochs.append(rec)
        if log_epoch is not None:
            log_epoch(rec)
        for epoch in range(1, max_epochs + 1):
            start, path, shares = s, 0.0, []
            for _, fidx, ridx in forget_anchored_windows(len(forget_items), len(retain_items), micro_batch * grad_accum,
                                                         1, gen):
                step += 1
                parts = {"forget": ([collate([forget_items[int(i)] for i in c]) for c in fidx.split(micro_batch)],
                                    alpha, unlikelihood_sum),
                         "retain": ([collate([retain_items[int(j)] for j in c]) for c in ridx.split(micro_batch)],
                                    1.0, nll_sum)}
                opt.zero_grad(set_to_none=True)
                means, tokens, grads = {}, {}, {}
                for kind, (batches, weight, loss_fn) in parts.items():
                    count = sum(int((b["labels"][:, 1:] != IGNORE_INDEX).sum()) for b in batches)
                    if count == 0:
                        raise ValueError(f"a {kind} window without answer tokens (step {step})")
                    total = 0.0
                    for b in batches:
                        x = to_device(b)
                        with ctx():
                            logits = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"],
                                           use_cache=False).logits
                        loss, _ = loss_fn(logits, x["labels"])
                        (loss * (weight / count)).backward()
                        total += float(loss.detach())
                        del logits, loss
                    means[kind], tokens[kind] = total / count, count
                    grads[kind] = grad_vector()
                reg = gates.regulariser()
                (beta * reg).backward()
                grads["regulariser"] = grad_vector()
                opened = gates.openness()
                (gamma * opened).backward()
                g_all = grad_vector()
                g_parts = {"forget": grads["forget"], "retain": grads["retain"] - grads["forget"],
                           "regulariser": grads["regulariser"] - grads["retain"],
                           "closing": g_all - grads["regulariser"]}
                lr_t = lr if lr_schedule == "constant" else lr * (total_steps - step + 1) / total_steps
                if lr_schedule != "constant":
                    for group in opt.param_groups:
                        group["lr"] = lr_t
                opt.step()
                gates.clamp_()
                after = flat()
                delta = after - s
                upd = update_stats(delta, previous)
                if previous is not None:
                    reversals += (delta * previous < 0).long()
                gate_path += delta.abs()
                path += upd["update_l2"]
                if upd["reversal_share"] is not None:
                    shares.append(upd["reversal_share"])
                pushed = int(((g_all > 0) & (s > 0)).sum())  # before the step: the gradient would lower them
                previous, s = delta, after
                rec = {"step": step, "epoch": epoch, "lr": lr_t, "forget_unlikelihood": means["forget"],
                       "retain_nll": means["retain"], "regulariser": float(reg), "openness": float(opened),
                       "objective": (means["retain"] + alpha * means["forget"] + gamma * float(opened)
                                     + beta * float(reg)),
                       "forget_tokens": tokens["forget"], "retain_tokens": tokens["retain"],
                       **{f"grad_{k}_l2": float(g.norm()) for k, g in g_parts.items()},
                       "grad_l2": float(g_all.norm()), "grad_max": float(g_all.abs().max()),
                       "gates_pushed_shut": pushed, **upd, **gate_stats(s)}
                steps.append(rec)
                if log_step is not None:
                    log_step(rec)
            epoch_paths.append(path)
            net = s - start
            probed = rule or epoch % probe_every == 0 or epoch == max_epochs
            rec = {"epoch": epoch, "steps": step,
                   **({f"probe_{k}": v for k, v in
                       probe_objective(model, gates, probe, to_device, alpha, beta, ctx, gamma).items()}
                      if probed else {}),
                   "epoch_net_l2": float(net.norm()), "epoch_net_max": float(net.abs().max()), "epoch_path_l2": path,
                   "epoch_drift": float(net.norm()) / path if path > 0 else 0.0,
                   "epoch_reversal_share": sum(shares) / len(shares) if shares else None, **gate_stats(s)}
            snapshots.append((1.0 - s).float().cpu())
            trailing = (trailing + [s])[-(convergence_epochs + 1):]
            converged = False
            if rule and epoch >= convergence_epochs:
                check = convergence_check([e["probe_objective"] for e in epochs[epoch - convergence_epochs:]]
                                          + [rec["probe_objective"]], float((s - trailing[0]).norm()),
                                          sum(epoch_paths[-convergence_epochs:]), tol, drift_max)
                rec.update({f"window_{k}": v for k, v in check.items()})
                converged = check["converged"]
            epochs.append(rec)
            if log_epoch is not None:
                log_epoch(rec)
            if converged:
                stop = "converged"
                break
    finally:
        gates.detach()
        for p, flag in flags:
            p.requires_grad_(flag)
        for m, mode in modes:
            m.train(mode)
    return {"steps": steps, "epochs": epochs, "stop": stop, "converged": stop == "converged",
            "epochs_run": len(epochs) - 1, "probe_retain_indices": probe_idx.tolist(),
            "contraction_snapshots": torch.stack(snapshots), "gate_path": gate_path.cpu(),
            "gate_reversals": reversals.cpu()}


def gradient_census(model: nn.Module, gates: DirectionGates, forget_items: Sequence, retain_items: Sequence,
                    collate: Callable[[list], dict], to_device: Callable[[dict], dict], *, epochs: int,
                    micro_batch: int, grad_accum: int, seed: int, autocast_dtype=None) -> dict:
    """The scale of the gate gradients at the current gates, without a step (user 2026-11-03: measure how to choose
    the learning rate): over the windows of ``epochs`` forget-anchored epochs (a generator seeded with ``seed``),
    the gradient of L_F and of L_R (each alone, a mean over its window's answer tokens) per window and gate, float32
    (windows x gates). The model is frozen as in ``train_gates``."""
    if min(epochs, micro_batch, grad_accum) <= 0:
        raise ValueError("need positive epochs, micro_batch and grad_accum")
    params = list(gates.gates.parameters())
    modes = [(m, m.training) for m in model.modules()]
    flags = [(p, p.requires_grad) for p in model.parameters()]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ctx = _autocast(model, autocast_dtype)
    windows = forget_anchored_windows(len(forget_items), len(retain_items), micro_batch * grad_accum, epochs,
                                      torch.Generator().manual_seed(int(seed)))
    rows = {"forget": [], "retain": []}
    try:
        gates.attach()
        for _, fidx, ridx in windows:
            for kind, items, idx, loss_fn in (("forget", forget_items, fidx, unlikelihood_sum),
                                              ("retain", retain_items, ridx, nll_sum)):
                for g in params:
                    g.grad = None
                batches = [collate([items[int(i)] for i in c]) for c in idx.split(micro_batch)]
                count = sum(int((b["labels"][:, 1:] != IGNORE_INDEX).sum()) for b in batches)
                if count == 0:
                    raise ValueError(f"a {kind} window without answer tokens")
                for b in batches:
                    x = to_device(b)
                    with ctx():
                        logits = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"],
                                       use_cache=False).logits
                    loss, _ = loss_fn(logits, x["labels"])
                    (loss / count).backward()
                    del logits, loss
                rows[kind].append(torch.cat([(g.grad if g.grad is not None else torch.zeros_like(g)).detach().flatten()
                                             for g in params]).float().cpu())
    finally:
        gates.detach()
        for g in params:
            g.grad = None
        for p, flag in flags:
            p.requires_grad_(flag)
        for m, mode in modes:
            m.train(mode)
    return {"windows": len(windows), "forget": torch.stack(rows["forget"]), "retain": torch.stack(rows["retain"])}


CENSUS_QUANTILES = (0.5, 0.9, 0.99)
CENSUS_STEPS = (1.0e-3, 1.0e-2, 1.0e-1)  # target gate moves per SGD step
CENSUS_ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0)
CENSUS_SHARES = (0.01, 0.1, 0.25, 0.5, 0.75, 0.9)


def census_summary(census: dict, alpha: float, gamma: float) -> dict:
    """Learning-rate and strength guides from ``gradient_census`` at s = 1 (first order, before any curvature acts),
    for J's gradient g = alpha g_F + g_R + gamma per window and gate:

    - ``abs_gradient``: quantiles over windows and gates of |alpha g_F|, |g_R| and |g|;
    - ``sgd_lr_for_step``: the SGD lr whose 99th-percentile step lr |g| equals each of CENSUS_STEPS;
    - ``adam_step_per_lr``: quantiles over gates of |mean g| / sqrt(mean g^2) over the windows, Adam's steady step per
      unit lr (1: every window agrees, -> 0: noise); ``adam_steps_to_close``: 1 / that at its median, x 1 / lr;
    - ``pushed_shut_share``: the share of gates whose mean g is > 0 (the step lowers them), for this alpha and gamma
      and for every alpha of CENSUS_ALPHAS at gamma 0;
    - ``closing_price``: per share of CENSUS_SHARES the gamma that pushes that share of the gates shut against L_R
      alone (alpha 0): the quantile of -mean g_R (gates where closing lowers L_R are pushed shut at any gamma)."""
    gf, gr = census["forget"].double(), census["retain"].double()
    g = alpha * gf + gr + gamma
    qs = torch.tensor(CENSUS_QUANTILES, dtype=torch.float64)

    def quant(x):
        return dict(zip((f"q{int(q * 100)}" for q in CENSUS_QUANTILES),
                        torch.quantile(x.flatten(), qs).tolist()))

    mean, rms = g.mean(0), g.square().mean(0).sqrt()
    share = torch.where(rms > 0, mean.abs() / rms.clamp_min(1e-300), torch.zeros_like(rms))
    q99 = float(torch.quantile(g.abs().flatten(), torch.tensor(0.99, dtype=torch.float64)))
    price = -gr.mean(0)
    adam_share = quant(share)
    return {"windows": census["windows"], "gates": int(g.shape[1]),
            "abs_gradient": {"alpha_forget": quant((alpha * gf).abs()), "retain": quant(gr.abs()),
                             "objective": quant(g.abs())},
            "abs_mean_gradient": {"alpha_forget": quant((alpha * gf.mean(0)).abs()), "retain": quant(gr.mean(0).abs()),
                                  "objective": quant(mean.abs())},
            "sgd_lr_for_step": {f"{d:g}": d / q99 if q99 > 0 else None for d in CENSUS_STEPS},
            "adam_step_per_lr": adam_share,
            "adam_steps_to_close_times_lr": 1.0 / adam_share["q50"] if adam_share["q50"] > 0 else None,
            "pushed_shut_share": float((mean > 0).double().mean()),
            "pushed_shut_share_by_alpha": {f"{a:g}": float(((a * gf.mean(0) + gr.mean(0)) > 0).double().mean())
                                           for a in CENSUS_ALPHAS},
            "closing_price": dict(zip((f"{s:g}" for s in CENSUS_SHARES),
                                      torch.quantile(price, torch.tensor(CENSUS_SHARES, dtype=torch.float64)).tolist())),
            "closing_free_share": float((price < 0).double().mean())}


def summarize_gates(per_target: Dict[str, dict]) -> Dict[str, dict]:
    """Totals per projection type (the last component of the target name) and over all targets (``all``) of the
    per-target reports (``summary``, ``removed_energy`` and ``materialize`` merged)."""
    groups: Dict[str, List[dict]] = {}
    for n, r in per_target.items():
        groups.setdefault(n.rsplit(".", 1)[-1], []).append(r)
    groups["all"] = list(per_target.values())
    out = {}
    for g, rows in groups.items():
        agg = {"targets": len(rows)}
        for key in ("k", "below_0p5", "below_0p1", "at_0", "at_1", "contraction_sum", "displacement2",
                    "forget_energy_removed", "forget_energy", "retain_energy_removed", "retain_energy"):
            agg[key] = sum(r[key] for r in rows)
        agg["gate_mean"] = sum(r["mean"] * r["k"] for r in rows) / agg["k"]
        for side in ("forget", "retain"):
            agg[f"{side}_share"] = agg[f"{side}_energy_removed"] / agg[f"{side}_energy"] if agg[f"{side}_energy"] > 0 else None
        agg["weight_rel_change_mean"] = sum(r["weight_rel_change"] for r in rows) / len(rows)
        agg["displacement"] = math.sqrt(sum(r["displacement"] ** 2 for r in rows))
        out[g] = agg
    return out
