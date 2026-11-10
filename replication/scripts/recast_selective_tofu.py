"""Driver of the selective input contrast (recast_selective.py; extension, user 2026-11-03) on TOFU forget10 from
Init Fine. (open-unlearning/tofu_Llama-3.2-1B-Instruct_full), targets o_proj and down_proj of every block.

--mode diagnose (no training):
  1. checks of the hypothesis behind down_proj block 1, the most specific down block by answer-token activation energy
     and the least by the answer NLL (25482599 against 25559333): the statistics cover answer tokens only while the
     edit acts on every position, the constant prompt prefix with the BOS token among them, whose massive activations
     the model needs. (a) Massive activations: per layer the largest |h| of the residual stream, and per block of the
     down_proj input, at each prefix position, over the prompt tokens and over the answer tokens, with its dimension;
     (b) per target the share of the prefix inputs' energy along the leading 5% of the old activation contrast (answer
     tokens, no protection) and that energy per prefix token against per answer token; (c) the answer NLL change when
     those directions are removed from one target only at the prefix positions, only elsewhere, or everywhere (the
     last repeats 25559333's per-layer numbers), for CHECK_TARGETS;
  2. the selection grid, PENCILS x LAMBDAS: families act (C_f vs C_r) and fisher (per-sequence Fisher input moments,
     forget vs retain, as 25559333), without protection, with the prefix span, and with the prefix span plus kappa (the
     retain non-prefix activation covariance); per target the count, the energy report and the lambda spectrum's
     quantiles; the joint edit of every target measured by inference at GRID_FLOORS;
  3. every target's edit alone (floor 0) at GUARD_SETTINGS, the loss guard's measurement.
--mode edit: one setting; per target the selection and the loss guard (its edit alone at --floor, measured by
  inference; the directions halved while dF / dR < --guard_ratio, at most --guard_steps times, else none), the edit of
  every target, its joint change, and the edited model saved in float32 to --model_dir for phase two
  (recast_tofu_gh200.sh with BASE_ID=<model_dir>, VARIANTS=(unrestricted), S_MINS=(1.0)).
  --family plain, gcontrast or gmoment (user 2026-11-04) instead builds the restricted warm start of recast_tofu.py with
  that basis (plain; gradient contrast at the plain basis's rank; per-example moments at their own EVR ranks), its
  forget gradient projected off the protected span; with protection each target also reports the unprotected update
  as the control against a recast_tofu run's per-target ranks and |dW| / |W|.
  The selective rule with gradient statistics (user 2026-11-04): --family grad takes the input-side Gram of the mean
  gradients, G_f^T G_f against G_r^T G_r (the gradient contrast's input side); --family fisher the per-sequence Fisher
  input moments (the per-example gradient moments E[g^T g]); --retain_stat act replaces the family's retain side by
  the act retain covariance over --retain_tokens (S2's, question and answer tokens). --loo yes measures every target's
  contribution within the joint edit (the joint change minus the change without it) and its edit alone;
  --retain_loss_tokens nonprefix (user 2026-11-05) puts the retain loss behind the gradient statistics (fisher, grad)
  on the question and answer tokens, S2's retain tokens, the forget loss staying on the answers.
  --loo_guard_ratio guards on that contribution as --guard_ratio does on the edit alone. --report_floors measures
  the joint edit at further floors.
Forget statistics over forget10, retain statistics over all of retain90; the NLL yardstick on the 400 forget and 400
fixed retain90 examples (seed 0, as 25559333), mean per-example answer NLL in float32.
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recast_selective as S  # noqa: E402
from recast_core import select_targets  # noqa: E402

PENCILS = [  # label, family, retain tokens of the act pencil, prefix protection, kappa
    ("act_old", "act", "answer", False, None),
    ("act_np", "act", "nonprefix", False, None),
    ("act_np_prefix", "act", "nonprefix", True, None),
    ("act_np_prefix_k1", "act", "nonprefix", True, 1.0),
    ("act_np_prefix_k4", "act", "nonprefix", True, 4.0),
    ("act_np_prefix_k16", "act", "nonprefix", True, 16.0),
    ("fisher", "fisher", None, False, None),
    ("fisher_prefix", "fisher", None, True, None),
    ("fisher_prefix_k1", "fisher", None, True, 1.0),
    ("fisher_prefix_k4", "fisher", None, True, 4.0),
    ("fisher_prefix_k16", "fisher", None, True, 16.0),
]
LAMBDAS = [1.0, 1.5, 2.0, 4.0, 8.0]  # act, rho 1 (25349441 / specificity_spectrum): lambda_1 5-17, lambda >= 2 for 80-155
# directions of a down block and 8-10 of an o block; the old best input contrast took about 400 / 60 per block
GRID_FLOORS = [0.0, 0.3]
GUARD_SETTINGS = [("act_np_prefix_k4", 1.0), ("fisher_prefix_k4", 1.0)]
CHECK_FRACTION = 0.05
CHECK_TARGETS = ["model.layers.0.mlp.down_proj", "model.layers.1.mlp.down_proj", "model.layers.2.mlp.down_proj",
                 "model.layers.3.mlp.down_proj", "model.layers.0.self_attn.o_proj", "model.layers.1.self_attn.o_proj"]
LAMBDA_QUANTILES = [0, 10, 100, 1000]  # report lambda at these 0-based ranks
RESTRICTED_FAMILIES = ("plain", "gcontrast", "gmoment")  # edit: restricted warm starts with the protected source


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", required=True, choices=["diagnose", "edit"])
    p.add_argument("--base_id", default="open-unlearning/tofu_Llama-3.2-1B-Instruct_full")
    p.add_argument("--out_json", required=True)
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--family", choices=["act", "fisher", "grad"] + list(RESTRICTED_FAMILIES),
                   help="edit: the contrast act, fisher or grad (G^T G of the mean gradients), or a restricted warm "
                        "start of the contract's form (rows "
                        "at max(cos(W_i, -G_r,i), floor), two-sided projection) with its forget gradient projected "
                        "off the protected span (user 2026-11-04): plain (SVD of the mean forget gradient at --evr), "
                        "gcontrast (the forget-vs-retain gradient contrast at --rho, --sides, the plain basis's rank) "
                        "or gmoment (the per-example forget-gradient moments, each side at its own --evr rank)")
    p.add_argument("--evr", type=float, help="edit, plain / gcontrast / gmoment: the EVR of the basis")
    p.add_argument("--sides", choices=["both", "input"], default="both", help="edit, gcontrast: the contrast sides")
    p.add_argument("--retain_tokens", choices=["answer", "nonprefix"], default="nonprefix",
                   help="edit, act (or --retain_stat act): the retain tokens of the pencil (the kappa protection "
                        "always uses nonprefix)")
    p.add_argument("--retain_stat", choices=["own", "act"], default="own",
                   help="edit, fisher / grad: the pencil's retain side, the family's own retain statistic or the act "
                        "retain covariance over --retain_tokens")
    p.add_argument("--retain_loss_tokens", choices=["answer", "nonprefix"], default="answer",
                   help="edit, fisher / grad: the tokens of the retain loss behind the gradient statistics, answer "
                        "(OpenUnlearning's labels) or nonprefix (question and answer); forget stays on answer tokens")
    p.add_argument("--loo", choices=["yes", "no"], default="no",
                   help="edit, act / fisher / grad: each target's contribution within the joint edit and alone")
    p.add_argument("--loo_guard_ratio", type=float, default=0.0,
                   help="edit: dF / dR threshold on each target's contribution within the joint edit, 0 = off "
                        "(halvings as --guard_steps)")
    p.add_argument("--report_floors", default="", help="edit: comma-separated further floors of the joint measurement")
    p.add_argument("--prefix", choices=["yes", "no"], default="yes", help="edit: protect the prefix inputs' span")
    p.add_argument("--kappa", type=float, help="edit: protect retain eigenvectors above kappa / n (none: off)")
    p.add_argument("--lambda_star", type=float, help="edit: the selectivity threshold")
    p.add_argument("--floor", type=float, help="edit: s in W - (1 - s) (W V) V^T")
    p.add_argument("--guard_ratio", type=float, default=2.0, help="edit: dF / dR threshold of the loss guard, 0 = off")
    p.add_argument("--guard_steps", type=int, default=4, help="edit: halvings of a target's directions at most")
    p.add_argument("--model_dir", help="edit: where the edited model is saved (float32)")
    args = p.parse_args()
    if args.mode == "edit" and (args.family is None or args.floor is None or not args.model_dir
                                or not 0.0 <= args.floor < 1.0):
        p.error("edit needs --family, --floor in [0, 1) and --model_dir")
    restricted = args.mode == "edit" and args.family in RESTRICTED_FAMILIES
    if restricted and (args.evr is None or not 0.0 < args.evr <= 1.0 or args.guard_ratio != 0
                       or args.lambda_star is not None):
        p.error(f"{args.family} needs --evr in (0, 1] and --guard_ratio 0, and takes no --lambda_star")
    if args.mode == "edit" and not restricted and (args.lambda_star is None or args.evr is not None):
        p.error("act and fisher need --lambda_star and take no --evr")
    if args.mode == "edit" and args.guard_ratio < 0 or args.guard_steps < 0 or args.loo_guard_ratio < 0:
        p.error("--guard_ratio, --loo_guard_ratio and --guard_steps must be >= 0")
    if restricted and (args.loo == "yes" or args.loo_guard_ratio > 0 or args.retain_stat != "own"
                       or args.retain_loss_tokens != "answer"):
        p.error(f"{args.family} takes no --loo, --loo_guard_ratio, --retain_stat or --retain_loss_tokens")
    if args.retain_loss_tokens != "answer" and not (args.family in ("fisher", "grad") and args.retain_stat == "own"):
        p.error("--retain_loss_tokens applies to the own retain statistic of fisher or grad")
    try:
        args.report_floors = [float(x) for x in args.report_floors.split(",") if x.strip()]
    except ValueError:
        p.error("--report_floors takes comma-separated numbers")
    if any(not 0.0 <= f < 1.0 for f in args.report_floors):
        p.error("--report_floors must lie in [0, 1)")

    from engram.benchmarks import _tofu_official as T

    device = "cuda"
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    result = {"args": vars(args), "mode": args.mode}

    def dump():
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=1)

    forget = load_dataset("locuslab/TOFU", "forget10")["train"]
    retain = load_dataset("locuslab/TOFU", "retain90")["train"]
    retain_eval = retain.select(sorted(torch.randperm(len(retain), generator=torch.Generator().manual_seed(0))[:400]
                                       .tolist()))
    tok = AutoTokenizer.from_pretrained(args.base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_id, torch_dtype=torch.float32,
                                                 attn_implementation="sdpa").to(device)
    model.eval()
    targets = select_targets(model, "down_o")
    names = list(targets)

    def feats(b):
        return {k: b[k].to(device) for k in ("input_ids", "attention_mask", "labels")}

    def loader(ds, bs=8):
        return DataLoader(T.QADataset(ds, tok, akey="answer"), batch_size=bs, collate_fn=T.Collator(tok, "right"))

    items = [T.QADataset(ds, tok, akey="answer")[i]["input_ids"].tolist() for ds in (forget, retain)
             for i in range(len(ds))]
    prefix_len = S.common_prefix_length(items)
    result["prefix"] = {"length": prefix_len, "tokens": tok.convert_ids_to_tokens(items[0][:prefix_len])}
    log(f"constant prefix: {prefix_len} tokens: {''.join(result['prefix']['tokens'])!r}")
    del items

    # ---------------- statistics ----------------
    cov_f, cnt_f, pre_f, dev_f = S.collect_position_statistics(model, loader(forget), targets, feats, prefix_len,
                                                               ("answer",))
    cov_r, cnt_r, pre_r, dev_r = S.collect_position_statistics(model, loader(retain), targets, feats, prefix_len,
                                                               ("answer", "nonprefix"))
    prefix_inputs = pre_r
    same = max(float((pre_f[n] - pre_r[n]).abs().max() / pre_r[n].abs().max()) for n in names)
    result["statistics"] = {"forget_tokens": cnt_f, "retain_tokens": cnt_r,
                            "prefix_deviation_within": max(max(dev_f.values()), max(dev_r.values())),
                            "prefix_deviation_forget_vs_retain": same}
    log(f"covariances: forget {cnt_f}, retain {cnt_r}; prefix inputs equal across sequences within "
        f"{result['statistics']['prefix_deviation_within']:.2e}, forget vs retain {same:.2e}")
    torch.cuda.empty_cache()

    def feats_retain(b):  # the retain loss of the gradient statistics, on answers or on question and answer tokens
        x = feats(b)
        if args.retain_loss_tokens == "nonprefix":
            x["labels"] = S.nonprefix_labels(x["input_ids"], x["attention_mask"], x["labels"], prefix_len)
        return x

    need_fisher = args.mode == "diagnose" or args.family == "fisher"
    own_retain = args.mode == "diagnose" or args.retain_stat == "own"  # the family's own retain statistic
    fisher_f = fisher_r = None
    if need_fisher:
        fisher_f = fisher_input_moments(model, targets, names, loader(forget), feats, device)
        if own_retain:
            fisher_r = fisher_input_moments(model, targets, names, loader(retain), feats_retain, device)
        log("per-sequence Fisher input moments" + ("" if own_retain else " (forget only)"))
    grads = None
    if restricted or args.mode == "edit" and args.family == "grad":  # as recast_tofu.py: batches of 8, retain first
        from recast_core import collect_mean_gradients
        g_r, n_r = collect_mean_gradients(model, loader(retain), targets, feats_retain)
        g_f, n_f = collect_mean_gradients(model, loader(forget), targets, feats)
        grads = {"forget": g_f, "retain": g_r}
        result["gradients"] = {"forget_examples": n_f, "retain_examples": n_r}
        log(f"mean gradients: forget {n_f}, retain {n_r} examples")

    eval_batches = {"forget": list(loader(forget, 16)), "retain": list(loader(retain_eval, 16))}
    orig = {n: targets[n].weight.detach().clone() for n in names}

    @torch.no_grad()
    def mean_nll(batches):
        total, count = 0.0, 0
        for b in batches:
            x = feats(b)
            logits = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).logits[:, :-1].float()
            labels = x["labels"][:, 1:]
            nll = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction="none")
            valid = (labels != -100).float()
            per_example = (nll * valid).sum(1) / valid.sum(1).clamp_min(1.0)
            total += float(per_example.sum())
            count += per_example.numel()
        return total / count

    @torch.no_grad()
    def evaluate(edits, floor=0.0):
        """Mean answer NLL on forget and retain with {target: V} shrunk to the floor; weights restored after."""
        for n, v in edits.items():
            targets[n].weight.copy_(S.input_shrink(orig[n], v.to(device), floor))
        res = {"forget": mean_nll(eval_batches["forget"]), "retain": mean_nll(eval_batches["retain"])}
        for n in edits:
            targets[n].weight.copy_(orig[n])
        return res

    base = evaluate({})
    result["baseline"] = base
    log(f"baseline answer NLL forget {base['forget']:.4f}, retain {base['retain']:.4f}")

    def point(res, directions):
        df, dr = res["forget"] - base["forget"], res["retain"] - base["retain"]
        return {"directions": directions, "d_forget": df, "d_retain": dr, "ratio": df / dr if dr > 0 else None}

    @torch.no_grad()
    def evaluate_deltas(deltas):
        """Mean answer NLL on forget and retain with W + delta in the given targets; weights restored after."""
        for n, d in deltas.items():
            targets[n].weight.copy_(orig[n] + d.to(device))
        res = {"forget": mean_nll(eval_batches["forget"]), "retain": mean_nll(eval_batches["retain"])}
        for n in deltas:
            targets[n].weight.copy_(orig[n])
        return res

    spectra = {}

    def protection(n, prefix, kappa):
        """The protected span of target n (prefix inputs, retain eigenvectors above kappa / n) and its report."""
        if kappa is not None and n not in spectra:
            spectra[n] = S.retain_spectrum(cov_r[n]["nonprefix"].to(device))
        b, info = S.protected_span(targets[n].in_features, kappa=kappa,
                                   prefix_inputs=prefix_inputs[n].to(device) if prefix else None,
                                   spectrum=spectra.get(n) if kappa is not None else None)
        return b.to(device), info

    def gram(g):
        """The input-side Gram G^T G of a mean gradient (out x in), float64 on the device."""
        g = g.to(device).double()
        return g.transpose(0, 1) @ g

    def pencil_inputs(n, family, retain_tokens, prefix, kappa, retain_stat="own"):
        """The pencil's forget and retain statistics: act C_f (answer tokens) and C_r (retain_tokens); fisher the
        per-sequence Fisher input moments; grad the Gram of the mean gradients. retain_stat act takes C_r over
        retain_tokens as the retain side of any family."""
        own = retain_stat == "own"
        if family == "act":
            num, den = cov_f[n]["answer"], cov_r[n][retain_tokens]
        elif family == "fisher":
            num, den = fisher_f[n], (fisher_r[n] if own else None)
        else:
            num, den = gram(grads["forget"][n]), (gram(grads["retain"][n]) if own else None)
        if not own:
            den = cov_r[n][retain_tokens]
        b, info = protection(n, prefix, kappa)
        return num.to(device), den.to(device), b, info

    def select(n, family, retain_tokens, prefix, kappa, lambda_star, retain_stat="own"):
        num, den, b, info = pencil_inputs(n, family, retain_tokens, prefix, kappa, retain_stat)
        v, lam = S.selective_directions(num, den, b, args.rho, lambda_star)
        rep = S.direction_report(v, cov_f[n]["answer"].to(device), cov_r[n]["nonprefix"].to(device),
                                 prefix_inputs[n].to(device))
        rep.update(info)
        rep["lambda_at"] = {str(i): float(lam[i]) for i in LAMBDA_QUANTILES if i < lam.numel()}
        rep["lambda_ge"] = {str(t): int((lam >= t).sum()) for t in LAMBDAS}
        return v.float().cpu(), rep

    if args.mode == "diagnose":
        diagnose(args, model, tok, targets, names, prefix_len, prefix_inputs, cov_f, cov_r, eval_batches, feats,
                 evaluate, point, select, result, dump, log, device)
    elif restricted:
        edit_restricted(args, model, tok, targets, names, orig, grads, protection, prefix_inputs, evaluate_deltas,
                        point, lambda spans: S.collect_example_moments(model, loader(forget), targets, feats, spans),
                        result, dump, log, device)
    else:
        edit(args, model, tok, targets, names, orig, evaluate, point, select, result, dump, log, device)
    log(f"[write] {args.out_json}")


def fisher_input_moments(model, targets, names, data, feats, device):
    """Per target the input side of the per-sequence empirical Fisher, E_e[g_e^T g_e] with g_e = B sum_t delta_t x_t^T
    (as 25559333 and recast_core.collect_example_gradient_moments), float32 on the CPU (accumulated in float64)."""
    flags = [(p, p.requires_grad) for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    for n in names:
        targets[n].weight.requires_grad_(True)
    saved = {}
    hooks = [targets[n].register_forward_hook(lambda mod, inp, out, n=n: saved.__setitem__(n, (inp[0].detach(), out)))
             for n in names]
    seq = {n: torch.zeros(targets[n].in_features, targets[n].in_features, dtype=torch.float64, device=device)
           for n in names}
    count = 0
    try:
        for batch in data:
            x_in = feats(batch)
            size = int(x_in["input_ids"].shape[0])
            loss = model(**x_in).loss
            deltas = torch.autograd.grad(loss, [saved[n][1] for n in names], allow_unused=True)
            mask = x_in["attention_mask"].unsqueeze(-1).float()
            for n, d in zip(names, deltas):
                if d is None:
                    continue
                x = saved[n][0].float() * mask
                d = d.float() * mask * float(size)
                seq[n].add_(x.flatten(0, 1).transpose(0, 1) @ (d @ d.transpose(1, 2) @ x).flatten(0, 1))
            saved.clear()
            count += size
    finally:
        for h in hooks:
            h.remove()
        for p, flag in flags:
            p.requires_grad_(flag)
    out = {n: (seq[n] / count).float().cpu() for n in names}
    del seq
    torch.cuda.empty_cache()
    return out


def diagnose(args, model, tok, targets, names, prefix_len, prefix_inputs, cov_f, cov_r, eval_batches, feats, evaluate,
             point, select, result, dump, log, device):
    # 1a. massive activations
    probe = feats({k: v[:8] for k, v in eval_batches["retain"][0].items()})
    masks = S.position_masks(probe["attention_mask"], probe["labels"], prefix_len)
    down_in = {}
    hooks = [m.register_forward_pre_hook(lambda mod, a, i=i: down_in.__setitem__(i, a[0].detach()))
             for i, m in enumerate(mod for n, mod in targets.items() if n.endswith("down_proj"))]
    with torch.no_grad():
        out = model(input_ids=probe["input_ids"], attention_mask=probe["attention_mask"], output_hidden_states=True)
    for h in hooks:
        h.remove()

    def summary(t):
        a = t.abs().float()
        per_pos = a[:, :prefix_len].amax(0)  # (prefix, dim): equal over the batch
        val, dim = per_pos.max(-1)
        row = {"prefix_max": [round(float(v), 2) for v in val], "prefix_argmax_dim": [int(d) for d in dim]}
        for cls in ("prompt", "answer"):
            sel = a[masks[cls]]
            m, idx = sel.max(-1)
            j = int(m.argmax())
            row[f"{cls}_max"], row[f"{cls}_argmax_dim"] = float(m[j]), int(idx[j])
            row[f"{cls}_median"] = float(sel.median())
        return row

    result["massive"] = {"residual": [summary(h) for h in out.hidden_states],
                         "down_input": [summary(down_in[i]) for i in sorted(down_in)]}
    for lay, row in enumerate(result["massive"]["residual"]):
        top = max(range(prefix_len), key=lambda i: row["prefix_max"][i])
        log(f"hidden state {lay} (0 = embeddings): prefix max {row['prefix_max'][top]:.1f} at position {top} "
            f"({result['prefix']['tokens'][top]!r}, dim {row['prefix_argmax_dim'][top]}), prompt max "
            f"{row['prompt_max']:.1f}, answer max {row['answer_max']:.1f} (median {row['answer_median']:.3f})")
    for blk, row in enumerate(result["massive"]["down_input"]):
        top = max(range(prefix_len), key=lambda i: row["prefix_max"][i])
        log(f"down_proj input block {blk}: prefix max {row['prefix_max'][top]:.1f} at position {top} "
            f"(neuron {row['prefix_argmax_dim'][top]}), prompt max {row['prompt_max']:.1f}, answer max "
            f"{row['answer_max']:.1f} (median {row['answer_median']:.4f})")
    del out, down_in
    dump()

    # 1b. prefix energy along the old activation contrast, 1c. position-restricted removal
    old = {}
    result["prefix_energy"] = {}
    for n in names:
        v, rep = select(n, "act", "answer", False, None, None)
        k = math.ceil(CHECK_FRACTION * v.shape[0])
        v = v[:, :k]
        old[n] = v
        x = prefix_inputs[n].double()
        e_prefix = float((x @ v.double()).square().sum() / x.shape[0])
        ca = cov_f[n]["answer"].double()
        cr = cov_r[n]["answer"].double()
        vd = v.double()
        result["prefix_energy"][n] = {
            "directions": k, "prefix_share": float((x @ vd).square().sum() / x.square().sum()),
            "prefix_per_token": e_prefix, "forget_answer_per_token": float(torch.trace(vd.T @ ca @ vd)),
            "retain_answer_per_token": float(torch.trace(vd.T @ cr @ vd))}
        r = result["prefix_energy"][n]
        log(f"prefix energy {n}: share {r['prefix_share']:.4f}, per token prefix {r['prefix_per_token']:.3e} vs answer "
            f"forget {r['forget_answer_per_token']:.3e} / retain {r['retain_answer_per_token']:.3e}")
    dump()

    result["position_removal"] = {}
    for n in CHECK_TARGETS:
        v = old[n].to(device)
        result["position_removal"][n] = {}
        for cls in ("all", "prefix", "nonprefix"):
            res = evaluate_positions(model, targets[n], v, cls, prefix_len, eval_batches, feats)
            result["position_removal"][n][cls] = point(res, v.shape[1])
        log(f"position removal {n}: " + "; ".join(f"{c} dF {q['d_forget']:+.4f} dR {q['d_retain']:+.4f}"
                                                  for c, q in result["position_removal"][n].items()))
        dump()
    del old

    # 2. selection grid
    result["grid"] = {}
    for label, family, rtok, prefix, kappa in PENCILS:
        sel, reps = {}, {}
        for n in names:
            sel[n], reps[n] = select(n, family, rtok, prefix, kappa, min(LAMBDAS))
        lam_rows = {}
        for star in LAMBDAS:
            edits = {n: sel[n][:, :reps[n]["lambda_ge"][str(star)]] for n in names}
            total = sum(v.shape[1] for v in edits.values())
            lam_rows[str(star)] = {"directions": total,
                                   "per_target": {n: v.shape[1] for n, v in edits.items()},
                                   **{f"floor_{f:g}": point(evaluate(edits, f), total) for f in GRID_FLOORS}}
        result["grid"][label] = {"family": family, "retain_tokens": rtok, "prefix": prefix, "kappa": kappa,
                                 "targets": {n: {k: v for k, v in r.items() if k != "directions"}
                                             for n, r in reps.items()},
                                 "lambda": lam_rows}
        log(f"grid {label}: " + "; ".join(
            f"l* {s}: {r['directions']} dirs, " + ", ".join(
                f"s {f:g} dF {r[f'floor_{f:g}']['d_forget']:+.3f} dR {r[f'floor_{f:g}']['d_retain']:+.3f}"
                for f in GRID_FLOORS) for s, r in lam_rows.items()))
        dump()
        del sel
        torch.cuda.empty_cache()

    # 3. per-target edits alone
    result["per_target"] = {}
    for label, star in GUARD_SETTINGS:
        _, family, rtok, prefix, kappa = next(p for p in PENCILS if p[0] == label)
        rows = {}
        for n in names:
            v, rep = select(n, family, rtok, prefix, kappa, star)
            rows[n] = (point(evaluate({n: v}), v.shape[1]) if v.shape[1] else
                       {"directions": 0, "d_forget": 0.0, "d_retain": 0.0, "ratio": None})
        result["per_target"][f"{label}_l{star:g}"] = rows
        log(f"per target {label} l* {star:g}: " + "; ".join(
            f"{n.replace('model.layers.', '')} {r['directions']} dF {r['d_forget']:+.4f} dR {r['d_retain']:+.4f}"
            for n, r in rows.items()))
        dump()


@torch.no_grad()
def evaluate_positions(model, target, v, cls, prefix_len, eval_batches, feats):
    """Answer NLL (forget, retain) with the input directions v removed from one target's input at the positions of a
    class only (all, prefix or nonprefix), through a pre-hook on the target (x <- x - (x V) V^T there)."""
    state = {"mask": None}

    def hook(module, a):
        x = a[0]
        m = state["mask"].unsqueeze(-1).to(x.dtype)
        return (x - m * ((x @ v) @ v.transpose(0, 1)),) + tuple(a[1:])

    handle = target.register_forward_pre_hook(hook)
    out = {}
    try:
        for split, batches in eval_batches.items():
            total, count = 0.0, 0
            for b in batches:
                x = feats(b)
                masks = S.position_masks(x["attention_mask"], x["labels"], prefix_len)
                state["mask"] = (torch.ones_like(masks["prefix"]) if cls == "all" else
                                 masks["prefix"] if cls == "prefix" else ~masks["prefix"])
                logits = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).logits[:, :-1].float()
                labels = x["labels"][:, 1:]
                nll = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction="none")
                valid = (labels != -100).float()
                per_example = (nll * valid).sum(1) / valid.sum(1).clamp_min(1.0)
                total += float(per_example.sum())
                count += per_example.numel()
            out[split] = total / count
    finally:
        handle.remove()
    return out


def edit(args, model, tok, targets, names, orig, evaluate, point, select, result, dump, log, device):
    kappa = args.kappa
    prefix = args.prefix == "yes"
    rows, chosen = {}, {}
    loo = args.loo == "yes" or args.loo_guard_ratio > 0

    def fmt(t):
        return f"{t['directions']} dF {t['d_forget']:+.4f} dR {t['d_retain']:+.4f}"

    for n in names:
        v, rep = select(n, args.family, args.retain_tokens, prefix, kappa, args.lambda_star, args.retain_stat)
        trials = []
        k = v.shape[1]
        if args.guard_ratio > 0:
            steps = 0
            while k > 0:
                q = point(evaluate({n: v[:, :k]}, args.floor), k)
                trials.append(q)
                if S.guard_passes(q, args.guard_ratio):
                    break
                if steps == args.guard_steps:
                    k = 0
                    break
                k //= 2
                steps += 1
        chosen[n] = v[:, :k]
        rows[n] = {**{key: val for key, val in rep.items()}, "selected": rep["directions"], "kept": k,
                   "guard_trials": trials}
        if loo:  # the edit alone, for comparison with the contribution within the joint edit
            rows[n]["alone"] = (trials[-1] if trials and trials[-1]["directions"] == k else
                                point(evaluate({n: chosen[n]}, args.floor), k) if k > 0 else None)
        log(f"{n}: {rep['directions']} selective directions (protected {rep['protected']}: kappa "
            f"{rep['kappa_directions']}, prefix {rep['prefix_rank']}), kept {k} after the guard; "
            + "; ".join(fmt(t) for t in trials)
            + (f"; alone {fmt(rows[n]['alone'])}" if loo and rows[n]["alone"] and not trials else ""))
        result["targets"] = rows
        dump()
    if loo:
        def measure(sel):
            return point(evaluate(sel, args.floor), sum(v.shape[1] for v in sel.values()))

        kept, report = S.leave_one_out(chosen, measure, args.loo_guard_ratio, args.guard_steps)
        result["loo_full"] = report["full"]
        log(f"joint edit before the leave-one-out guard: {fmt(report['full'])}")
        for n in names:
            rows[n]["loo"] = report["targets"][n]
            log(f"{n}: contribution within the joint edit " + "; ".join(fmt(t) for t in report["targets"][n]["trials"])
                + f", kept {kept[n]} of {chosen[n].shape[1]}")
        chosen = {n: chosen[n][:, :kept[n]] for n in names}
        result["targets"] = rows
        dump()
    joint = point(evaluate(chosen, args.floor), sum(v.shape[1] for v in chosen.values()))
    result["joint"] = joint
    log(f"joint edit at floor {args.floor:g}: {joint['directions']} directions, dF {joint['d_forget']:+.4f}, dR "
        f"{joint['d_retain']:+.4f}")
    if args.report_floors:
        result["joint_floors"] = {str(f): point(evaluate(chosen, f), joint["directions"]) for f in args.report_floors}
        log("joint edit at the report floors: "
            + "; ".join(f"{f}: {fmt(q)}" for f, q in result["joint_floors"].items()))
    with torch.no_grad():
        for n, v in chosen.items():
            targets[n].weight.copy_(S.input_shrink(orig[n], v.to(device), args.floor))
    os.makedirs(args.model_dir, exist_ok=True)
    model.save_pretrained(args.model_dir)
    tok.save_pretrained(args.model_dir)
    torch.save({n: v.cpu() for n, v in chosen.items()}, os.path.join(args.model_dir, "selective_directions.pt"))
    result["model_dir"] = args.model_dir
    log(f"[save] edited model (float32), tokenizer and directions -> {args.model_dir}")
    dump()


def edit_restricted(args, model, tok, targets, names, orig, grads, protection, prefix_inputs, evaluate_deltas, point,
                    collect_moments, result, dump, log, device):
    """A restricted warm start (--family plain, gcontrast or gmoment) with the protected span, every target, then the
    joint change and the save. Without protection each is recast_tofu.py's restricted_shrink warm start with that basis
    (forget_only source, full-space alignment; gcontrast at --restriction_rank_multiplier 1, gmoment at its own EVR
    ranks); with protection every target also reports that unprotected update (ranks, kept forget-gradient energy,
    |dW| / |W|, prefix output change), the control against the per-target report of a recast_tofu run."""
    prefix = args.prefix == "yes"
    spans = {n: protection(n, prefix, args.kappa) for n in names}
    moments = None
    if args.family == "gmoment":
        moments, count = collect_moments({n: b for n, (b, _) in spans.items()})
        result["moments"] = {"forget_examples": count}
        log(f"per-example forget-gradient moments: {count} examples")

    def update(n, b):
        gf, gr = grads["forget"][n].to(device), grads["retain"][n].to(device)
        if args.family == "plain":
            return S.plain_restricted_update(orig[n], gf, gr, args.evr, args.floor, b)
        if args.family == "gcontrast":
            return S.contrast_restricted_update(orig[n], gf, gr, args.evr, args.floor, args.rho, args.sides, b)
        m_out, m_in, m_out_protected = moments[n]
        return S.moment_restricted_update(orig[n], gf, gr, (m_out if b is None else m_out_protected, m_in),
                                          args.evr, args.floor, b)

    def prefix_change(n, d):
        x = prefix_inputs[n].to(device).float()
        return float(((d @ x.transpose(0, 1)).norm() / (orig[n] @ x.transpose(0, 1)).norm().clamp_min(1e-30)))

    rows, deltas = {}, {}
    for n in names:
        b, info = spans[n]
        protected = b if b.shape[1] > 0 else None
        d, rep = update(n, protected)
        deltas[n] = d
        rows[n] = {**rep, **info, "prefix_output_change": prefix_change(n, d)}
        if protected is not None:  # the control: the same basis without protection
            d0, rep0 = update(n, None)
            rows[n]["unprotected"] = {k: rep0[k] for k in ("rank", "rank_in", "kept_forget_gradient_energy",
                                                           "weight_rel_change")}
            rows[n]["unprotected"]["prefix_output_change"] = prefix_change(n, d0)
            del d0
        if moments is not None:
            del moments[n]
        torch.cuda.empty_cache()
        ctrl = rows[n].get("unprotected")
        log(f"{n}: rank {rep['rank']} / {rep['rank_in']} (protected {info['protected']}: kappa "
            f"{info['kappa_directions']}, prefix {info['prefix_rank']}), forget gradient kept "
            f"{rep['kept_forget_gradient_energy']:.3f} (source share after the projection "
            f"{rep['source_share_after_projection']:.3f}), |dW| / |W| {rep['weight_rel_change']:.4f}, rows at the "
            f"floor {rep['shrink_at_floor']}, prefix output change {rows[n]['prefix_output_change']:.2e}"
            + (f"; unprotected: rank {ctrl['rank']} / {ctrl['rank_in']}, |dW| / |W| {ctrl['weight_rel_change']:.4f}, "
               f"prefix output change {ctrl['prefix_output_change']:.2e}" if ctrl else ""))
        result["targets"] = rows
        dump()
    joint = point(evaluate_deltas(deltas), sum(r["rank"] for r in rows.values()))
    result["joint"] = joint
    log(f"joint edit at floor {args.floor:g}: rank sum {joint['directions']}, dF {joint['d_forget']:+.4f}, dR "
        f"{joint['d_retain']:+.4f}")
    with torch.no_grad():
        for n, d in deltas.items():
            targets[n].weight.copy_(orig[n] + d)
    os.makedirs(args.model_dir, exist_ok=True)
    model.save_pretrained(args.model_dir)
    tok.save_pretrained(args.model_dir)
    result["model_dir"] = args.model_dir
    log(f"[save] edited model (float32) and tokenizer -> {args.model_dir}")
    dump()


if __name__ == "__main__":
    main()
