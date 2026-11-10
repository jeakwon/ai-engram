"""Experiment 2: the response of losses and scores to the strength t of a fixed ReCAST update.

The D update (``--direction full``) or its signed version (``--direction signed``, components with c_j > 0) is
computed once at Init Fine. for every target of the profile; for every t the model W(t) = W_Init + t Delta W is
rebuilt from the Init weights (all targets with the same t) and measured without any optimizer:

- A, response curve over ``--t_grid``: forget / retain loss as the source gradients define it (example-weighted
  mean of batch-mean answer NLL) and their change from Init; KL(Init || W(t)) per answer token on forget and on
  retain_perturbed; the full TOFU scores of a bfloat16 copy (as the runs score); the displacement ||t Delta W||
  over all targets and per projection type.
- B, local diagnostics: the first-order prediction t g^T Delta; central finite differences of phi(t) at t = 0 for
  every step in ``--fd_steps`` (first and second directional derivatives); the directional curvature
  Delta^T H Delta by Hessian-vector products (double backward with math attention, batch ``--hvp_batch_size``);
  the second-order prediction t g^T Delta + t^2 / 2 Delta^T H Delta and the remainders.
- C, signed against full: one job per direction; the curves compare by t and by displacement.

Controls (stop on failure unless marked logged): t = 0 restores the Init weights exactly and the Init losses;
||W(t) - W_Init|| = t ||Delta W|| at every t; g^T Delta from the stored gradients equals the sum of the per-target
first-order terms of warm_start, and the first-order term accumulated during the Hessian-vector pass; the forget
loss at t = 1 equals ``--reference_t1_forget_loss`` (the warm start of the runs) to 1e-3; the grid is evaluated in a
shuffled order and its first value again at the end (difference logged).

Usage (GH200 node, ai-engram-gh200 environment, offline cache): see contraction_response_gh200.sh.
"""

import argparse
import copy
import json
import os
import platform
import random

from recast_core import (collect_input_covariances, collect_mean_gradients, parse_blocks, select_targets,
                         warm_start)
from table3_tofu import COLUMNS, Stage, git_info, to_jsonable

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for k in ("--base_id", "--dataset_id", "--forget_split", "--retain_split", "--holdout_split", "--targets",
              "--target_blocks", "--attn_implementation", "--anchors_raw", "--out_json"):
        p.add_argument(k, required=True)
    p.add_argument("--direction", required=True, choices=["full", "signed"])
    p.add_argument("--grad_batch_size", type=int, required=True)
    p.add_argument("--eval_batch_size", type=int, required=True)
    p.add_argument("--hvp_batch_size", type=int, required=True)
    p.add_argument("--s_min", type=float, required=True)
    p.add_argument("--restriction_evr", type=float, required=True)
    p.add_argument("--rho", type=float, required=True)
    p.add_argument("--rank_multiplier", type=float, required=True)
    p.add_argument("--t_grid", type=float, nargs="+", required=True)
    p.add_argument("--fd_steps", type=float, nargs="+", required=True)
    p.add_argument("--eval_tofu", required=True, choices=["yes", "no"])
    p.add_argument("--reference_t1_forget_loss", type=float, help="forget loss of the warm start in the runs")
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from datasets import load_dataset
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from engram.benchmarks import _tofu_official as T

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    timings = {}
    out = {"args": vars(args), "git": git_info(repo), "host": platform.node(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "timings_s": timings, "controls": {}}
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    forget = load_dataset(args.dataset_id, args.forget_split)["train"]
    retain = load_dataset(args.dataset_id, args.retain_split)["train"]
    D = {"fp": load_dataset(args.dataset_id, f"{args.forget_split}_perturbed")["train"],
         "ho": load_dataset(args.dataset_id, args.holdout_split)["train"],
         "rp": load_dataset(args.dataset_id, "retain_perturbed")["train"],
         "ra": load_dataset(args.dataset_id, "real_authors_perturbed")["train"],
         "wf": load_dataset(args.dataset_id, "world_facts_perturbed")["train"]}
    with open(args.anchors_raw) as f:
        anchors = json.load(f)
    ft, rt = anchors["Init Fine."], anchors["Retain"]
    model = AutoModelForCausalLM.from_pretrained(args.base_id, torch_dtype=torch.float32,
                                                 attn_implementation=args.attn_implementation).to(device)
    model.eval()
    targets = select_targets(model, args.targets, blocks=parse_blocks(args.target_blocks))

    def feats(b):
        return {k: b[k].to(device) for k in ("input_ids", "attention_mask", "labels")}

    def loader(ds, bs):
        return DataLoader(T.QADataset(ds, tok, akey="answer"), batch_size=bs, collate_fn=T.Collator(tok, "right"))

    @torch.no_grad()
    def mean_loss(ds):
        total, count = 0.0, 0
        for b in loader(ds, args.grad_batch_size):
            x = feats(b)
            size = int(x["input_ids"].shape[0])
            total += float(model(**x).loss) * size
            count += size
        return total / count

    with Stage("gradients and covariances", timings):
        g_r, _ = collect_mean_gradients(model, loader(retain, args.grad_batch_size), targets, feats)
        g_f, _ = collect_mean_gradients(model, loader(forget, args.grad_batch_size), targets, feats)
        c_r, _ = collect_input_covariances(model, loader(retain, args.grad_batch_size), targets, feats)
        c_f, _ = collect_input_covariances(model, loader(forget, args.grad_batch_size), targets, feats)
    with Stage("update direction", timings):
        work = {}
        for n, m in targets.items():
            lin = nn.Linear(m.in_features, m.out_features, bias=False, device=device, dtype=torch.float32)
            with torch.no_grad():
                lin.weight.copy_(m.weight)
            work[n] = lin
        _, report = warm_start(work, g_r, args.s_min, g_f, args.restriction_evr, rank_multiplier=args.rank_multiplier,
                               representation_output="gradient", forget_covariances=c_f, retain_covariances=c_r,
                               whitening_rho=args.rho, component_diagnostics=True,
                               component_selection="signed" if args.direction == "signed" else None)
        delta = {n: (work[n].weight.detach() - targets[n].weight.detach()).float() for n in targets}
        del work, c_r, c_f
    w0 = {n: m.weight.detach().clone() for n, m in targets.items()}
    gtd = {"forget": sum(float((g_f[n].double() * delta[n].double()).sum()) for n in targets),
           "retain": sum(float((g_r[n].double() * delta[n].double()).sum()) for n in targets)}
    # Control: g^T Delta from the stored gradients equals warm_start's per-target first-order terms.
    per_target = sum(r["components"]["predicted_forget_change"] for r in report.values() if "components" in r)
    bound = sum(float(g_f[n].norm()) * float(delta[n].norm()) for n in targets) * 1e-5
    if abs(gtd["forget"] - per_target) > bound + 1e-9:
        raise SystemExit(f"g_f^T Delta {gtd['forget']} differs from the per-target terms {per_target}")
    out["controls"]["first_order_consistency"] = {"direct": gtd["forget"], "per_target": per_target, "bound": bound}
    del g_r, g_f
    torch.cuda.empty_cache()
    norm_full = sum(float(d.square().sum()) for d in delta.values()) ** 0.5
    norm_type = {p: sum(float(d.square().sum()) for n, d in delta.items() if n.endswith(p)) ** 0.5
                 for p in PROJECTIONS if any(n.endswith(p) for n in delta)}
    print(f"[direction] {args.direction}: ||Delta W|| {norm_full:.3f} ({json.dumps({k: round(v, 3) for k, v in norm_type.items()})}); "
          f"g_f^T Delta {gtd['forget']:+.5f}, g_r^T Delta {gtd['retain']:+.5f}")
    print(f"[control] g_f^T Delta equals the per-target first-order terms ({per_target:+.5f})")

    def set_t(t):
        with torch.no_grad():
            for n, m in targets.items():
                m.weight.copy_(w0[n] + float(t) * delta[n])

    with Stage("Init losses", timings):
        loss0 = {"forget": mean_loss(forget), "retain": mean_loss(retain)}
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def kl_to_init(ds):
        """KL(Init || model) per answer token (labels != -100), next-token distributions."""
        total, count = 0.0, 0
        for b in loader(ds, args.eval_batch_size):
            x = feats(b)
            mask = x["labels"][:, 1:] != -100
            lp = F.log_softmax(model(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).logits[:, :-1][mask].float(), -1)
            lq = F.log_softmax(ref(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).logits[:, :-1][mask].float(), -1)
            total += float((lq.exp() * (lq - lp)).sum())
            count += int(mask.sum())
        return total / count

    def tofu_scores():
        frozen = copy.deepcopy(model).to(torch.bfloat16).eval()
        raw = T.compute_full_tofu(frozen, tok, D, bs=args.eval_batch_size)
        del frozen
        torch.cuda.empty_cache()
        scores = T.evaluate_scores(raw, rt, ft)
        keep = ("exact_memorization", "forget_Q_A_Prob", "forget_Q_A_ROUGE", "forget_truth_ratio", "model_utility",
                "forget_Q_A_gibberish", "mia_loss")
        return scores, {k: raw[k] for k in keep}

    def measure(t, full_eval):
        set_t(t)
        row = {"t": t}
        moved = {n: float((targets[n].weight.detach() - w0[n]).norm()) for n in targets}
        row["displacement"] = sum(v * v for v in moved.values()) ** 0.5
        row["displacement_by_type"] = {p: sum(v * v for n, v in moved.items() if n.endswith(p)) ** 0.5 for p in norm_type}
        # Control: the displacement is t times the update, target by target.
        bad = [n for n in targets if abs(moved[n] - abs(t) * float(delta[n].norm())) > 1e-4 * float(delta[n].norm()) + 1e-6]
        if bad:
            raise SystemExit(f"displacement at t = {t} is not |t| ||Delta W|| at {bad[:3]}")
        loss = {"forget": mean_loss(forget), "retain": mean_loss(retain)}
        row.update({"loss": loss, "change": {s: loss[s] - loss0[s] for s in loss},
                    "first_order": {s: t * gtd[s] for s in loss}})
        if full_eval:
            row["kl_to_init"] = {"forget": kl_to_init(forget), "retain_perturbed": kl_to_init(D["rp"])}
            if args.eval_tofu == "yes":
                row["scores"], row["raw"] = tofu_scores()
        return row

    # A: the grid in a shuffled order, its first value again at the end.
    order = list(args.t_grid)
    random.Random(0).shuffle(order)
    rows = {}
    with Stage("response curve", timings):
        for t in order:
            rows[t] = measure(t, True)
            r = rows[t]
            print(f"[curve] t {t:.3f}: displacement {r['displacement']:.3f}; loss forget {r['loss']['forget']:.4f} "
                  f"({r['change']['forget']:+.4f}, first order {r['first_order']['forget']:+.5f}), retain "
                  f"{r['loss']['retain']:.4f} ({r['change']['retain']:+.4f}, {r['first_order']['retain']:+.5f}); KL "
                  f"forget {r['kl_to_init']['forget']:.4f} retain {r['kl_to_init']['retain_perturbed']:.4f}"
                  + ("; " + ", ".join(f"{c}={r['scores'][c]:.4f}" for c in COLUMNS) if "scores" in r else ""))
        again = measure(order[0], False)
    def save_partial(stage):
        out.update({"loss_init": loss0, "g_dot_delta": gtd, "update_norm": norm_full, "update_norm_by_type": norm_type,
                    "grid": [rows[t] for t in sorted(rows)], "completed_stage": stage})
        with open(args.out_json, "w") as f:
            json.dump(to_jsonable(out), f, indent=2)

    save_partial("response curve")
    diff = {s: again["loss"][s] - rows[order[0]]["loss"][s] for s in loss0}
    out["controls"]["order_independence"] = {"t": order[0], "difference": diff}
    print(f"[control] t {order[0]} evaluated again after the grid: loss difference forget {diff['forget']:+.2e}, "
          f"retain {diff['retain']:+.2e} (logged)")
    if 0.0 in rows:
        set_t(0.0)
        if not all(torch.equal(targets[n].weight, w0[n]) for n in targets):
            raise SystemExit("t = 0 does not restore the Init weights")
        if any(abs(rows[0.0]["loss"][s] - loss0[s]) > 1e-6 * max(1.0, loss0[s]) for s in loss0):
            raise SystemExit(f"t = 0 losses {rows[0.0]['loss']} differ from Init {loss0}")
        print("[control] t = 0 restores the Init weights and losses")
    if args.reference_t1_forget_loss is not None and 1.0 in rows:
        gap = abs(rows[1.0]["loss"]["forget"] - args.reference_t1_forget_loss)
        out["controls"]["t1_reference"] = {"loss": rows[1.0]["loss"]["forget"], "reference": args.reference_t1_forget_loss}
        if gap > 1e-3:
            raise SystemExit(f"t = 1 forget loss {rows[1.0]['loss']['forget']} differs from the runs' "
                             f"{args.reference_t1_forget_loss}")
        print(f"[control] t = 1 reproduces the warm start of the runs (forget loss {rows[1.0]['loss']['forget']:.4f})")

    # B: finite differences at t = 0 and the directional curvature by Hessian-vector products.
    fd = []
    with Stage("finite differences", timings):
        for h in args.fd_steps:
            plus, minus = measure(h, False), measure(-h, False)
            fd.append({"h": h, **{s: {"first": (plus["loss"][s] - minus["loss"][s]) / (2 * h),
                                      "second": (plus["loss"][s] - 2 * loss0[s] + minus["loss"][s]) / (h * h)}
                                  for s in loss0}})
            print(f"[fd] h {h:g}: forget phi' {fd[-1]['forget']['first']:+.5f} phi'' {fd[-1]['forget']['second']:+.4f}; "
                  f"retain phi' {fd[-1]['retain']['first']:+.5f} phi'' {fd[-1]['retain']['second']:+.4f} "
                  f"(g^T Delta {gtd['forget']:+.5f} / {gtd['retain']:+.5f})")
    out["finite_differences"] = fd
    save_partial("finite differences")
    set_t(0.0)
    hvp = {}
    params = [targets[n].weight for n in targets]
    flags = [(p, p.requires_grad) for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    with Stage("Hessian-vector products", timings), sdpa_kernel(SDPBackend.MATH):
        for side, ds in (("forget", forget), ("retain", retain)):
            curv, first, count = 0.0, 0.0, 0
            for b in loader(ds, args.hvp_batch_size):
                x = feats(b)
                size = int(x["input_ids"].shape[0])
                loss = model(**x).loss
                grads = torch.autograd.grad(loss, params, create_graph=True)
                s = sum((g * delta[n]).sum() for g, n in zip(grads, targets))
                hv = torch.autograd.grad(s, params)
                curv += size * float(sum((h_ * delta[n]).sum() for h_, n in zip(hv, targets)))
                first += size * float(s.detach())
                count += size
                del loss, grads, s, hv
            hvp[side] = {"curvature": curv / count, "first": first / count}
            print(f"[hvp] {side}: Delta^T H Delta {hvp[side]['curvature']:+.5f}, g^T Delta in the same pass "
                  f"{hvp[side]['first']:+.5f} (stored {gtd[side]:+.5f})")
    for p, flag in flags:
        p.requires_grad_(flag)
    # Logged: the loss is the example-weighted mean of batch-mean losses, so its gradient depends on how examples are
    # batched; with --hvp_batch_size equal to --grad_batch_size the HVP pass reproduces the stored g^T Delta.
    out["controls"]["hvp_first_order"] = {side: {"hvp_pass": hvp[side]["first"], "stored": gtd[side],
                                                 "relative_gap": abs(hvp[side]["first"] - gtd[side]) / max(abs(gtd[side]), 1e-12)}
                                          for side in hvp}
    print("[control] g^T Delta in the Hessian-vector pass vs stored (logged): " + ", ".join(
        f"{side} relative gap {v['relative_gap']:.2e}" for side, v in out["controls"]["hvp_first_order"].items())
        + f" (batch sizes {args.hvp_batch_size} / {args.grad_batch_size})")
    for r in rows.values():
        t = r["t"]
        r["second_order"] = {s: t * gtd[s] + 0.5 * t * t * hvp[s]["curvature"] for s in loss0}
        r["remainder_first"] = {s: r["change"][s] - r["first_order"][s] for s in loss0}
        r["remainder_second"] = {s: r["change"][s] - r["second_order"][s] for s in loss0}
    grid = [rows[t] for t in sorted(rows)]
    out.update({"loss_init": loss0, "g_dot_delta": gtd, "update_norm": norm_full, "update_norm_by_type": norm_type,
                "grid": grid, "finite_differences": fd, "hvp": hvp, "completed_stage": "all",
                "cuda_max_memory_gb": torch.cuda.max_memory_allocated() / 2 ** 30})
    with open(args.out_json, "w") as f:
        json.dump(to_jsonable(out), f, indent=2)
    print(f"[done] {args.out_json}")


if __name__ == "__main__":
    main()
