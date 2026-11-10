"""Signs of the first-order forget contributions c_j = <G_f, D_j> of the ReCAST update components, by module type.

For every decoder projection type (q, k, v, o, gate, up, down; ``--targets``) and basis (plain, D with the gradient
output side, both at the explicit budget ``--rank_multiplier``), the warm start is applied to copies of the Init
Fine. weights with ``component_diagnostics`` (recast_core.decompose_update) and the script reports, per type: the
number of components with c_j > 0 / < 0, the update energy fraction with c_j < 0, the positive and negative sums of
c_j, the fraction of the |c_j| mass that is positive, the energy fraction with c_j < 0 among the 10 % largest
components (by sigma), and per block the energy fraction with c_j < 0. Nothing is trained or evaluated.

With ``--update_models`` the same statistics describe the displacement Delta = W_model - W_Init of each listed
model (Retain, a saved Engram edit) on the same targets: ``<name>:full`` decomposes Delta itself (its SVD,
c_j = sigma_j p_j^T G_f q_j with the Init Fine. forget gradient), ``<name>:D`` its projection U U^T Delta V V^T onto
the D basis of the ReCAST update, with the share of ||Delta||^2 that the projection keeps.

With ``--loss_models`` it also gives the loss the source gradients differentiate (the example-weighted mean of each
batch's mean answer NLL) on forget and retain for each listed model (Init Fine., Retain), with the base tokenizer.

Positive control: in every target the components rebuild the update and sum c_j equals <G_f, Delta W> (the control
of recast_tofu.py), or the script stops.

Usage (GH200 node, ai-engram-gh200 environment, offline cache): see component_signs_gh200.sh.
"""

import argparse
import json
import os
import platform
import re

from recast_core import (collect_input_covariances, collect_mean_gradients, component_report, decompose_update,
                         parse_blocks, project, select_targets, warm_start)
from table3_tofu import Stage, git_info, to_jsonable

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def summarize(rows):
    energy = sum(r["energy"] for r in rows)
    pos_c = sum(r["contribution_positive"] for r in rows)
    neg_c = sum(r["contribution_negative"] for r in rows)
    top_e, top_neg = 0.0, 0.0
    for r in rows:
        k = max(1, len(r["sigma"]) // 10)
        for s, c in list(zip(r["sigma"], r["contribution"]))[:k]:  # sigma comes in descending order
            top_e += s * s
            top_neg += s * s if c < 0 else 0.0
    return {"targets": len(rows), "components": sum(r["components"] for r in rows),
            "positive": sum(r["positive"] for r in rows), "negative": sum(r["negative"] for r in rows),
            "energy_negative_fraction": sum(r["energy_negative_fraction"] * r["energy"] for r in rows) / energy
            if energy > 0 else 0.0,
            "contribution_positive": pos_c, "contribution_negative": neg_c,
            "abs_contribution_positive_fraction": pos_c / (pos_c - neg_c) if pos_c - neg_c > 0 else 0.0,
            "top_decile_energy_negative_fraction": top_neg / top_e if top_e > 0 else 0.0}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for k in ("--base_id", "--dataset_id", "--forget_split", "--retain_split", "--targets", "--target_blocks",
              "--attn_implementation", "--out_json"):
        p.add_argument(k, required=True)
    p.add_argument("--grad_batch_size", type=int, required=True)
    p.add_argument("--s_min", type=float, required=True)
    p.add_argument("--restriction_evr", type=float, required=True)
    p.add_argument("--rho", type=float, required=True)
    p.add_argument("--rank_multiplier", type=float, required=True)
    p.add_argument("--loss_models", nargs="*", default=[], help="model ids whose forget / retain loss to measure")
    p.add_argument("--update_models", nargs="*", default=[],
                   help="model ids whose displacement from --base_id to decompose on the same targets")
    args = p.parse_args()

    import torch
    import torch.nn as nn
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from engram.benchmarks import _tofu_official as T

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    timings = {}
    out = {"args": vars(args), "git": git_info(repo), "host": platform.node(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "timings_s": timings}
    device = "cuda"
    forget = load_dataset(args.dataset_id, args.forget_split)["train"]
    retain = load_dataset(args.dataset_id, args.retain_split)["train"]
    tok = AutoTokenizer.from_pretrained(args.base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def feats(b):
        return {k: b[k].to(device) for k in ("input_ids", "attention_mask", "labels")}

    def loader(ds):
        return DataLoader(T.QADataset(ds, tok, akey="answer"), batch_size=args.grad_batch_size,
                          collate_fn=T.Collator(tok, "right"))

    @torch.no_grad()
    def mean_loss(model, ds):
        model.eval()
        total, count = 0.0, 0
        for b in loader(ds):
            x = feats(b)
            size = int(x["input_ids"].shape[0])
            total += float(model(**x).loss) * size
            count += size
        return total / count

    losses = {}
    for model_id in args.loss_models:
        with Stage(f"losses {model_id}", timings):
            m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32,
                                                     attn_implementation=args.attn_implementation).to(device)
            losses[model_id] = {"forget": mean_loss(m, forget), "retain": mean_loss(m, retain)}
            del m
            torch.cuda.empty_cache()
        print(f"[loss] {model_id}: forget {losses[model_id]['forget']:.4f}, retain {losses[model_id]['retain']:.4f}")

    model = (AutoModelForCausalLM.from_pretrained(args.base_id, torch_dtype=torch.float32,
                                                  attn_implementation=args.attn_implementation).to(device))
    targets = select_targets(model, args.targets, blocks=parse_blocks(args.target_blocks))
    with Stage("gradients and covariances", timings):
        g_r, _ = collect_mean_gradients(model, loader(retain), targets, feats)
        g_f, _ = collect_mean_gradients(model, loader(forget), targets, feats)
        c_r, n_r = collect_input_covariances(model, loader(retain), targets, feats)
        c_f, n_f = collect_input_covariances(model, loader(forget), targets, feats)
    print(f"[data] answer tokens retain {n_r}, forget {n_f}; targets {len(targets)}")

    kinds = {"plain": {}, "D": dict(representation_output="gradient", forget_covariances=c_f, retain_covariances=c_r,
                                    whitening_rho=args.rho)}
    summary, per_block, bases = {}, {}, {}
    for kind, extra in kinds.items():
        with Stage(f"components {kind}", timings):
            work = {}
            for n, m in targets.items():
                lin = nn.Linear(m.in_features, m.out_features, bias=False, device=device, dtype=torch.float32)
                with torch.no_grad():
                    lin.weight.copy_(m.weight)
                work[n] = lin
            restrictions, report = warm_start(work, g_r, args.s_min, g_f, args.restriction_evr,
                                              rank_multiplier=args.rank_multiplier, component_diagnostics=True, **extra)
            if kind == "D":
                bases = {n: r.basis for n, r in restrictions.items()}
            del restrictions
            del work
            torch.cuda.empty_cache()
        comp = {n: r["components"] for n, r in report.items() if "components" in r}
        bad = [n for n, c in comp.items() if c["reconstruction_abs_error"] > 1e-5 * report[n]["proposal_norm"]
               or abs(c["contribution_sum"] - c["first_order_forget_direct"])
               > c["grad_forget_norm"] * c["reconstruction_abs_error"] + 1e-6 * abs(c["first_order_forget_direct"]) + 1e-12]
        if bad:
            raise SystemExit(f"{kind}: component control failed at {bad[:3]}")
        summary[kind] = {proj: summarize([c for n, c in comp.items() if n.endswith(proj)])
                         for proj in PROJECTIONS if any(n.endswith(proj) for n in comp)}
        summary[kind]["all"] = summarize(list(comp.values()))
        per_block[kind] = {n: {"block": int(re.search(r"layers\.(\d+)\.", n).group(1)), "rank": report[n]["rank"],
                               "energy_negative_fraction": c["energy_negative_fraction"],
                               "positive": c["positive"], "components": c["components"]} for n, c in comp.items()}
        for proj, s in summary[kind].items():
            blocks = sorted(v["energy_negative_fraction"] for n, v in per_block[kind].items() if n.endswith(proj)) \
                if proj != "all" else []
            print(f"[signs] {kind:5s} {proj:9s} components {s['components']:6d} c>0 {s['positive']:6d} c<0 "
                  f"{s['negative']:6d}; energy c<0 {s['energy_negative_fraction']:.4f} (top decile "
                  f"{s['top_decile_energy_negative_fraction']:.4f}); sum c+ {s['contribution_positive']:.4e} c- "
                  f"{s['contribution_negative']:.4e}, |c| positive {s['abs_contribution_positive_fraction']:.4f}"
                  + (f"; per block energy c<0 min {blocks[0]:.3f} median {blocks[len(blocks) // 2]:.3f} max "
                     f"{blocks[-1]:.3f}" if blocks else ""))
    kept_share = {}
    for model_id in args.update_models:
        label = os.path.basename(model_id.rstrip("/"))
        with Stage(f"displacement {label}", timings):
            other = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32,
                                                         attn_implementation=args.attn_implementation)
            o_mods = dict(other.named_modules())
            comp = {f"{label}:full": {}, f"{label}:D": {}}
            norms = {f"{label}:full": {}, f"{label}:D": {}}
            for n, m in targets.items():
                delta = (o_mods[n].weight.detach().to(device, torch.float32) - m.weight.detach().float())
                eye = torch.eye(delta.shape[1], device=device, dtype=torch.float64)
                for kind, (u, vh) in ((f"{label}:full", (None, eye)), (f"{label}:D", (bases[n].u, bases[n].vh))):
                    if kind.endswith(":D") and not bases[n].active:
                        continue
                    part = delta if u is None else project(delta, u, vh)
                    c = decompose_update(part, u, vh, g_f[n])
                    ones = torch.ones_like(c.sigma)
                    applied = part.double()
                    comp[kind][n] = component_report(c, ones, part, applied, g_f[n], g_r[n], keep_arrays=True)
                    norms[kind][n] = (float(part.norm()) ** 2, float(delta.norm()) ** 2)
                del eye
            del other, o_mods
            torch.cuda.empty_cache()
        for kind, rows in comp.items():
            bad = [n for n, c in rows.items() if c["reconstruction_abs_error"] > 1e-5 * max(c["original_norm"], 1e-12)
                   or abs(c["contribution_sum"] - c["first_order_forget_direct"])
                   > c["grad_forget_norm"] * c["reconstruction_abs_error"] + 1e-6 * abs(c["first_order_forget_direct"]) + 1e-12]
            if bad:
                raise SystemExit(f"{kind}: component control failed at {bad[:3]}")
            summary[kind] = {proj: summarize([c for n, c in rows.items() if n.endswith(proj)])
                             for proj in PROJECTIONS if any(n.endswith(proj) for n in rows)}
            summary[kind]["all"] = summarize(list(rows.values()))
            kept_share[kind] = {proj: sum(v[0] for n, v in norms[kind].items() if proj == "all" or n.endswith(proj))
                                / max(sum(v[1] for n, v in norms[kind].items() if proj == "all" or n.endswith(proj)), 1e-300)
                                for proj in list(summary[kind])}
            per_block[kind] = {n: {"block": int(re.search(r"layers\.(\d+)\.", n).group(1)),
                                   "energy_negative_fraction": c["energy_negative_fraction"], "positive": c["positive"],
                                   "components": c["components"], "update_norm": c["original_norm"]}
                               for n, c in rows.items()}
            for proj, s_ in summary[kind].items():
                print(f"[signs] {kind} {proj:9s} components {s_['components']:6d} c>0 {s_['positive']:6d} c<0 "
                      f"{s_['negative']:6d}; energy c<0 {s_['energy_negative_fraction']:.4f} (top decile "
                      f"{s_['top_decile_energy_negative_fraction']:.4f}); sum c+ {s_['contribution_positive']:.4e} c- "
                      f"{s_['contribution_negative']:.4e}, |c| positive {s_['abs_contribution_positive_fraction']:.4f}; "
                      f"share of ||Delta||^2 {kept_share[kind][proj]:.4f}")
    print("[control] components rebuild every update and sum c_j = <G_f, Delta W> at every target")
    out.update({"losses": losses, "summary": summary, "per_target": per_block, "displacement_share": kept_share})
    with open(args.out_json, "w") as f:
        json.dump(to_jsonable(out), f, indent=2)
    print(f"[done] {args.out_json}")


if __name__ == "__main__":
    main()
