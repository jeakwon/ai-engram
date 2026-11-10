"""Distributions of the first-order quantities behind the signed shrink, from a --component_diagnostics run.

Reads ``components.json`` and ``results.json`` of one recast_tofu.py run and, for its targets, recomputes the
forget and retain mean gradients at Init Fine. (as the run) to report, per module type and over all targets:

- the component contributions c_j = <G_f, D_j>: quantiles, mean, sum;
- the cosine of each component with the forget gradient, cos_j = c_j / (sigma_j ||G_f||) (||D_j|| = sigma_j), and
  its root mean square times sqrt(out * in): about 1 for directions as aligned with G_f as random ones;
- per target: the predicted forget / retain change <G_f, Delta W>, <G_r, Delta W>, the cosine of the update with
  G_f and G_r, ||G_f||, ||G_r||, ||Delta W||, and the share of ||G_f||^2 inside the restriction (the basis evr).

Positive control: the recomputed ||G_f|| of every target equals the run's to 1e-4, or the script stops.

Usage (GH200 node, ai-engram-gh200 environment, offline cache):
    python replication/scripts/first_order_stats.py --run_dir <recast run dir> --out_json <file>
"""

import argparse
import json
import math
import os
import platform
import statistics

from recast_core import collect_mean_gradients, parse_blocks, select_targets
from table3_tofu import Stage, git_info, to_jsonable

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def quantiles(values):
    v = sorted(values)
    pick = lambda q: v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]  # noqa: E731
    return {"n": len(v), "mean": statistics.mean(v), "std": statistics.pstdev(v), "min": v[0], "q05": pick(0.05),
            "q25": pick(0.25), "median": pick(0.5), "q75": pick(0.75), "q95": pick(0.95), "max": v[-1]}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run_dir", required=True)
    p.add_argument("--out_json", required=True)
    args = p.parse_args()

    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from engram.benchmarks import _tofu_official as T

    with open(os.path.join(args.run_dir, "results.json")) as f:
        run = json.load(f)
    with open(os.path.join(args.run_dir, "components.json")) as f:
        comp = json.load(f)
    a = run["args"]
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    timings = {}
    out = {"args": vars(args), "run_args": a, "git": git_info(repo), "host": platform.node(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "timings_s": timings}
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(a["base_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a["base_id"], torch_dtype=torch.float32,
                                                 attn_implementation=a["attn_implementation"]).to(device)
    targets = select_targets(model, a["targets"], blocks=parse_blocks(a["target_blocks"]))
    targets = {n: m for n, m in targets.items() if n in comp}

    def feats(b):
        return {k: b[k].to(device) for k in ("input_ids", "attention_mask", "labels")}

    def loader(split):
        ds = load_dataset(a["dataset_id"], split)["train"]
        return DataLoader(T.QADataset(ds, tok, akey="answer"), batch_size=a["grad_batch_size"],
                          collate_fn=T.Collator(tok, "right"))

    with Stage("gradients", timings):
        g_r, _ = collect_mean_gradients(model, loader(a["retain_split"]), targets, feats)
        g_f, _ = collect_mean_gradients(model, loader(a["forget_split"]), targets, feats)
    norms = {n: {"f": float(g_f[n].norm()), "r": float(g_r[n].norm())} for n in targets}
    bad = [n for n in targets if abs(norms[n]["f"] - comp[n]["grad_forget_norm"]) > 1e-4 * comp[n]["grad_forget_norm"]]
    if bad:
        raise SystemExit(f"recomputed ||G_f|| differs from the run at {bad[:3]}")
    print(f"[control] ||G_f|| of all {len(targets)} targets equals the run's")

    per_target, by_type = {}, {}
    for n, c in comp.items():
        w = targets[n].weight
        size = w.shape[0] * w.shape[1]
        cos = [cj / (s * c["grad_forget_norm"]) for s, cj in zip(c["sigma"], c["contribution"]) if s > 0]
        rms = math.sqrt(sum(x * x for x in cos) / len(cos)) if cos else 0.0
        per_target[n] = {
            "grad_forget_norm": norms[n]["f"], "grad_retain_norm": norms[n]["r"], "update_norm": c["original_norm"],
            "predicted_forget": c["contribution_sum"], "predicted_retain": c["predicted_retain_change"],
            "cos_update_forget": c["contribution_sum"] / (c["original_norm"] * norms[n]["f"]) if c["original_norm"] else 0.0,
            "cos_update_retain": c["predicted_retain_change"] / (c["original_norm"] * norms[n]["r"]) if c["original_norm"] else 0.0,
            "subspace_forget_gradient_share": run["warm_start"]["per_target"][n]["evr"],
            "component_cos_rms_over_random": rms * math.sqrt(size), "rank": run["warm_start"]["per_target"][n]["rank"]}
        proj = next(q for q in PROJECTIONS if n.endswith(q))
        for key in (proj, "all"):
            t = by_type.setdefault(key, {"c": [], "cos": [], "cos_scaled": []})
            t["c"] += c["contribution"]
            t["cos"] += cos
            t["cos_scaled"] += [x * math.sqrt(size) for x in cos]
    summary = {}
    for key, t in by_type.items():
        names = [n for n in per_target if key == "all" or n.endswith(key)]
        summary[key] = {
            "c": quantiles(t["c"]), "cos": quantiles(t["cos"]), "cos_scaled": quantiles(t["cos_scaled"]),
            "cos_rms_over_random": math.sqrt(sum(x * x for x in t["cos_scaled"]) / len(t["cos_scaled"])),
            **{k: quantiles([per_target[n][k] for n in names]) for k in (
                "grad_forget_norm", "grad_retain_norm", "update_norm", "predicted_forget", "predicted_retain",
                "cos_update_forget", "cos_update_retain", "subspace_forget_gradient_share")},
            "predicted_forget_sum": sum(per_target[n]["predicted_forget"] for n in names),
            "predicted_retain_sum": sum(per_target[n]["predicted_retain"] for n in names)}
        s = summary[key]
        print(f"[stats] {key:9s} c_j: median {s['c']['median']:+.2e} q05 {s['c']['q05']:+.2e} q95 {s['c']['q95']:+.2e} "
              f"sum {sum(t['c']):+.4f}; cos_j*sqrt(D) rms {s['cos_rms_over_random']:.3f} (q05 {s['cos_scaled']['q05']:+.2f} "
              f"q95 {s['cos_scaled']['q95']:+.2f}); per target ||G_f|| median {s['grad_forget_norm']['median']:.3f} "
              f"[{s['grad_forget_norm']['min']:.3f}, {s['grad_forget_norm']['max']:.3f}], ||G_r|| median "
              f"{s['grad_retain_norm']['median']:.3f} [{s['grad_retain_norm']['min']:.3f}, {s['grad_retain_norm']['max']:.3f}], "
              f"||Delta W|| median {s['update_norm']['median']:.2f}; predicted dL_f per target median "
              f"{s['predicted_forget']['median']:+.2e} [{s['predicted_forget']['min']:+.2e}, {s['predicted_forget']['max']:+.2e}] "
              f"sum {s['predicted_forget_sum']:+.4f}, dL_r sum {s['predicted_retain_sum']:+.4f}; cos(Delta W, G_f) median "
              f"{s['cos_update_forget']['median']:+.2e}, cos(Delta W, G_r) median {s['cos_update_retain']['median']:+.2e}; "
              f"G_f share in subspace median {s['subspace_forget_gradient_share']['median']:.3f}")
    out.update({"summary": summary, "per_target": per_target})
    with open(args.out_json, "w") as f:
        json.dump(to_jsonable(out), f, indent=2)
    print(f"[done] {args.out_json}")


if __name__ == "__main__":
    main()
