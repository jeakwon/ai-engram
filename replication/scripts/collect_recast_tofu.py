"""Aggregate ReCAST-on-TOFU runs by configuration: mean and sample standard deviation over seeds.

Reads every ``results.json`` under the results root, groups runs whose arguments are equal apart
from the seed and the bookkeeping fields, and prints one row per configuration with n, the mean
and the standard deviation of each Table 3 column, plus the per-seed values. A seed run more than
once (a repeat of a deterministic run) counts once: the run with the latest Slurm job is kept and
the others are listed as repeats. Standard library only, so it runs anywhere.

Usage:
    python replication/scripts/collect_recast_tofu.py --root <experiment_results>/ai-engram/recast_tofu \
        [--table3 <.../table3.json>] [--min_seeds 1] [--out summary.json]
"""

import argparse
import glob
import json
import os
import statistics

COLUMNS = ["Memorization", "Utility", "Privacy", "EM", "FQ", "Overall"]
# eval_trajectory and the component, score and core diagnostics and save_model_dir only add measurements or files; the
# trained model is the same
BOOKKEEPING = {"seed", "out_dir", "run_name", "wandb_tags", "wandb_mode", "wandb_project", "anchors_raw", "eval_trajectory",
               "component_diagnostics", "score_diagnostics", "core_diagnostics", "contraction_diagnostics",
               "save_model_dir"}
GATE_KEYS = ("basis", "rho", "candidates", "optim", "lr", "max_epochs", "convergence_epochs", "tol", "drift_max",
             "probe_retain", "micro_batch", "grad_accum", "alpha", "gamma", "beta", "autocast", "binarize_share",
             "binary_s", "lambda_star", "init", "cut", "cut_s", "rotate_evr", "lr_schedule", "probe_every")
SHOWN = ["base_id", "phase_one_dtype", "train_dtype", "variant", "targets", "target_blocks", "s_min", "restriction_evr",
         "restriction_source",
         "restriction_alignment_scope", "shrink_layer_scaling", "phase2_free", "untie_lm_head", "train_lm_head", "lr",
         "lr_scheduler", "lr_points", "epochs", "max_grad_norm", "retain_loss", "kl_weight", "retain_input_filter",
         "retain_filter_kappa", "restriction_whitening", "whitening_rho", "restriction_representation",
         "representation_mix", "restriction_rank_multiplier", "restriction_forget_output_target", "component_selection",
         "warm_start_scale", "warm_start_scale_target", "warm_start_scale_by_type", "direction_weighting",
         "deepspeed_config", "optim", "per_device_batch_size", "grad_accum", "gradient_checkpointing",
         "loss_normalization", "shrink_score", "weight_contraction", "contraction_strength", "contraction_residual",
         *(f"direction_gates_{k}" for k in GATE_KEYS)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True)
    p.add_argument("--table3", help="table3.json of the replication run, for the Engram reference rows")
    p.add_argument("--min_seeds", type=int, default=1, help="only print configurations with at least this many runs")
    p.add_argument("--out", help="optional JSON file for the aggregated rows")
    return p.parse_args()


def with_defaults(run_args):
    """Runs older than an argument ran its present default; fill it in so equal configurations group together."""
    a = dict(run_args)
    gated = a.get("variant") == "direction_gates"  # no ReCAST update: no shrink or restriction arguments
    defaults = {"retain_loss": "nll", "target_blocks": "all", "phase_one_dtype": a.get("train_dtype"),
                "gradient_checkpointing": "no", "loss_normalization": "window_tokens",
                **({} if gated else {"shrink_score": "original"})}
    if a.get("variant") not in ("unrestricted", "direction_gates"):
        defaults.update({"retain_input_filter": "none", "restriction_whitening": "none",
                         "restriction_representation": "none", "shrink_layer_scaling": "none",
                         "restriction_alignment_scope": "full_space"})
    if a.get("variant") == "restricted_semu":
        defaults["phase2_free"] = "none"
    for k, v in defaults.items():
        if a.get(k) is None:
            a[k] = v
    return a


def main():
    args = parse_args()
    groups = {}
    for path in sorted(glob.glob(os.path.join(args.root, "*", "results.json"))):
        with open(path) as f:
            run = json.load(f)
        if "scores" not in run:  # a run that stopped before evaluation
            continue
        run["args"] = with_defaults(run["args"])
        key = tuple(sorted((k, json.dumps(v)) for k, v in run["args"].items()  # an unset argument = an absent one
                           if k not in BOOKKEEPING and v is not None))
        groups.setdefault(key, []).append(run)

    rows = []
    for key, runs in groups.items():
        by_seed = {}
        for r in sorted(runs, key=lambda r: int(r["runtime"]["slurm_job_id"] or 0)):
            by_seed.setdefault(r["args"]["seed"], []).append(r)
        repeats = [f"{r['runtime']['slurm_job_id']}" for rs in by_seed.values() for r in rs[:-1]]
        runs = [rs[-1] for _, rs in sorted(by_seed.items())]
        if len(runs) < args.min_seeds:
            continue
        cfg = {k: json.loads(v) for k, v in key}
        stats = {}
        for c in COLUMNS:
            values = [r["scores"][c] for r in runs]
            stats[c] = {"mean": statistics.mean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "values": values}
        rows.append({"config": {k: cfg.get(k) for k in SHOWN}, "n": len(runs),
                     "seeds": [r["args"]["seed"] for r in runs],
                     "jobs": [f"{r['runtime']['slurm_job_id']}" for r in runs], "repeats_dropped": repeats,
                     "stats": stats})
    rows.sort(key=lambda r: -r["stats"]["Overall"]["mean"])

    for r in rows:
        c = r["config"]
        print(f"n={r['n']} seeds={r['seeds']} {c['variant']} targets={c['targets']} blocks={c['target_blocks'] or 'all'} s_min={c['s_min']} "
              f"evr={c['restriction_evr']} untie={c['untie_lm_head']} train_head={c['train_lm_head']} "
              f"lr={c['lr']} sched={c['lr_scheduler']} points={c['lr_points']} epochs={c['epochs']} "
              f"clip={c['max_grad_norm']} source={c['restriction_source']} align={c['restriction_alignment_scope'] or 'full_space'} "
              f"scaling={c['shrink_layer_scaling'] or 'none'} "
              f"free={c['phase2_free'] or ('none' if c['variant'] == 'restricted_semu' else '-')} "
              f"loss={c['retain_loss'] or 'nll'}{'' if c['kl_weight'] is None else c['kl_weight']} "
              f"filter={c['retain_input_filter'] or 'none'}{'' if c['retain_filter_kappa'] is None else c['retain_filter_kappa']} "
              f"whiten={c['restriction_whitening'] or 'none'}{'' if c['whitening_rho'] is None else c['whitening_rho']} "
              f"repr={c['restriction_representation'] or 'none'}{'' if c['representation_mix'] is None else '/mix' + str(c['representation_mix'])} "
              f"rankx={c['restriction_rank_multiplier'] or '-'} qf={c['restriction_forget_output_target'] or '-'} "
              f"components={c['component_selection'] or '-'} scale={c['warm_start_scale_target'] or c['warm_start_scale'] or '-'} "
              f"dirw={c['direction_weighting'] or '-'}"
              + ("" if c["variant"] != "direction_gates" else
                 " gates=" + "/".join(f"{k}:{c['direction_gates_' + k]}" for k in GATE_KEYS))
              + (f" (repeats dropped: {', '.join(r['repeats_dropped'])})" if r["repeats_dropped"] else ""))
        print("    " + "  ".join(f"{col} {r['stats'][col]['mean']:.4f}±{r['stats'][col]['std']:.4f}" for col in COLUMNS))
        print("    Overall per seed: " + ", ".join(f"{v:.4f}" for v in r["stats"]["Overall"]["values"]))
    if args.table3:
        with open(args.table3) as f:
            ref = json.load(f)["scores"]
        for name in ("Engram (alpha=0.6)", "Engram (alpha_W-Norm)"):
            print(f"reference {name}: " + "  ".join(f"{col} {ref[name][col]:.4f}" for col in COLUMNS))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
