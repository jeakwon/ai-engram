"""The global input-contrast selection on down_proj + o_proj by direction order (score = lambda, energy_ratio), and the
size of the directions it removed.

Part A, every finished run (``results.json`` with scores) of ``restricted_shrink`` on ``down_o`` with the global
family ``input_contrast``, ridge 1, s 0.04, the energy budget unit and none of the per-direction s rules or KL terms:
Mem / Util / Priv / EM / FQ / Overall per (order, alpha, budget floor), mean and standard deviation over the seeds,
with the seeds and the job IDs; a seed run more than once counts once, the run with the latest Slurm job kept (as
collect_recast_tofu.py), the others listed as repeats.

Part B, for each run of ``--runs``: per projection type the directions removed (total, fewest and most per target),
the mean own energy e_j = ||W0 v_j||^2 of a removed direction (the displacement squared over beta^2 = (1 - s)^2 and
the count; the last, partly removed direction makes it a slight underestimate), sum ||W0||_F^2 (from the
displacement and the relative weight change of every target) and the relative squared displacement
R = ||dW||^2 / sum ||W0||^2 over all targets.

Usage (login node, no torch): python3 global_order_summary.py --root <recast_tofu results> --runs <dir> ...
"""

import argparse
import glob
import json
import os
import statistics

COLUMNS = ("Memorization", "Utility", "Privacy", "EM", "FQ", "Overall")
EXCLUDED = ("global_family_s_gamma", "global_family_s_rule", "global_family_s_threshold_gamma", "kl_teacher")


def comparable(a: dict) -> bool:
    return (a.get("variant") == "restricted_shrink" and a.get("targets") == "down_o"
            and a.get("global_family") == "input_contrast" and a.get("global_family_rho") == 1.0
            and a.get("global_family_floor") == 0.04 and a.get("global_family_budget_unit") in (None, "energy")
            and a.get("retain_loss") == "nll" and all(a.get(k) is None for k in EXCLUDED))


def part_a(root: str) -> list:
    groups = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "results.json"))):
        with open(path) as f:
            r = json.load(f)
        if "scores" not in r or not comparable(r["args"]):
            continue
        a = r["args"]
        key = (a.get("global_family_order") or "score", a.get("global_family_alpha"), a["s_min"])
        job = str(r["runtime"]["slurm_job_id"])
        groups.setdefault(key, []).append((a["seed"], job, r["scores"]))
    out = []
    for (order, alpha, floor), runs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][2])):
        by_seed = {}
        for run in sorted(runs, key=lambda t: int(t[1])):
            by_seed.setdefault(run[0], []).append(run)
        repeats = [j for rs in by_seed.values() for _, j, _ in rs[:-1]]
        runs = [rs[-1] for _, rs in sorted(by_seed.items())]
        row = {"order": order, "alpha": alpha, "budget_floor": floor, "seeds": [s for s, _, _ in runs],
               "jobs": [j for _, j, _ in runs], "repeats_dropped": repeats}
        for c in COLUMNS:
            xs = [sc[c] for _, _, sc in runs]
            row[c] = statistics.mean(xs)
            row[f"{c}_std"] = statistics.stdev(xs) if len(xs) > 1 else 0.0
        out.append(row)
    return out


def part_b(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "results.json")) as f:
        r = json.load(f)
    beta2 = (1.0 - r["global_family"]["global"]["floor"]) ** 2
    types = {}
    for name, v in r["warm_start"]["per_target"].items():
        g = v["global_family"]
        t = types.setdefault(name.rsplit(".", 1)[-1], {"directions": [], "e_sum": 0.0, "w2": 0.0})
        t["directions"].append(g["directions"])
        t["e_sum"] += g["displacement"] ** 2 / beta2
        t["w2"] += (g["displacement"] / v["weight_rel_change"]) ** 2 if v["weight_rel_change"] > 0 else 0.0
    w2 = sum(t["w2"] for t in types.values())
    return {"run": os.path.basename(run_dir), "order": r["args"].get("global_family_order"),
            "budget_floor": r["args"]["s_min"], "seed": r["args"]["seed"], "overall": r["scores"]["Overall"],
            "sum_w0_sq": w2, "relative_displacement2": r["global_family"]["global"]["displacement"] ** 2 / w2,
            "per_type": {k: {"directions": sum(t["directions"]), "per_target_min": min(t["directions"]),
                             "per_target_max": max(t["directions"]), "mean_e": t["e_sum"] / sum(t["directions"]),
                             "sum_w0_sq": t["w2"]} for k, t in types.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True)
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--out_json")
    args = p.parse_args()
    out = {"orders": part_a(args.root), "runs": [part_b(d) for d in args.runs]}
    for row in out["orders"]:
        print(f"{row['order']:>12} alpha {row['alpha']} floor {row['budget_floor']:<5} seeds {row['seeds']} "
              f"jobs {','.join(sorted(set(row['jobs'])))}"
              + (f" (repeats dropped {','.join(row['repeats_dropped'])})" if row["repeats_dropped"] else "") + ": " + " / ".join(f"{row[c]:.4f}" for c in COLUMNS[:4])
              + f" / {row['FQ']:.2f} / {row['Overall']:.4f} +- {row['Overall_std']:.4f}")
    for b in out["runs"]:
        print(f"{b['run']}: order {b['order']}, floor {b['budget_floor']}, seed {b['seed']}, Overall {b['overall']:.4f}; "
              f"sum ||W0||^2 {b['sum_w0_sq']:.0f}, R {b['relative_displacement2']:.4f}; "
              + "; ".join(f"{k}: {t['directions']} directions ({t['per_target_min']}-{t['per_target_max']} per target), "
                          f"mean e_j {t['mean_e']:.3f}" for k, t in b["per_type"].items()))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
