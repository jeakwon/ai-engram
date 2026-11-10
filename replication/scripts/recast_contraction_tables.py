"""Weight-component contraction on the down modules: top and random components against ReCAST at the same displacement.

Selects the down-only runs (targets mlp_down) of the common recipe of recast_stage1_tables.py at any EVR and groups them
by (EVR, rank multiplier, floor, arm, strength), the arm being the weight contraction mode (top, random; with
``_residual<r>`` when contracted components keep a singular value) or recast (no contraction), the latest run per seed. For each group: the scores after phase two (per seed, mean and sample std), the
forget / retain answer NLL before and after phase two (trajectory, mean over seeds), the whole displacement ||dW|| and
||P(W)|| from the warm-start report, and the contraction report of recast_tofu.py (means over seeds). For each pair of
arms at the same (EVR, rank, floor): a Welch t-test of every score. Writes JSON and prints the tables.

Usage (Slurm, ai-engram-gh200 environment): python recast_contraction_tables.py --root <recast_tofu results> --out <json>
"""

import argparse
import glob
import json
import math
import os

from scipy import stats as scipy_stats

from recast_stage1_tables import COLUMNS, RECIPE, mean_std

ARMS = ("top", "random", "recast")


def selected(a):
    a = {**a, "phase_one_dtype": a.get("phase_one_dtype") or a.get("train_dtype")}  # older runs: one dtype for both
    if a.get("targets") != "mlp_down" or any(a.get(k) != v for k, v in RECIPE.items() if k != "restriction_evr"):
        return False
    return (a.get("direction_weighting") is None and a.get("deepspeed_config") is None
            and a.get("component_selection") is None and a.get("warm_start_scale") is None
            and a.get("warm_start_scale_target") is None and a.get("warm_start_scale_by_type") is None
            and (a.get("shrink_score") or "original") == "original" and float(a.get("epochs", -1)) == 5.0)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    groups = {}
    for f in glob.glob(os.path.join(args.root, "*", "results.json")):
        with open(f) as fh:
            r = json.load(fh)
        a = r.get("args", {})
        if "scores" not in r or not selected(a):
            continue
        if a.get("write_compensation") is not None:
            continue
        arm = a.get("weight_contraction") or "recast"
        if a.get("output_contrast_rho") is not None:
            arm = f"outcontrast{a['output_contrast_rho']:g}"
        if a.get("contraction_residual"):
            arm = f"{arm}_residual{a['contraction_residual']:g}"
        key = (a["restriction_evr"], a.get("restriction_rank_multiplier"), json.dumps(a["s_min"]), arm,
               a.get("contraction_strength"))
        prev = groups.setdefault(key, {}).get(a["seed"])
        if prev is None or int(r["runtime"]["slurm_job_id"]) > int(prev["runtime"]["slurm_job_id"]):
            groups[key][a["seed"]] = r

    rows = []
    for (evr, m, s_min, arm, strength), by_seed in sorted(groups.items(), key=lambda kv: str(kv[0])):
        seeds = sorted(by_seed)
        rs = [by_seed[s] for s in seeds]
        values = {c: [x["scores"][c] for x in rs] for c in COLUMNS}
        report = rs[0]["warm_start"]["per_target"]
        row = {"evr": evr, "rank_multiplier": m, "floor": json.loads(s_min), "arm": arm, "strength": strength,
               "n": len(rs), "seeds": seeds, "run_dirs": [os.path.basename(x["args"]["out_dir"]) for x in rs],
               "values": values, "after": {c: mean_std(v) for c, v in values.items()},
               "update_norm": math.sqrt(sum(t["applied_norm"] ** 2 for t in report.values())),
               "projected_norm": math.sqrt(sum(t.get("projected_weight_norm", float("nan")) ** 2 for t in report.values())),
               "rank_mean": sum(t["rank"] for t in report.values()) / len(report)}
        traj = [x["trajectory"] for x in rs if x.get("trajectory")]
        if traj:
            row["nll"] = {f"{side}_{when}": sum(t[i][side]["nll"] for t in traj) / len(traj)
                          for side in ("forget", "retain", "real_authors", "world_facts")
                          for when, i in (("warm_start", 0), ("after", -1))}
        ct = [x["contraction"] for x in rs if "contraction" in x]
        if ct:
            row["contraction"] = {k: sum(c[k] for c in ct) / len(ct) for k, v in ct[0].items()
                                  if isinstance(v, (int, float)) and not isinstance(v, bool)
                                  and all(c.get(k) is not None for c in ct)}
        rows.append(row)

    pairs = []
    cells = {}
    for r in rows:
        cells.setdefault((r["evr"], r["rank_multiplier"], json.dumps(r["floor"])), {})[r["arm"]] = r
    for (evr, m, floor), arms in sorted(cells.items(), key=lambda kv: str(kv[0])):
        names = [a for a in ARMS if a in arms] + sorted(a for a in arms if a not in ARMS)
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                rx, ry = arms[x], arms[y]
                if rx["n"] < 2 or ry["n"] < 2:
                    continue
                welch = {}
                for c in COLUMNS:
                    t, pval = scipy_stats.ttest_ind(rx["values"][c], ry["values"][c], equal_var=False)
                    welch[c] = {"difference": rx["after"][c]["mean"] - ry["after"][c]["mean"], "t": float(t), "p": float(pval)}
                pairs.append({"evr": evr, "rank_multiplier": m, "floor": json.loads(floor), "a": x, "b": y, "welch": welch})

    with open(args.out, "w") as fh:
        json.dump({"rows": rows, "pairs": pairs}, fh, indent=1)

    for r in rows:
        a, ct, nll = r["after"], r.get("contraction"), r.get("nll", {})
        print(f"evr {r['evr']} m {r['rank_multiplier']} floor {r['floor']} {r['arm']:6s} n {r['n']}: "
              + " / ".join(f"{a[c]['mean']:.4f}" for c in COLUMNS) + f" +- {a['Overall']['std']:.4f}"
              + f" | ||dW|| {r['update_norm']:.2f} of ||P(W)|| {r['projected_norm']:.2f}, rank {r['rank_mean']:.1f}"
              + (f" | NLL forget {nll['forget_warm_start']:.2f} -> {nll['forget_after']:.2f}, retain "
                 f"{nll['retain_warm_start']:.2f} -> {nll['retain_after']:.3f}" if nll else "")
              + (f" | contracted {ct['contracted_mean']:.1f} of {ct['components_mean']:.1f}, kappa rank "
                 f"{ct['kappa_rank_mean']:.3f} (weighted {ct['kappa_rank_weighted']:.3f}), input contrast moved "
                 f"{ct['input_contrast_moved']:.4f} vs all {ct['input_contrast_all']:.4f}" if ct else ""))
    for x in pairs:
        w = x["welch"]
        print(f"[pair] evr {x['evr']} m {x['rank_multiplier']} floor {x['floor']}: {x['a']} - {x['b']} "
              + ", ".join(f"{c} {w[c]['difference']:+.4f} (p {w[c]['p']:.3f})" for c in COLUMNS))
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
