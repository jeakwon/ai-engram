"""Stage 1 and Stage 2 tables (shrink per module type, row-score rules) from the ReCAST run directories.

Selects the runs of the common recipe (restricted_shrink, basis D rho 30 gradient, EVR 0.97, no warm-start variant,
fp32_autocast phase two of 5 epochs with warm-up then linear to 0, clipping 1.0, from Init Fine.) and groups them by
(targets, floor, rank multiplier, row-score rule, epochs), the latest run per seed. For each group: the scores after
phase two (per seed, mean and sample std), a Welch t-test of each score against the reference group, the warm-start
scores of the matching warm-start-only run, the forget / retain answer NLL before and after phase two (trajectory, mean
over seeds), and from the warm-start report the whole update norm B, the per-type energies E_k = sum ||P(W_l)||^2 and
the mean relative weight change per type. Per-type floors get their budget B from the per-type energies of the uniform
warm start of the same targets and rank, and the uniform floor of the same budget
(``recast_core.uniform_floor_for_budget``); every uniform row also gives the matched-displacement floors of the other
targets and ranks. Writes JSON and prints the tables.

Usage (Slurm, ai-engram-gh200 environment): python recast_stage1_tables.py --root <recast_tofu results> --out <json>
"""

import argparse
import glob
import json
import math
import os

from scipy import stats as scipy_stats

from recast_core import update_budget, uniform_floor_for_budget

COLUMNS = ["Memorization", "Utility", "Privacy", "EM", "FQ", "Overall"]
RECIPE = {"variant": "restricted_shrink", "restriction_representation": "gradient", "whitening_rho": 30.0,
          "restriction_evr": 0.97, "restriction_source": "forget_only", "retain_loss": "nll", "train_dtype": "float32",
          "phase_one_dtype": "float32", "max_grad_norm": 1.0, "lr_points": "0:0,1:1e-5,5:0",
          "base_id": "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"}
REFERENCE = ("down_up_o", json.dumps(0.2), 1.25, "original")


def in_recipe(a):
    a = {**a, "phase_one_dtype": a.get("phase_one_dtype") or a.get("train_dtype")}  # older runs: one dtype for both
    if any(a.get(k) != v for k, v in RECIPE.items()):
        return False
    return (a.get("direction_weighting") is None and a.get("deepspeed_config") is None
            and a.get("component_selection") is None and a.get("warm_start_scale") is None
            and a.get("warm_start_scale_target") is None and a.get("warm_start_scale_by_type") is None
            and a.get("weight_contraction") is None and a.get("output_contrast_rho") is None
            and a.get("write_compensation") is None and a.get("family_contraction") is None
            and a.get("epochs") in (0, 0.0, 5, 5.0))


def mean_std(xs):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return {"mean": m, "std": sd}


def budget_of(report):
    """B, the per-type energies, the mean relative weight change and the mean share at the floor per type."""
    by_type = {}
    for name, r in report.items():
        by_type.setdefault(name.rsplit(".", 1)[-1], []).append(r)
    energies = {k: sum(r.get("projected_weight_norm", float("nan")) ** 2 for r in rs) for k, rs in by_type.items()}
    return {"B": math.sqrt(sum(r["applied_norm"] ** 2 for r in report.values())), "E": energies,
            "rel_change": {k: sum(r["weight_rel_change"] for r in rs) / len(rs) for k, rs in by_type.items()},
            "at_floor": {k: sum(r["shrink_at_floor"] for r in rs) / len(rs) for k, rs in by_type.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", help="comma-separated seeds to keep (e.g. 3,4,5 for a test on seeds no selection used)")
    args = p.parse_args()
    keep = None if args.seeds is None else {int(s) for s in args.seeds.split(",")}

    groups = {}
    for f in glob.glob(os.path.join(args.root, "*", "results.json")):
        with open(f) as fh:
            r = json.load(fh)
        a = r.get("args", {})
        if "scores" not in r or not in_recipe(a) or (keep is not None and a.get("seed") not in keep):
            continue
        if float(a["epochs"]) == 0 and not any("projected_weight_norm" in v for v in r["warm_start"]["per_target"].values()):
            continue  # warm starts before the report had ||P(W)||
        key = (a["targets"], json.dumps(a["s_min"], sort_keys=True), a.get("restriction_rank_multiplier"),
               a.get("shrink_score") or "original", float(a["epochs"]))
        prev = groups.setdefault(key, {}).get(a["seed"])
        if prev is None or int(r["runtime"]["slurm_job_id"]) > int(prev["runtime"]["slurm_job_id"]):
            groups[key][a["seed"]] = r

    rows = []
    for (targets, s_min, m, rule, epochs), by_seed in sorted(groups.items(), key=lambda kv: kv[0]):
        if epochs == 0:
            continue
        seeds = sorted(by_seed)
        rs = [by_seed[s] for s in seeds]
        values = {c: [x["scores"][c] for x in rs] for c in COLUMNS}
        row = {"key": [targets, s_min, m, rule], "targets": targets, "s_min": json.loads(s_min), "rank_multiplier": m,
               "shrink_score": rule, "n": len(rs), "seeds": seeds,
               "run_dirs": [os.path.basename(x["args"]["out_dir"]) for x in rs],
               "values": values, "after": {c: mean_std(v) for c, v in values.items()}}
        traj = [x["trajectory"] for x in rs if x.get("trajectory")]
        if traj:
            row["nll"] = {f"{side}_{when}": sum(t[i][side]["nll"] for t in traj) / len(traj)
                          for side in ("forget", "retain", "real_authors", "world_facts")
                          for when, i in (("warm_start", 0), ("after", -1))}
        ws = groups.get((targets, s_min, m, rule, 0.0))
        report = rs[0]["warm_start"]["per_target"]
        if ws:
            w = ws[max(ws)]
            row["warm_start_scores"] = {c: w["scores"][c] for c in COLUMNS}
            row["warm_start_run"] = os.path.basename(w["args"]["out_dir"])
            report = w["warm_start"]["per_target"]  # it reports ||P(W)||; the warm start is the same for every seed
        row["budget"] = budget_of(report)
        rows.append(row)

    ref = next((r for r in rows if tuple(r["key"]) == REFERENCE), None)
    for r in rows:
        if ref is None or r is ref or r["n"] < 2 or ref["n"] < 2:
            continue
        r["welch_vs_reference"] = {}
        for c in COLUMNS:
            t, pval = scipy_stats.ttest_ind(r["values"][c], ref["values"][c], equal_var=False)
            r["welch_vs_reference"][c] = {"difference": r["after"][c]["mean"] - ref["after"][c]["mean"],
                                          "t": float(t), "p": float(pval)}

    # Budgets of per-type floors from the energies of the uniform warm start of the same targets and rank.
    uniform_e = {(r["targets"], r["rank_multiplier"]): r["budget"]["E"] for r in rows
                 if not isinstance(r["s_min"], dict) and r["shrink_score"] == "original"
                 and not any(math.isnan(v) for v in r["budget"]["E"].values())}
    for r in rows:
        e = uniform_e.get((r["targets"], r["rank_multiplier"]))
        if e is None:
            continue
        floors = r["s_min"] if isinstance(r["s_min"], dict) else {k: r["s_min"] for k in e}
        b = update_budget(floors, e)
        r["budget"]["B_from_energies"] = b
        r["budget"]["uniform_floor_same_budget"] = uniform_floor_for_budget(b, e)
        r["budget"]["t_equivalent"] = {k: (1.0 - v) / 0.8 for k, v in floors.items()}  # against the reference floor 0.2

    # Per-type floors against the uniform floor of the same budget (rounded to the 4 decimals the runs used).
    uniform_rows = {(r["targets"], r["rank_multiplier"], round(r["s_min"], 4)): r for r in rows
                    if not isinstance(r["s_min"], dict) and r["shrink_score"] == "original"}
    for r in rows:
        u = r["budget"].get("uniform_floor_same_budget")
        twin = uniform_rows.get((r["targets"], r["rank_multiplier"], round(u, 4))) if isinstance(r["s_min"], dict) and u is not None else None
        if twin is None or r["n"] < 2 or twin["n"] < 2:
            continue
        r["uniform_same_budget"] = {"floor": twin["s_min"], "n": twin["n"], "after": twin["after"], "welch": {}}
        for c in COLUMNS:
            t, pval = scipy_stats.ttest_ind(r["values"][c], twin["values"][c], equal_var=False)
            r["uniform_same_budget"]["welch"][c] = {"difference": r["after"][c]["mean"] - twin["after"][c]["mean"],
                                                    "t": float(t), "p": float(pval)}

    # Matched-displacement floors: for each uniform row as the reference budget, the uniform floor of every other
    # (targets, rank) whose whole update has the same norm (None: it would need a floor below 0).
    matched = []
    for base in rows:
        if isinstance(base["s_min"], dict) or base["shrink_score"] != "original":
            continue
        for (targets, m), e in uniform_e.items():
            if (targets, m) != (base["targets"], base["rank_multiplier"]):
                matched.append({"reference": f"{base['targets']} m {base['rank_multiplier']} floor {base['s_min']}",
                                "budget": base["budget"]["B"], "targets": targets, "rank_multiplier": m,
                                "floor": uniform_floor_for_budget(base["budget"]["B"], e),
                                "max_budget": math.sqrt(sum(e.values()))})
    with open(args.out, "w") as fh:
        json.dump({"reference": list(REFERENCE), "rows": rows, "matched_floors": matched}, fh, indent=1)

    for r in rows:
        a = r["after"]
        floors = r["s_min"] if not isinstance(r["s_min"], dict) else ",".join(f"{k[0]}{v:g}" for k, v in r["s_min"].items())
        ws = r.get("warm_start_scores")
        nll = r.get("nll", {})
        b = r["budget"]
        w = r.get("welch_vs_reference", {}).get("Overall")
        print(f"{r['targets']:12s} m {r['rank_multiplier']} floor {floors} rule {r['shrink_score']} n {r['n']}: "
              + " / ".join(f"{a[c]['mean']:.4f}" for c in COLUMNS) + f" +- {a['Overall']['std']:.4f}"
              + (f" (vs reference {w['difference']:+.4f}, p {w['p']:.3f})" if w else "")
              + (f" | warm start " + " / ".join(f"{ws[c]:.3f}" for c in COLUMNS) if ws else "")
              + (f" | NLL forget {nll['forget_warm_start']:.2f} -> {nll['forget_after']:.2f}, retain "
                 f"{nll['retain_warm_start']:.2f} -> {nll['retain_after']:.3f}" if nll else "")
              + f" | B {b['B']:.2f}" + (f" (uniform floor of this budget {b['uniform_floor_same_budget']:.4f})"
                                         if isinstance(r["s_min"], dict) and b.get("uniform_floor_same_budget") is not None
                                         else "")
              + " | rel change " + ", ".join(f"{k} {v:.3f}" for k, v in b["rel_change"].items())
              + (f" | vs uniform floor {r['uniform_same_budget']['floor']} (Overall "
                 f"{r['uniform_same_budget']['after']['Overall']['mean']:.4f}): "
                 f"{r['uniform_same_budget']['welch']['Overall']['difference']:+.4f}, p "
                 f"{r['uniform_same_budget']['welch']['Overall']['p']:.3f}" if "uniform_same_budget" in r else ""))
    for x in matched:
        print(f"[matched] budget of {x['reference']} (B {x['budget']:.2f}): {x['targets']} m {x['rank_multiplier']} floor "
              f"{'infeasible' if x['floor'] is None else format(x['floor'], '.4f')} (B at floor 0: {x['max_budget']:.2f})")
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
