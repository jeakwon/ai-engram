"""Signed-shrink experiment report: every component mode right after the warm start and after phase two.

Reads ``results.json`` of ReCAST runs (recast_tofu.py, with or without ``--component_selection``), groups them
by component mode (none = the original update) and number of epochs (0 = the warm start alone), and writes a
Markdown table with mean and sample standard deviation over seeds of Overall, Memorization, Utility, Privacy,
EM, FQ and the real-authors ROUGE (a Model Utility sub-metric), next to the update statistics (kept components,
energy fraction, Frobenius norm, effective rank) and the first-order check (actual against predicted forget /
retain loss change of the warm start, from runs with ``--component_diagnostics yes``). It then gives the numbers
behind six questions: the share of update energy with c_j < 0, its distribution over module types, the forget
change of the signed update, the first-order accuracy, signed against uniform shrink at equal norm, and whether
the differences persist after phase two. Interpretation is left to the reader. Standard library only.

Positive control: the runs of a report share every argument except the component mode, epochs, seed and
bookkeeping fields, or the script stops.

Usage (compute node):
    python replication/scripts/component_report.py --results <results.json> [...] --out_md <file> --out_json <file>
"""

import argparse
import json
import os
import statistics

COLUMNS = ["Overall", "Memorization", "Utility", "Privacy", "EM", "FQ"]
FREE = {"component_selection", "component_diagnostics", "epochs", "seed", "out_dir", "run_name", "wandb_tags",
        "wandb_mode", "wandb_project", "anchors_raw", "eval_trajectory"}
MODES = ["none", "full", "signed", "random", "uniform_matched"]


def stat(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values)}


def fmt(s, digits=4):
    return "-" if s is None else f"{s['mean']:.{digits}f} ± {s['std']:.{digits}f}"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--out_md", required=True)
    p.add_argument("--out_json", required=True)
    args = p.parse_args()

    runs, shared = [], None
    for path in args.results:
        with open(path) as f:
            run = json.load(f)
        if "scores" not in run:
            print(f"[skip] {path}: no scores")
            continue
        a = run["args"]
        key = {k: v for k, v in a.items() if k not in FREE and v is not None}
        if shared is None:
            shared = key
        elif key != shared:
            diff = sorted(k for k in set(key) | set(shared) if key.get(k) != shared.get(k))
            raise SystemExit(f"{path} differs from the first run in {diff}")
        raw_path = os.path.join(os.path.dirname(path), "raw_metrics.json")
        ra_rouge = None
        if os.path.exists(raw_path):
            with open(raw_path) as f:
                raw = next(iter(json.load(f).values()))
            ra_rouge = raw.get("_model_utility_subs", {}).get("ra_Q_A_ROUGE")
        runs.append({"mode": a.get("component_selection") or "none", "epochs": int(a["epochs"]), "seed": a["seed"],
                     "job": run["runtime"].get("slurm_job_id"), "scores": run["scores"], "ra_rouge": ra_rouge,
                     "components": run.get("components")})
    print(f"[control] {len(runs)} runs share every argument except mode, epochs, seed and bookkeeping")

    groups = {}
    for r in runs:
        groups.setdefault((r["mode"], r["epochs"]), []).append(r)
    rows = {}
    for (mode, epochs), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], MODES.index(kv[0][0]))):
        row = {"mode": mode, "epochs": epochs, "seeds": sorted(r["seed"] for r in rs),
               "jobs": [r["job"] for r in rs], **{c: stat([r["scores"][c] for r in rs]) for c in COLUMNS},
               "ra_rouge": stat([r["ra_rouge"] for r in rs])}
        comp = [r["components"]["per_projection"]["all"] for r in rs if r["components"]]
        if comp:
            row.update({k: stat([c[k] for c in comp]) for k in (
                "kept", "components", "applied_energy_fraction", "applied_norm", "original_norm",
                "effective_rank_applied_mean", "energy_negative_fraction", "predicted_forget_change",
                "predicted_retain_change")})
        checks = [r["components"]["first_order_check"] for r in rs if r["components"]
                  and "first_order_check" in r["components"]]
        for side in ("forget", "retain"):
            row[f"actual_d{side[0]}"] = stat([c[side]["actual_change"] for c in checks])
            row[f"rel_error_d{side[0]}"] = stat([c[side]["rel_error"] for c in checks])
        rows[f"{mode}@{epochs}"] = row

    stage = {0: "warm start only", 10: "after phase two"}
    lines = ["# Signed shrink: component modes of the D update", "",
             "Mean ± sample std over seeds. Overall, Memorization, Utility, Privacy, EM, FQ as Table 3; ra ROUGE is "
             "the real-authors ROUGE sub-metric of Model Utility.", "",
             "| stage | mode | n | Overall | Mem | Util | Priv | EM | FQ | ra ROUGE |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows.values():
        lines.append(f"| {stage.get(r['epochs'], str(r['epochs']) + ' epochs')} | {r['mode']} | {r['Overall']['n']} | "
                     + " | ".join(fmt(r[c], 2 if c == "FQ" else 4) for c in COLUMNS) + f" | {fmt(r['ra_rouge'])} |")
    lines += ["", "| mode | kept / components | applied energy | ||ΔW|| applied / original | effective rank | "
                  "predicted dL_f | actual dL_f | rel. error dL_f | predicted dL_r | actual dL_r |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows.values():
        if r["epochs"] != 0 or "kept" not in r:
            continue
        lines.append(f"| {r['mode']} | {fmt(r['kept'], 0)} / {fmt(r['components'], 0)} | "
                     f"{fmt(r['applied_energy_fraction'])} | {fmt(r['applied_norm'], 3)} / {fmt(r['original_norm'], 3)} | "
                     f"{fmt(r['effective_rank_applied_mean'], 1)} | {fmt(r['predicted_forget_change'], 4)} | "
                     f"{fmt(r['actual_df'], 4)} | {fmt(r['rel_error_df'], 3)} | {fmt(r['predicted_retain_change'], 4)} | "
                     f"{fmt(r['actual_dr'], 4)} |")

    # Numbers behind the six questions.
    ref = next((r for r in runs if r["components"] and r["mode"] in ("none", "full")), None)
    answers = {}
    if ref:
        pp = ref["components"]["per_projection"]
        answers["q1_energy_negative_fraction"] = pp["all"]["energy_negative_fraction"]
        answers["q2_by_module"] = {k: {"energy_negative_fraction": v["energy_negative_fraction"],
                                       "negative": v["negative"], "components": v["components"]}
                                   for k, v in pp.items() if k != "all"}
    pick = lambda m, e, c: rows.get(f"{m}@{e}", {}).get(c)  # noqa: E731
    answers["q3_actual_forget_change"] = {m: pick(m, 0, "actual_df") for m in MODES}
    answers["q3_warm_start_scores"] = {m: {c: pick(m, 0, c) for c in COLUMNS} for m in MODES}
    answers["q4_first_order_rel_error"] = {m: {"forget": pick(m, 0, "rel_error_df"), "retain": pick(m, 0, "rel_error_dr")}
                                           for m in MODES}
    answers["q5_signed_vs_uniform_matched"] = {e: {m: {c: pick(m, e, c) for c in COLUMNS}
                                                   for m in ("signed", "uniform_matched", "random")} for e in (0, 10)}
    answers["q6_after_phase_two"] = {m: {c: pick(m, 10, c) for c in COLUMNS} for m in MODES}
    lines += ["", "## Numbers behind the diagnostic questions", ""]
    if "q1_energy_negative_fraction" in answers:
        lines.append(f"1. Update energy with c_j < 0 (original D): {answers['q1_energy_negative_fraction']:.4f}.")
        lines.append("2. By module type: " + "; ".join(
            f"{k} {v['energy_negative_fraction']:.4f} ({v['negative']} of {v['components']} components)"
            for k, v in answers["q2_by_module"].items()) + ".")
    lines.append("3. Actual forget-loss change of the warm start: " + "; ".join(
        f"{m} {fmt(v)}" for m, v in answers["q3_actual_forget_change"].items() if v) + ".")
    lines.append("4. First-order relative error (forget): " + "; ".join(
        f"{m} {fmt(v['forget'], 3)}" for m, v in answers["q4_first_order_rel_error"].items() if v["forget"]) + ".")
    for e in (0, 10):
        lines.append(f"{5 if e == 0 else 6}. {stage[e]}, Overall signed / uniform_matched / random: " + " / ".join(
            fmt(answers["q5_signed_vs_uniform_matched"][e][m]["Overall"]) for m in ("signed", "uniform_matched", "random"))
            + ".")
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(args.out_json, "w") as f:
        json.dump({"shared_args": shared, "rows": rows, "answers": answers}, f, indent=2)
    print("\n".join(lines))
    print(f"[done] {args.out_md} {args.out_json}")


if __name__ == "__main__":
    main()
