"""Table 6 of arXiv 2606.14997 for our runs: every sub-metric behind Memorization, Utility and Privacy.

``engram.benchmarks._tofu_official.evaluate_scores`` returns only the aggregates. This script applies
its formulas step by step to raw metrics and keeps the intermediate values: EM, ES, PP and TR rescaled
against the anchors, MU and FF divided by Init Fine., the four MIA indistinguishability scores, FQ.
Positive control: the aggregates it rebuilds must equal ``evaluate_scores`` to 1e-12, or it stops.

Usage (GH200 node, ai-engram-gh200 environment; numpy, scipy and scikit-learn):
    python replication/scripts/table6_breakdown.py --anchors_raw <table3 raw_metrics.json> \
        --run "label=<raw_metrics.json>" [--run ...] --out_json <file> --out_md <file>
"""

import argparse
import json

import numpy as np
import scipy.stats as st
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from engram.benchmarks import _tofu_official as T

ATTACKS = ["loss", "zlib", "min_k", "min_k++"]
HEADER = ["Overall", "Mem", "EM", "ES", "PP", "TR", "Util", "MU", "FF", "Priv",
          "S_loss", "S_zlib", "S_min-k", "S_min-k++", "FQ"]


def absr(x, mn, mx):  # as evaluate_scores
    return float(np.clip(abs(x - mn) / abs(mx - mn + 1e-12), 0, 1))


def divr(x, mx):
    return float(np.clip(x / mx, 0, 1))


def hm(values):
    return float(st.hmean([max(v, 1e-10) for v in values]))


def indist(rv, u):
    y = np.r_[np.zeros(len(rv)), np.ones(len(u))]
    try:
        auc = roc_auc_score(y, np.r_[rv, u])
    except ValueError:
        return 1.0
    return float(1 - 2 * abs(auc - 0.5))


def breakdown(exp, retain, ft):
    em = absr(exp["exact_memorization"], retain["exact_memorization"], ft["exact_memorization"])
    es = absr(exp["extraction_strength"], retain["extraction_strength"], ft["extraction_strength"])
    pp = absr(exp["forget_Q_A_Prob"], retain["forget_Q_A_Prob"], ft["forget_Q_A_Prob"])
    tr = absr(exp["forget_truth_ratio"], retain["forget_truth_ratio"], ft["forget_truth_ratio"])
    mu = divr(exp["model_utility"], ft["model_utility"])
    ff = divr(exp["forget_Q_A_gibberish"], ft["forget_Q_A_gibberish"])
    s = {a: indist(retain["_mia_fs"][a], exp["_mia_fs"][a]) for a in ATTACKS}
    row = {"Mem": hm([1 - em, 1 - es, 1 - pp, 1 - tr]), "EM": em, "ES": es, "PP": pp, "TR": tr,
           "Util": hm([mu, ff]), "MU": mu, "FF": ff, "Priv": hm([s[a] for a in ATTACKS]),
           "S_loss": s["loss"], "S_zlib": s["zlib"], "S_min-k": s["min_k"], "S_min-k++": s["min_k++"],
           "FQ": float(np.log10(max(ks_2samp(exp["_forget_tr"], retain["_forget_tr"]).pvalue, 1e-30)))}
    row["Overall"] = hm([row["Mem"], row["Util"], row["Priv"]])
    # Positive control: identical to the packaged aggregation.
    ref = T.evaluate_scores(exp, retain, ft)
    for ours, theirs in (("Overall", "Overall"), ("Mem", "Memorization"), ("Util", "Utility"),
                         ("Priv", "Privacy"), ("EM", "EM"), ("FQ", "FQ")):
        if abs(row[ours] - ref[theirs]) > 1e-12:
            raise SystemExit(f"breakdown {ours}={row[ours]} differs from evaluate_scores {theirs}={ref[theirs]}")
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--anchors_raw", required=True, help="raw_metrics.json with 'Init Fine.' and 'Retain'")
    p.add_argument("--anchor_rows", nargs="*", default=[], help="further rows of the anchors file to include")
    p.add_argument("--run", action="append", default=[], help="label=path to a raw_metrics.json with one row")
    p.add_argument("--out_json", required=True)
    p.add_argument("--out_md", required=True)
    args = p.parse_args()

    with open(args.anchors_raw) as f:
        anchors = json.load(f)
    ft, rt = anchors["Init Fine."], anchors["Retain"]
    rows = {name: breakdown(anchors[name], rt, ft) for name in ["Init Fine.", "Retain", *args.anchor_rows]}
    for spec in args.run:
        label, _, path = spec.partition("=")
        with open(path) as f:
            raw = json.load(f)
        rows[label] = breakdown(next(iter(raw.values())), rt, ft)
    print("[control] every row's aggregates equal evaluate_scores to 1e-12")

    lines = ["| Method | " + " | ".join(HEADER) + " |", "|" + "---|" * (len(HEADER) + 1)]
    for name, r in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{r[h]:.4f}" for h in HEADER) + " |")
    table = "\n".join(lines)
    print(table)
    with open(args.out_json, "w") as f:
        json.dump({"anchors_raw": args.anchors_raw, "runs": args.run, "rows": rows}, f, indent=2)
    with open(args.out_md, "w") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()
