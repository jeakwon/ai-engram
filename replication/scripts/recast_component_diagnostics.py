"""Weight-core components of the down modules: how kappa, input contrast and gradient alignment relate, and what each
ordering of the contraction would move at the same budget.

Reads the warm-start-only runs with --contraction_diagnostics (recast_tofu.py; per target and component j: kappa_j,
the input contrast of v_j and a_j = u_j^T G v_j for the forget and retain gradients). Per run (one EVR):
- the mean rank, ||P(W)|| over all targets and the displacement (1 - floor) ||P(W)|| at floors 0.1 and 0.2;
- Spearman correlations over all components between kappa, the input contrast and the alignments;
- the mean input contrast by kappa decile within each target (decile 1 = the largest kappa);
- for each ordering, the contraction at beta 1 with the budget ``--budget`` per target (as recast_contraction, the last
  component partly): components moved, mean normalised kappa rank, the input contrast of the moved energy and the
  first-order loss changes sum_j -beta_j kappa_j a_j for forget and retain (Stage 2 found first-order predictions of
  whole warm starts far off in size, so they rank orderings rather than predict losses). Orderings: top (kappa
  descending), random (mean of 20 draws), contrast (input contrast descending), forget_gain (-a_forget / kappa
  descending: the first-order forget-loss increase per unit of moved energy) and forget_minus_retain
  (-(a_forget - a_retain) / kappa descending).

Usage (Slurm, ai-engram-gh200 environment): python recast_component_diagnostics.py --root <recast_tofu results> --out <json>
"""

import argparse
import glob
import json
import math
import os
import random

from scipy import stats as scipy_stats


def greedy(kappa, order, budget):
    """beta per component: 1 in ``order`` until sum beta^2 kappa^2 = budget^2 sum kappa^2, the last one partly."""
    k2 = [k * k for k in kappa]
    rest = budget ** 2 * sum(k2)
    beta = [0.0] * len(kappa)
    for j in order:
        if k2[j] <= 0.0:
            continue
        if k2[j] < rest:
            beta[j], rest = 1.0, rest - k2[j]
        else:
            beta[j] = math.sqrt(rest / k2[j])
            break
    return beta


def selection_stats(targets, orders, budget):
    moved_total, contrast_moved, rank_sum, rank_n, count, d_forget, d_retain = 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0
    for t, order in zip(targets, orders):
        kappa, c, af, ar = t["kappa"], t["input_contrast"], t["alignment"]["forget"], t["alignment"]["retain"]
        beta = greedy(kappa, order, budget)
        n = len(kappa)
        for j, b in enumerate(beta):
            if b <= 0.0:
                continue
            moved = b * b * kappa[j] * kappa[j]
            moved_total += moved
            contrast_moved += c[j] * moved
            rank_sum += j / max(n - 1, 1)   # components come in kappa order
            rank_n += 1
            count += 1
            d_forget += -b * kappa[j] * af[j]
            d_retain += -b * kappa[j] * ar[j]
    return {"components": count, "kappa_rank_mean": rank_sum / rank_n if rank_n else None,
            "input_contrast_moved": contrast_moved / moved_total if moved_total else None,
            "first_order_forget": d_forget, "first_order_retain": d_retain}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--budget", type=float, default=0.9, help="displacement per target as a share of ||P(W)|| (1 - floor)")
    args = p.parse_args()

    runs = []
    for f in sorted(glob.glob(os.path.join(args.root, "*", "results.json"))):
        with open(f) as fh:
            r = json.load(fh)
        a = r.get("args", {})
        if a.get("contraction_diagnostics") != "yes" or float(a.get("epochs", -1)) != 0.0:
            continue
        per = r["warm_start"]["per_target"]
        targets = [dict(v["contraction_components"], name=k) for k, v in per.items() if "contraction_components" in v]
        pw = math.sqrt(sum(v.get("projected_weight_norm", 0.0) ** 2 for v in per.values()))
        row = {"run": os.path.basename(a["out_dir"]), "targets": a["targets"], "evr": a["restriction_evr"],
               "rank_multiplier": a.get("restriction_rank_multiplier"),
               "rank_mean": sum(v["rank"] for v in per.values()) / len(per), "projected_norm": pw,
               "displacement": {"floor 0.1": 0.9 * pw, "floor 0.2": 0.8 * pw},
               "components": sum(len(t["kappa"]) for t in targets)}
        flat = {k: [x for t in targets for x in (t[k] if k in ("kappa", "input_contrast") else t["alignment"][k])]
                for k in ("kappa", "input_contrast", "forget", "retain")}
        pairs = [("kappa", "input_contrast"), ("kappa", "forget"), ("kappa", "retain"), ("input_contrast", "forget"),
                 ("forget", "retain")]
        row["spearman"] = {f"{x} ~ {y}": float(scipy_stats.spearmanr(flat[x], flat[y])[0]) for x, y in pairs}
        row["spearman"]["kappa ~ |forget|"] = float(scipy_stats.spearmanr(flat["kappa"], [abs(v) for v in flat["forget"]])[0])
        deciles = [[] for _ in range(10)]
        for t in targets:
            n = len(t["kappa"])
            for j, c in enumerate(t["input_contrast"]):
                deciles[min(9, j * 10 // max(n, 1))].append(c)
        row["contrast_by_kappa_decile"] = [sum(d) / len(d) if d else None for d in deciles]
        row["contrast_quantiles"] = {q: float(sorted(flat["input_contrast"])[int(q * (len(flat["input_contrast"]) - 1))])
                                     for q in (0.01, 0.1, 0.5, 0.9, 0.99)}

        def by(score):
            return [sorted(range(len(t["kappa"])), key=lambda j: score(t, j), reverse=True) for t in targets]

        sel = {"top": selection_stats(targets, by(lambda t, j: t["kappa"][j]), args.budget),
               "contrast": selection_stats(targets, by(lambda t, j: t["input_contrast"][j]), args.budget),
               "forget_gain": selection_stats(
                   targets, by(lambda t, j: -t["alignment"]["forget"][j] / max(t["kappa"][j], 1e-300)), args.budget),
               "forget_minus_retain": selection_stats(
                   targets, by(lambda t, j: -(t["alignment"]["forget"][j] - t["alignment"]["retain"][j])
                               / max(t["kappa"][j], 1e-300)), args.budget),
               "all_uniform": None}
        rng = random.Random(0)
        draws = []
        for _ in range(20):
            orders = []
            for t in targets:
                o = list(range(len(t["kappa"])))
                rng.shuffle(o)
                orders.append(o)
            draws.append(selection_stats(targets, orders, args.budget))
        sel["random"] = {k: (sum(d[k] for d in draws) / len(draws) if all(d[k] is not None for d in draws) else None)
                         for k in draws[0]}
        # ReCAST (beta = budget on every component), the same displacement
        dfu = sum(-args.budget * k * a for t in targets for k, a in zip(t["kappa"], t["alignment"]["forget"]))
        dru = sum(-args.budget * k * a for t in targets for k, a in zip(t["kappa"], t["alignment"]["retain"]))
        e = sum(k * k for t in targets for k in t["kappa"])
        cu = sum(c * k * k for t in targets for k, c in zip(t["kappa"], t["input_contrast"])) / e if e else None
        sel["all_uniform"] = {"components": row["components"], "kappa_rank_mean": 0.5, "input_contrast_moved": cu,
                              "first_order_forget": dfu, "first_order_retain": dru}
        row["selections"] = sel
        runs.append(row)

    runs.sort(key=lambda r: r["evr"])
    with open(args.out, "w") as fh:
        json.dump({"budget": args.budget, "runs": runs}, fh, indent=1)
    for r in runs:
        print(f"[evr {r['evr']}] rank mean {r['rank_mean']:.1f}, ||P(W)|| {r['projected_norm']:.2f}, displacement at floor "
              f"0.1 {r['displacement']['floor 0.1']:.2f}, at 0.2 {r['displacement']['floor 0.2']:.2f}, "
              f"{r['components']} components")
        print("    spearman " + ", ".join(f"{k} {v:+.3f}" for k, v in r["spearman"].items()))
        print("    contrast quantiles " + ", ".join(f"q{q} {v:.3f}" for q, v in r["contrast_quantiles"].items()))
        print("    contrast by kappa decile " + " ".join(f"{v:.3f}" for v in r["contrast_by_kappa_decile"] if v is not None))
        for name, s in r["selections"].items():
            print(f"    {name:20s} components {s['components']:7.1f}, kappa rank {s['kappa_rank_mean']:.3f}, contrast moved "
                  f"{s['input_contrast_moved']:.4f}, first-order forget {s['first_order_forget']:+.4e}, retain "
                  f"{s['first_order_retain']:+.4e}")
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
