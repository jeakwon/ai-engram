"""Summary of direction-gate learning-rate calibrations (recast_tofu.py --direction_gates_calibration_trials).

For every calibration directory given: the basis, the census guides (results.json) and per trial, from
direction_gates_calibration_history.json, the probe at the start and at the end (J, L_R = retain NLL, L_F, forget
NLL, R = ||dW||^2 / sum ||W0||^2), the changes of the forget and retain NLL and their ratio, the largest probe retain
NLL over the epochs and its epoch (the retain loss first rising, then falling once L_F saturates), the gate mean, the
gates below 0.5 and at 0, the drift and the mean reversal share over the trial, the largest step, and the seconds per
epoch. Standard library only.

Usage: python3 gate_calibration_summary.py <calibration dir> ... [--out_json <file>]
"""

import argparse
import json
import os


def trial_rows(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "results.json")) as f:
        res = json.load(f)
    cal = res["direction_gates_calibration"]
    with open(os.path.join(run_dir, cal["history_file"])) as f:
        hist = json.load(f)
    rows = []
    for summary, rec in zip(cal["trials"], hist["trials"]):
        ep = rec["epochs"]
        first, last = ep[0], ep[-1]
        retain = [e["probe_retain_nll"] for e in ep]
        peak = max(range(len(retain)), key=lambda i: retain[i])
        d_forget = last["probe_forget_nll"] - first["probe_forget_nll"]
        d_retain = last["probe_retain_nll"] - first["probe_retain_nll"]
        rows.append({
            "optim": rec["optim"], "lr": rec["lr"], "epochs": len(ep) - 1,
            "j_start": first["probe_objective"], "j_end": last["probe_objective"],
            "retain_nll_start": first["probe_retain_nll"], "retain_nll_end": last["probe_retain_nll"],
            "forget_nll_start": first["probe_forget_nll"], "forget_nll_end": last["probe_forget_nll"],
            "forget_unlikelihood_end": last["probe_forget_unlikelihood"], "regulariser_end": last["probe_regulariser"],
            "d_forget_nll": d_forget, "d_retain_nll": d_retain,
            "forget_over_retain": d_forget / d_retain if d_retain > 0 else None,
            "retain_nll_peak": retain[peak], "retain_nll_peak_epoch": peak,
            "gate_mean": last["gate_mean"], "gates_below_0p5": last["gates_below_0p5"], "gates_at_0": last["gates_at_0"],
            "drift": summary["drift"], "reversal_share_mean": summary["reversal_share_mean"],
            "update_max": summary["update_max"], "seconds_per_epoch": summary["seconds"] / max(len(ep) - 1, 1)})
    return {"run": os.path.basename(run_dir.rstrip("/")), "basis": cal["config"]["direction_gates_basis"],
            "alpha": cal["config"]["direction_gates_alpha"], "gamma": cal["config"]["direction_gates_gamma"],
            "census": cal["census"], "trials": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+")
    p.add_argument("--out_json")
    args = p.parse_args()
    out = [trial_rows(d) for d in args.runs]
    for r in out:
        print(f"== {r['basis']} (alpha {r['alpha']}, gamma {r['gamma']}): {r['run']}")
        print("   optim lr      | J start->end    | retain NLL start->end (peak@ep) | forget NLL start->end | dF/dR  "
              "| R end    | gate mean below0.5 at0 | drift rev  max step | s/ep")
        for t in r["trials"]:
            ratio = "-" if t["forget_over_retain"] is None else f"{t['forget_over_retain']:.1f}"
            print(f"   {t['optim']:<5} {t['lr']:<7g} | {t['j_start']:.3f}->{t['j_end']:.3f} | "
                  f"{t['retain_nll_start']:.3f}->{t['retain_nll_end']:.3f} ({t['retain_nll_peak']:.3f}@{t['retain_nll_peak_epoch']}) | "
                  f"{t['forget_nll_start']:.3f}->{t['forget_nll_end']:.3f} | {ratio:>6} | {t['regulariser_end']:.2e} | "
                  f"{t['gate_mean']:.4f} {t['gates_below_0p5']:>5} {t['gates_at_0']:>4} | {t['drift']:.3f} "
                  f"{t['reversal_share_mean']:.3f} {t['update_max']:.2e} | {t['seconds_per_epoch']:.1f}")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
