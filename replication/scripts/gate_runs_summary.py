"""Scores and phase-two trajectories of finished recast_tofu.py runs (direction gates and any reference run).

Per run directory: the label (variant; for direction_gates basis, alpha, gamma, optimizer and lr, the gate epochs run
and the stop reason), Mem / Util / Priv / EM / FQ / Overall, and per trajectory point (epoch 0 = after phase one,
before phase two) the forget NLL and exact memorization and the retain, real-authors and world-facts NLL of
``--eval_trajectory nll``. Standard library only.

Usage: python3 gate_runs_summary.py <run dir> ... [--out_json <file>]
"""

import argparse
import json
import os

COLUMNS = ("Memorization", "Utility", "Privacy", "EM", "FQ", "Overall")


def summary(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "results.json")) as f:
        r = json.load(f)
    a = r["args"]
    label = a["variant"]
    gates = r.get("direction_gates")
    if a["variant"] == "direction_gates":
        label += (f" {a['direction_gates_basis']} alpha {a['direction_gates_alpha']:g} gamma "
                  f"{a['direction_gates_gamma']:g} {a['direction_gates_optim']} lr {a['direction_gates_lr']:g}")
        if a.get("direction_gates_lambda_star") is not None:
            label += f" lambda* {a['direction_gates_lambda_star']:g}"
        if a.get("direction_gates_init") not in (None, 1.0):
            label += f" init {a['direction_gates_init']:g}"
        if a.get("direction_gates_binarize_share") is not None:
            label += f" binarized share {a['direction_gates_binarize_share']:g} s {a['direction_gates_binary_s']:g}"
        elif a.get("direction_gates_cut") is not None:
            label += " every candidate"
        if a.get("direction_gates_cut") == "shrink" and a.get("direction_gates_cut_s") is not None:
            label += f" cut to {a['direction_gates_cut_s']:g}"
        elif a.get("direction_gates_cut") == "rotate":
            label += f" rotated (forget subspace share {a['direction_gates_rotate_evr']:g}"
            label += (")" if a.get("direction_gates_cut_s") is None else
                      f", keeping {a['direction_gates_cut_s']:g} along the own direction)")
        if a.get("direction_gates_convergence_epochs") == 0 and a.get("direction_gates_max_epochs"):
            label += (f" fixed {a['direction_gates_max_epochs']} epochs lr {a['direction_gates_lr_schedule']}, probe "
                      f"every {a['direction_gates_probe_every']}")
    elif a.get("global_family"):
        label += (f" global {a['global_family']} order {a.get('global_family_order') or 'score'} floor {a['s_min']}"
                  f" s {a['global_family_floor']}")
    traj = [{"epoch": p["epoch"], "forget_nll": p["forget"]["nll"], "forget_em": p["forget"]["em"],
             "retain_nll": p["retain"]["nll"], "real_authors_nll": p["real_authors"]["nll"],
             "world_facts_nll": p["world_facts"]["nll"]} for p in r.get("trajectory", [])]
    return {"run": os.path.basename(run_dir.rstrip("/")), "label": label, "seed": a["seed"],
            "job": r["runtime"].get("slurm_job_id"), "scores": {c: r["scores"][c] for c in COLUMNS},
            "gate_epochs": gates["epochs_run"] if gates else None, "gate_stop": gates["stop"] if gates else None,
            "trajectory": traj}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+")
    p.add_argument("--out_json")
    args = p.parse_args()
    out = [summary(d) for d in args.runs]
    for s in out:
        sc = s["scores"]
        extra = "" if s["gate_epochs"] is None else f", gate epochs {s['gate_epochs']} ({s['gate_stop']})"
        print(f"== {s['label']} (seed {s['seed']}, job {s['job']}{extra})")
        print("   Mem / Util / Priv / EM / FQ / Overall: " + " / ".join(f"{sc[c]:.4f}" for c in COLUMNS[:4])
              + f" / {sc['FQ']:.2f} / {sc['Overall']:.4f}")
        for t in s["trajectory"]:
            print(f"   epoch {t['epoch']}: forget NLL {t['forget_nll']:.3f} EM {t['forget_em']:.3f} | retain NLL "
                  f"{t['retain_nll']:.3f} | real authors {t['real_authors_nll']:.3f} | world facts "
                  f"{t['world_facts_nll']:.3f}")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
