"""Training-free proxies of ReCAST configurations, checked against the trained runs.

For every listed configuration the script applies only the warm start (phase one) at Init Fine.,
measures two kinds of proxies and restores the weights:

- layer proxies, per target: the change of the layer outputs on forget and retain answer tokens,
  q = tr(D C D^T) / tr(W C W^T) with D = W* - W and C the input covariance of the set, as means over
  targets and as energy-weighted totals Q = sum tr(D C D^T) / sum tr(W C W^T), with the selectivity
  Q_f / Q_r;
- functional proxies, on the warm-started model itself: the increase of the mean answer-token NLL
  over Init Fine. on forget (dL_f), retain (dL_r), real authors (dL_ra) and world facts (dL_wf), the
  last two being the pre-TOFU knowledge behind Model Utility, and the ratios dL_f / dL_r (fsel_r) and
  dL_f / dL_world (fsel_w, dL_world the mean of dL_ra and dL_wf).

With --trained_summary (a collect_recast_tofu.py --out JSON) it joins each configuration with its
trained runs (phase two fixed: restricted_semu free_all, constant lr 1e-5, retain NLL, 10 epochs) and
reports Spearman correlations of the proxies with Memorization, Utility, Privacy, EM, FQ and Overall
across configurations.

A configuration is ``label:key=value,...`` with keys profile (a target profile inside
--collect_targets; default --targets), basis (plain | whiten | repr), rho, m (rank multiplier), qf
(forget-output target, recast_core ``forget_output_target``; without m or qf the rank is the EVR
rank), s_min, evr, out (repr output side), mix (repr input mix) and kappa (ridge filter A on the
plain basis).

Usage (GH200 node, ai-engram-gh200 environment, offline cache): see warmstart_proxy_gh200.sh.
"""

import argparse
import json
import math
import os
import platform

from recast_core import (collect_input_covariances, collect_mean_gradients, parse_blocks, ridge_input_filter,
                         select_targets, warm_start)
from table3_tofu import Stage, git_info, to_jsonable

PHASE_TWO = {"variant": "restricted_semu", "target_blocks": "all", "lr_points": "0:0,1:1e-5,10:1e-5",
             "phase2_free": "all", "retain_loss": "nll", "epochs": 10.0}
METRICS = ("Memorization", "Utility", "Privacy", "EM", "FQ", "Overall")
PROXIES = ("Q_f", "Q_r", "selectivity", "mean_q_f", "mean_q_r", "mean_selectivity", "weight_change",
           "dL_f", "dL_r", "dL_ra", "dL_wf", "fsel_r", "fsel_w")
NLL_SETS = {"f": "forget", "r": "retain", "ra": "real_authors_perturbed", "wf": "world_facts_perturbed"}


def parse_config(spec, default_profile):
    label, _, body = spec.partition(":")
    cfg = {"label": label, "profile": default_profile, "basis": "plain", "s_min": 0.3, "evr": 0.97}
    for item in filter(None, body.split(",")):
        k, _, v = item.partition("=")
        cfg[k] = v if k in ("basis", "out", "profile") else float(v)
    if cfg["basis"] not in ("plain", "whiten", "repr"):
        raise SystemExit(f"{label}: unknown basis {cfg['basis']}")
    if "m" in cfg and "qf" in cfg:
        raise SystemExit(f"{label}: m and qf are two budgets; give one")
    return cfg


def trained_args(cfg):
    """The trained-run arguments this configuration corresponds to (fields of the collector rows)."""
    return dict(PHASE_TWO, targets=cfg["profile"], s_min=cfg["s_min"], restriction_evr=cfg["evr"],
                restriction_rank_multiplier=cfg.get("m"), restriction_forget_output_target=cfg.get("qf"),
                retain_input_filter="ridge" if "kappa" in cfg else "none", retain_filter_kappa=cfg.get("kappa"),
                restriction_whitening="retain_input" if cfg["basis"] == "whiten" else "none",
                whitening_rho=cfg.get("rho"),
                restriction_representation=cfg.get("out", "none") if cfg["basis"] == "repr" else "none",
                representation_mix=cfg.get("mix"))


def same(x, y):
    if x is None or y is None:
        return x is None and y is None
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return abs(float(x) - float(y)) <= 1e-9 * max(1.0, abs(float(x)))
    return x == y


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for k in ("--base_id", "--dataset_id", "--forget_split", "--retain_split", "--targets", "--target_blocks",
              "--attn_implementation", "--out_json"):
        p.add_argument(k, required=True)
    p.add_argument("--collect_targets", help="profile whose targets get gradients and covariances; it must contain "
                                             "every configuration's profile (default --targets)")
    p.add_argument("--grad_batch_size", type=int, required=True)
    p.add_argument("--config", action="append", required=True)
    p.add_argument("--trained_summary", help="collect_recast_tofu.py --out JSON to correlate with")
    args = p.parse_args()

    import scipy.stats
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from engram.benchmarks import _tofu_official as T

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    timings = {}
    out = {"args": vars(args), "git": git_info(repo), "host": platform.node(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "timings_s": timings}
    configs = [parse_config(s, args.targets) for s in args.config]
    print(f"[run] {len(configs)} configurations")
    device = "cuda"
    data = {k: load_dataset(args.dataset_id, {"f": args.forget_split, "r": args.retain_split}.get(k, split))["train"]
            for k, split in NLL_SETS.items()}
    tok = AutoTokenizer.from_pretrained(args.base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = (AutoModelForCausalLM.from_pretrained(args.base_id, torch_dtype=torch.float32,
                                                  attn_implementation=args.attn_implementation).to(device))
    blocks = parse_blocks(args.target_blocks)
    targets = select_targets(model, args.collect_targets or args.targets, blocks=blocks)
    profiles = {cfg["profile"]: select_targets(model, cfg["profile"], blocks=blocks) for cfg in configs}
    for name, sub in profiles.items():
        if set(sub) - set(targets):
            raise SystemExit(f"profile {name} has targets outside --collect_targets")

    def feats(b):
        return {k: b[k].to(device) for k in ("input_ids", "attention_mask", "labels")}

    def loader(ds):
        return DataLoader(T.QADataset(ds, tok, akey="answer"), batch_size=args.grad_batch_size,
                          collate_fn=T.Collator(tok, "right"))

    @torch.no_grad()
    def answer_nll():
        """Mean NLL per answer token (labels != -100) of each set, the model in eval mode."""
        model.eval()
        nll = {}
        for k, ds in data.items():
            total, count = 0.0, 0
            for b in loader(ds):
                x = feats(b)
                logits = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).logits[:, :-1].float()
                labels = x["labels"][:, 1:]
                total += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                               ignore_index=-100, reduction="sum"))
                count += int((labels != -100).sum())
            nll[k] = total / count
        return nll

    with Stage("gradients and covariances", timings):
        g_r, _ = collect_mean_gradients(model, loader(data["r"]), targets, feats)
        g_f, _ = collect_mean_gradients(model, loader(data["f"]), targets, feats)
        c_r, n_r = collect_input_covariances(model, loader(data["r"]), targets, feats)
        c_f, n_f = collect_input_covariances(model, loader(data["f"]), targets, feats)
    print(f"[data] answer tokens retain {n_r}, forget {n_f}; targets {len(targets)}; "
          f"NLL sets {({k: len(v) for k, v in data.items()})}")
    original = {n: m.weight.detach().clone() for n, m in targets.items()}
    base_energy = {n: {"f": float(((original[n] @ c_f[n]) * original[n]).sum()),
                       "r": float(((original[n] @ c_r[n]) * original[n]).sum())} for n in targets}
    with Stage("Init Fine. NLL", timings):
        base_nll = answer_nll()
    print(f"[nll] Init Fine.: " + "  ".join(f"{k} {v:.4f}" for k, v in base_nll.items()))

    proxies = {}
    for cfg in configs:
        sub = profiles[cfg["profile"]]
        with Stage(f"warm start {cfg['label']}", timings):
            extra = {"rank_multiplier": cfg.get("m")}
            if cfg["basis"] == "whiten":
                extra.update(whitening=c_r, whitening_rho=cfg["rho"])
            elif cfg["basis"] == "repr":
                extra.update(representation_output=cfg["out"], retain_covariances=c_r, whitening_rho=cfg["rho"],
                             representation_mix=cfg.get("mix", 1.0))
            if cfg["basis"] == "repr" or "qf" in cfg:
                extra.update(forget_covariances=c_f, forget_output_target=cfg.get("qf"))
            if "kappa" in cfg:
                extra["input_filters"] = {n: ridge_input_filter(c_r[n], cfg["kappa"])[0] for n in sub}
            _, report = warm_start(sub, g_r, cfg["s_min"], g_f, cfg["evr"], **extra)  # in place
            rows, num_f, num_r, den_f, den_r = [], 0.0, 0.0, 0.0, 0.0
            for n in sub:
                d = sub[n].weight.detach() - original[n]
                ef, er = float(((d @ c_f[n]) * d).sum()), float(((d @ c_r[n]) * d).sum())
                num_f, num_r = num_f + ef, num_r + er
                den_f, den_r = den_f + base_energy[n]["f"], den_r + base_energy[n]["r"]
                rows.append({"q_f": ef / base_energy[n]["f"], "q_r": er / base_energy[n]["r"],
                             "weight_change": float(d.norm() / original[n].norm())})
            nll = answer_nll()
            with torch.no_grad():
                for n in sub:
                    sub[n].weight.copy_(original[n])
        k = len(rows)
        dl = {f"dL_{s}": nll[s] - base_nll[s] for s in NLL_SETS}
        world = (dl["dL_ra"] + dl["dL_wf"]) / 2
        proxies[cfg["label"]] = {
            "Q_f": num_f / den_f, "Q_r": num_r / den_r, "selectivity": (num_f / den_f) / (num_r / den_r),
            "mean_q_f": sum(r["q_f"] for r in rows) / k, "mean_q_r": sum(r["q_r"] for r in rows) / k,
            "mean_selectivity": sum(r["q_f"] / r["q_r"] for r in rows if r["q_r"] > 0) / k,
            "weight_change": sum(r["weight_change"] for r in rows) / k, **dl,
            "fsel_r": dl["dL_f"] / dl["dL_r"] if dl["dL_r"] != 0 else math.nan,
            "fsel_w": dl["dL_f"] / world if world != 0 else math.nan,
            "rank_sum": sum(r.get("rank", 0) for r in report.values()),
            "reference_rank_sum": sum(r.get("reference_rank", 0) for r in report.values()), "targets": k}
        print(f"[proxy] {cfg['label']}: " + "  ".join(f"{p_} {proxies[cfg['label']][p_]:.4f}" for p_ in PROXIES)
              + f"  rank_sum {proxies[cfg['label']]['rank_sum']}")
    # Positive control: every warm start was undone (the Init Fine. forget NLL comes back).
    end_nll = answer_nll()
    if any(abs(end_nll[s] - base_nll[s]) > 1e-4 * max(1.0, abs(base_nll[s])) for s in NLL_SETS):
        raise RuntimeError(f"weights not restored: NLL {end_nll} vs Init Fine. {base_nll}")
    print("[control] Init Fine. NLL restored after the last configuration")
    out.update({"configs": configs, "base_nll": base_nll, "proxies": proxies})

    if args.trained_summary:
        with open(args.trained_summary) as f:
            rows = json.load(f)
        joined = {}
        for cfg in configs:
            want = trained_args(cfg)
            hits = [r for r in rows if all(same(r["config"].get(k), v) for k, v in want.items())]
            if len(hits) != 1:
                print(f"[join] {cfg['label']}: {len(hits)} trained rows, skipped")
                continue
            joined[cfg["label"]] = {"n": hits[0]["n"], **{m: hits[0]["stats"][m]["mean"] for m in METRICS}}
        print(f"[join] {len(joined)} of {len(configs)} configurations have trained runs")
        if len(joined) < 3:
            raise SystemExit("fewer than 3 joined configurations: no correlations")
        corr = {}
        labels = list(joined)
        for p_ in PROXIES:
            corr[p_] = {}
            use = [l for l in labels if math.isfinite(proxies[l][p_])]
            for m in METRICS:
                rho, pval = scipy.stats.spearmanr([proxies[l][p_] for l in use], [joined[l][m] for l in use])
                corr[p_][m] = {"spearman": float(rho), "p": float(pval), "n": len(use)}
        print("[corr] Spearman rho (p) across configurations, proxy vs trained metric:")
        for p_ in PROXIES:
            print(f"[corr] {p_:16s} " + "  ".join(f"{m} {corr[p_][m]['spearman']:+.2f} ({corr[p_][m]['p']:.3f})"
                                                   for m in METRICS))
        for l in labels:
            print(f"[pair] {l:28s} Q_f {proxies[l]['Q_f']:.4f} Q_r {proxies[l]['Q_r']:.4f} dL_f {proxies[l]['dL_f']:.4f}"
                  f" dL_r {proxies[l]['dL_r']:.4f} dL_world {(proxies[l]['dL_ra'] + proxies[l]['dL_wf']) / 2:.4f}"
                  f" | n {joined[l]['n']} " + " ".join(f"{m[:4]} {joined[l][m]:.4f}" for m in METRICS))
        out.update({"trained": joined, "correlations": corr})
    with open(args.out_json, "w") as f:
        json.dump(to_jsonable(out), f, indent=2)
    print(f"[done] {args.out_json}")


if __name__ == "__main__":
    main()
