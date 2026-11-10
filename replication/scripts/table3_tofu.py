"""Replicate Table 3 of arXiv 2606.14997 (Llama-3.2-1B-Instruct, TOFU forget10).

Rows: Init Fine. (the TOFU fine-tuned model), Retain (the gold model trained without the forget
set), Engram (alpha = 0.6) and Engram (alpha_W-Norm); columns: Memorization, Utility, Privacy,
Exact Memorization (EM) and log10 Forget Quality (FQ), plus the Overall score of Table 6.

The code path is the upstream reproduction, tests/test_tofu_evaluate.py: answer-token-masked
covariances collected with EngramEditor, the closed-form engram, and the official TOFU metrics
of engram.benchmarks._tofu_official (identical code to tests/_tofu_evaluate.py). The first
--inverse_methods entry gives the main result; any further entry is a sensitivity check on the
same statistics.

Usage (on a GH200 node, inside the ai-engram-gh200 environment, offline cache):
    python replication/scripts/table3_tofu.py --base_id ... --retain_id ... (all arguments required)
"""

import argparse
import json
import os
import platform
import subprocess
import time
from collections import Counter

PAPER_TABLE3 = {  # Table 3 (Mem., Util., Priv., EM, FQ) and the Overall column of Table 6
    "Init Fine.": {"Memorization": 0.0000, "Utility": 1.0000, "Privacy": 0.0038, "EM": 1.0000,
                   "FQ": -21.4083, "Overall": 0.0000},
    "Retain": {"Memorization": 1.0000, "Utility": 0.9933, "Privacy": 1.0000, "EM": 0.0000,
               "FQ": 0.0000, "Overall": 0.9977},
    "Engram (alpha=0.6)": {"Memorization": 0.9176, "Utility": 0.8801, "Privacy": 0.4832,
                           "EM": 0.0069, "FQ": -0.6125, "Overall": 0.6984},
    "Engram (alpha_W-Norm)": {"Memorization": 0.9627, "Utility": 0.9256, "Privacy": 0.6453,
                              "EM": 0.0276, "FQ": -0.0637, "Overall": 0.8177},
}
PAPER_BASELINES = {  # Table 3, quoted from the paper; not replicated (scope decision 2026-10-21)
    "GradDiff": (0.1849, 0.9934, 0.1471, 0.7182, -15.8262),
    "IdkNLL": (0.4547, 0.9102, 0.0578, 0.6211, -13.4845),
    "UNDIAL": (0.5644, 0.9155, 0.2272, 0.7399, -11.3335),
    "RMU": (0.8660, 0.7471, 0.6799, 0.2953, -0.2357),
    "AltPO": (0.9254, 0.9864, 0.9549, 0.0953, -0.7424),
    "NPO": (0.9339, 0.9501, 0.9484, 0.0948, -1.9986),
    "SimNPO": (0.9435, 0.9706, 0.9232, 0.1035, -0.0897),
    "IdkDPO": (0.9532, 0.9660, 0.7751, 0.1155, -1.1856),
}
COLUMNS = ["Memorization", "Utility", "Privacy", "EM", "FQ", "Overall"]


def parse_args():
    p = argparse.ArgumentParser(description="Replicate Table 3 of arXiv 2606.14997")
    # Models and data
    p.add_argument("--base_id", required=True, help="Init Fine. model (TOFU fine-tuned on full)")
    p.add_argument("--retain_id", required=True, help="Retain model (trained without the forget split)")
    p.add_argument("--dataset_id", required=True)
    p.add_argument("--forget_split", required=True, help="e.g. forget10; statistics use <split>_perturbed")
    p.add_argument("--retain_split", required=True, help="e.g. retain90; only for the reference-set check")
    p.add_argument("--holdout_split", required=True, help="e.g. holdout10; membership-inference holdout")
    p.add_argument("--reference_config", required=True, help="dataset config of the reference (total) set")
    p.add_argument("--n_total", type=int, required=True, help="items drawn from the reference config")
    p.add_argument("--total_seed", type=int, required=True, help="shuffle seed of the reference draw")
    p.add_argument("--model_dtype", required=True, choices=["bfloat16", "float32"])
    p.add_argument("--attn_implementation", required=True)
    # Engram
    p.add_argument("--cov_batch_size", type=int, required=True)
    p.add_argument("--target_modules", required=True,
                   help="'all-linear' (every nn.Linear, incl. lm_head) or a comma-separated suffix list")
    p.add_argument("--inverse_methods", nargs="+", required=True, choices=["eigh", "svd"],
                   help="first = main result, others = sensitivity checks")
    p.add_argument("--plain_alpha", type=float, required=True, help="Engram (alpha = k), uniform strength")
    p.add_argument("--wnorm_alpha", type=float, required=True, help="global strength of alpha_W-Norm")
    p.add_argument("--wnorm_power", type=float, required=True, help="exponent on the max-rescaled W-Norm")
    # Evaluation, bookkeeping
    p.add_argument("--eval_batch_size", type=int, required=True)
    p.add_argument("--seed", type=int, required=True, help="torch seed (greedy decoding; no sampling)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--wandb_mode", required=True, choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_project", required=True)
    p.add_argument("--wandb_tags", nargs="+", required=True)
    return p.parse_args()


def to_jsonable(x):
    import numpy as np
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def git_info(repo):
    def run(*cmd):
        try:
            return subprocess.check_output(["git", "-C", repo, *cmd], text=True).strip()
        except Exception as e:  # noqa: BLE001 -- provenance is best effort, reported as such
            return f"unknown ({e})"
    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"),
            "dirty_files": run("status", "--porcelain", "--untracked-files=no")}


class Stage:
    """Prints the elapsed time of a stage; the timings size later Slurm limits."""

    def __init__(self, name, timings):
        self.name, self.timings = name, timings

    def __enter__(self):
        self.t0 = time.time()
        print(f"[stage] {self.name} ...", flush=True)

    def __exit__(self, *exc):
        dt = time.time() - self.t0
        self.timings[self.name] = round(dt, 1)
        print(f"[stage] {self.name} done in {dt:.0f} s", flush=True)


def qa_pairs(ds):
    return Counter(zip(ds["question"], ds["answer"]))


def check_reference_set(load_dataset, args, full, forget_perturbed):
    """Positive control: is Sigma_total = Sigma_forget + Sigma_retain (Eq. 4 with X- = retain)?"""
    print(f"[check] reference: {args.reference_config} rows={len(full)}, drawn n_total={args.n_total} "
          f"(covers the whole config: {args.n_total == len(full)})")
    try:
        forget = load_dataset(args.dataset_id, args.forget_split)["train"]
        retain = load_dataset(args.dataset_id, args.retain_split)["train"]
    except Exception as e:  # noqa: BLE001 -- the check is informative, not part of the experiment
        print(f"[check] SKIPPED reference-set composition: could not load splits ({e!r})")
        return None
    pf, pr, pfull, pfp = qa_pairs(forget), qa_pairs(retain), qa_pairs(full), qa_pairs(forget_perturbed)
    out = {
        "full_rows": len(full), "forget_rows": len(forget), "retain_rows": len(retain),
        "forget_perturbed_pairs_equal_forget_pairs": pfp == pf,
        "full_pairs_equal_forget_plus_retain": pfull == pf + pr,
        "forget_pairs_missing_from_full": sum((pf - pfull).values()),
        "n_total_covers_full": args.n_total == len(full),
    }
    print(f"[check] reference-set composition: {out}")
    return out


def main():
    args = parse_args()
    import numpy as np
    import torch
    import transformers
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import engram
    from engram import EditorConfig, EngramEditor, compose, count_ratio, weight_norm
    from engram.benchmarks import _tofu_official as T
    from engram.inverse import default_rtol

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda"
    timings = {}
    runtime = {
        "git": git_info(repo), "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "engram": f"{engram.__version__} from {engram.__file__}", "torch": torch.__version__,
        "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"), "hf_datasets_offline": os.environ.get("HF_DATASETS_OFFLINE"),
    }
    print(f"[run] args={json.dumps(vars(args))}")
    print(f"[run] runtime={json.dumps(runtime)}")

    import wandb
    run = wandb.init(project=args.wandb_project, name=args.run_name, tags=args.wandb_tags,
                     mode=args.wandb_mode, config={**vars(args), "runtime": runtime})
    print(f"[wandb] mode={args.wandb_mode} url={getattr(run, 'url', None)}")

    raw_path = os.path.join(args.out_dir, "raw_metrics.json")
    raw = {}

    def save_raw():
        with open(raw_path, "w") as f:
            json.dump(to_jsonable(raw), f)

    # ---------------- data ----------------
    with Stage("load datasets", timings):
        D = {
            "fp": load_dataset(args.dataset_id, f"{args.forget_split}_perturbed")["train"],
            "ho": load_dataset(args.dataset_id, args.holdout_split)["train"],
            "rp": load_dataset(args.dataset_id, "retain_perturbed")["train"],
            "ra": load_dataset(args.dataset_id, "real_authors_perturbed")["train"],
            "wf": load_dataset(args.dataset_id, "world_facts_perturbed")["train"],
        }
        full = load_dataset(args.dataset_id, args.reference_config)["train"]
        total = full.shuffle(seed=args.total_seed).select(range(args.n_total))
        print(f"[data] rows: {({k: len(v) for k, v in D.items()})} reference={len(total)}")
        reference_check = check_reference_set(load_dataset, args, full, D["fp"])

    tok = AutoTokenizer.from_pretrained(args.base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        print(f"[override] tokenizer pad_token was None -> eos_token {tok.eos_token!r} (as upstream test)")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.model_dtype]

    def load(mid):
        return (AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dtype, attn_implementation=args.attn_implementation)
                .to(device).eval())

    # ---------------- statistics (answer-token masked) ----------------
    base = load(args.base_id)
    base_head = base.get_output_embeddings().weight
    base_head_copy = base_head.detach().clone()  # to prove the edits never touch the base model
    print(f"[model] {args.base_id}: tie_word_embeddings={base.config.tie_word_embeddings}, "
          f"lm_head shares the input embedding tensor: "
          f"{base_head.data_ptr() == base.get_input_embeddings().weight.data_ptr()}")
    editor = EngramEditor(base, EditorConfig(storage_device=torch.device(device), absorb_bias=True))
    target_modules = None if args.target_modules == "all-linear" else args.target_modules.split(",")

    def feats(b):
        return {"input_ids": b["input_ids"].to(device), "attention_mask": b["attention_mask"].to(device)}

    def covdl(ds):
        return DataLoader(T.QADataset(ds, tok, akey="answer"), batch_size=args.cov_batch_size,
                          collate_fn=T.Collator(tok, "right"))

    mask_fn = lambda b: b["labels"] != T.IGNORE_INDEX  # noqa: E731 -- covariance over answer tokens only
    with Stage("collect statistics", timings):
        g_forget = editor.collect_statistics(covdl(D["fp"]), target_modules=target_modules,
                                             batch_fn=feats, mask_fn=mask_fn)
        g_total = editor.collect_statistics(covdl(total), target_modules=target_modules,
                                            batch_fn=feats, mask_fn=mask_fn)
    counts = sorted({(g_forget.count[k], g_total.count[k]) for k in g_forget.keys()})
    dims = sorted({g_total[k].shape[-1] for k in g_total.keys()})
    print(f"[stats] layers={len(g_forget)} distinct (n, N) answer-token counts={counts} "
          f"(one pair = the n/N factor is the same for every layer)")
    print(f"[stats] pseudo-inverse cut rtol = D * eps_float32: "
          f"{({d: default_rtol(d) for d in dims})} (Appendix B.3 states 1e-6)")

    # ---------------- anchors ----------------
    with Stage("evaluate Init Fine.", timings):
        raw["Init Fine."] = T.compute_full_tofu(base, tok, D, bs=args.eval_batch_size)
        save_raw()
    with Stage("evaluate Retain", timings):
        retain_model = load(args.retain_id)
        raw["Retain"] = T.compute_full_tofu(retain_model, tok, D, bs=args.eval_batch_size)
        del retain_model
        torch.cuda.empty_cache()
        save_raw()
    ft, rt = raw["Init Fine."], raw["Retain"]

    # ---------------- Engram conditions ----------------
    conditions = [
        (f"Engram (alpha={args.plain_alpha:g})", args.plain_alpha, lambda: count_ratio(1.0)),
        ("Engram (alpha_W-Norm)", args.wnorm_alpha, lambda: compose(count_ratio(1.0), weight_norm(args.wnorm_power))),
    ]
    engram_kwargs = {
        "eigh": dict(compute_erank=False, rank_fraction=None, inverse_method="eigh", inverse_solver="exact",
                     rank_floor="rtol", inverse_precision=torch.float64, condition_cap=None, cut="rtol"),
        # The svd path is torch.linalg.pinv in float32 with the same rtol; the eigh-only
        # arguments (solver, floor, cut, inverse_precision) are not used on it.
        "svd": dict(compute_erank=False, rank_fraction=None, inverse_method="svd", condition_cap=None),
    }
    controls = {}
    previous = None
    for method in args.inverse_methods:
        suffix = "" if method == args.inverse_methods[0] else f" [{method}]"
        with Stage(f"compute engram ({method})", timings):
            result = editor.compute_engram_weights(g_forget, g_total, **engram_kwargs[method])
        print(f"[engram] {method}: {engram_kwargs[method]} -> {len(result.layers)} layers")
        rel = {k: i.projection.norm().item() / i.weight_fro for k, i in result.layers.items()}
        top = sorted(rel, key=rel.get, reverse=True)[:8]
        print(f"[engram] {method}: largest ||P||/||W|| (Fig. 9 expects q/k/gate): "
              f"{[(k, round(rel[k], 4)) for k in top]}")
        if previous is not None:
            diffs = [(result.layers[k].projection - previous.layers[k].projection).norm().item()
                     / previous.layers[k].projection.norm().item() for k in previous.layers]
            controls[f"projection_rel_diff_{method}_vs_{args.inverse_methods[0]}"] = {
                "mean": float(np.mean(diffs)), "max": float(np.max(diffs))}
            print(f"[engram] {method} vs {args.inverse_methods[0]}: relative projection difference "
                  f"mean={np.mean(diffs):.4%} max={np.max(diffs):.4%}")
            del previous
            torch.cuda.empty_cache()
        for name, alpha, scale_fn in conditions:
            row = name + suffix
            scale = scale_fn()
            factors = scale(result.layers)
            print(f"[edit] {row}: alpha={alpha}, per-layer factor f_l min={min(factors.values()):.4e} "
                  f"max={max(factors.values()):.4e} -> alpha*f_l max={alpha * max(factors.values()):.4e}")
            with Stage(f"edit + evaluate {row}", timings):
                edited = editor.apply(result, alpha=alpha, scale=scale).eval()
                tied = (edited.get_output_embeddings().weight.data_ptr()
                        == edited.get_input_embeddings().weight.data_ptr())
                em, bm = dict(edited.named_modules()), dict(base.named_modules())
                change = {k: ((em[k].weight.float() - bm[k].weight.float()).norm()
                              / bm[k].weight.float().norm()).item()
                          for k in (top[0], "lm_head") if k in result.layers}
                del em, bm
                print(f"[edit] {row}: relative weight change {change}; lm_head still tied to the input "
                      f"embedding after the edit: {tied}")
                raw[row] = T.compute_full_tofu(edited, tok, D, bs=args.eval_batch_size)
                del edited
                torch.cuda.empty_cache()
                save_raw()
        previous = result
    del previous, g_forget, g_total
    torch.cuda.empty_cache()
    controls["base_untouched"] = bool(torch.equal(base.get_output_embeddings().weight, base_head_copy))
    del base_head_copy
    print(f"[check] base model untouched by the edits (apply works on deep copies): {controls['base_untouched']}")

    # ---------------- scores ----------------
    scores = {row: T.evaluate_scores(m, rt, ft) for row, m in raw.items()}
    lines = ["| Method | " + " | ".join(COLUMNS) + " |", "|" + "---|" * (len(COLUMNS) + 1)]
    for row, s in scores.items():
        lines.append(f"| {row} | " + " | ".join(f"{s[c]:.4f}" for c in COLUMNS) + " |")
        paper = PAPER_TABLE3.get(row)
        if paper:
            lines.append(f"| paper | " + " | ".join(f"{paper[c]:.4f}" for c in COLUMNS) + " |")
    for row, vals in PAPER_BASELINES.items():
        lines.append(f"| {row} (paper, not replicated) | " + " | ".join(f"{v:.4f}" for v in vals) + " | - |")
    table = "\n".join(lines)
    print("[result] Table 3 (+ Overall of Table 6)\n" + table)

    summary = {"args": vars(args), "runtime": runtime, "timings_s": timings, "reference_check": reference_check,
               "controls": controls, "scores": scores, "paper": PAPER_TABLE3, "paper_baselines_not_replicated": PAPER_BASELINES}
    with open(os.path.join(args.out_dir, "table3.json"), "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)
    with open(os.path.join(args.out_dir, "table3.md"), "w") as f:
        f.write(table + "\n")
    print(f"[result] written to {args.out_dir}/table3.json, table3.md, raw_metrics.json")

    wt = wandb.Table(columns=["Method"] + COLUMNS, data=[[r] + [s[c] for c in COLUMNS] for r, s in scores.items()])
    run.log({"table3": wt, **{f"{r}/{c}": s[c] for r, s in scores.items() for c in COLUMNS}})
    run.summary.update({"timings_s": timings, "controls": to_jsonable(controls)})
    run.finish()


if __name__ == "__main__":
    main()
