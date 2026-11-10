"""The Engram edits of Table 3 saved as Hugging Face checkpoints, so that later jobs can start from them.

Engram (alpha = k) and Engram (alpha_W-Norm) are computed exactly as table3_tofu.py does: answer-token covariances
of ``<forget_split>_perturbed`` and of the reference draw collected with EngramEditor, the eigh closed form, the
same uniform and W-Norm strengths. Each edited model and the base tokenizer go to ``--out_root/<tag>``; the
checkpoint is reloaded and scored with the official TOFU metrics against the Table 3 anchors.

Positive controls: every reloaded tensor equals the edited one and lm_head stays tied to the input embedding, or
the script stops; the reloaded model's scores are printed next to those of the Table 3 run (``--table3_raw``).

Usage (GH200 node, ai-engram-gh200 environment, offline cache): see engram_edit_gh200.sh.
"""

import argparse
import json
import os
import platform

from table3_tofu import COLUMNS, Stage, git_info, to_jsonable


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for k in ("--base_id", "--dataset_id", "--forget_split", "--holdout_split", "--reference_config",
              "--attn_implementation", "--target_modules", "--table3_raw", "--out_root", "--out_json"):
        p.add_argument(k, required=True)
    for k in ("--n_total", "--total_seed", "--cov_batch_size", "--eval_batch_size"):
        p.add_argument(k, type=int, required=True)
    for k in ("--plain_alpha", "--wnorm_alpha", "--wnorm_power"):
        p.add_argument(k, type=float, required=True)
    p.add_argument("--model_dtype", required=True, choices=["bfloat16", "float32"])
    args = p.parse_args()

    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from engram import EditorConfig, EngramEditor, compose, count_ratio, weight_norm
    from engram.benchmarks import _tofu_official as T

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    timings = {}
    out = {"args": vars(args), "git": git_info(repo), "host": platform.node(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "timings_s": timings, "rows": {}}
    device = "cuda"
    D = {"fp": load_dataset(args.dataset_id, f"{args.forget_split}_perturbed")["train"],
         "ho": load_dataset(args.dataset_id, args.holdout_split)["train"],
         "rp": load_dataset(args.dataset_id, "retain_perturbed")["train"],
         "ra": load_dataset(args.dataset_id, "real_authors_perturbed")["train"],
         "wf": load_dataset(args.dataset_id, "world_facts_perturbed")["train"]}
    total = load_dataset(args.dataset_id, args.reference_config)["train"].shuffle(seed=args.total_seed).select(
        range(args.n_total))
    with open(args.table3_raw) as f:
        table3 = json.load(f)
    ft, rt = table3["Init Fine."], table3["Retain"]

    tok = AutoTokenizer.from_pretrained(args.base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.model_dtype]

    def load(mid):
        return (AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dtype, attn_implementation=args.attn_implementation)
                .to(device).eval())

    base = load(args.base_id)
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
    with Stage("compute engram (eigh)", timings):
        result = editor.compute_engram_weights(
            g_forget, g_total, compute_erank=False, rank_fraction=None, inverse_method="eigh", inverse_solver="exact",
            rank_floor="rtol", inverse_precision=torch.float64, condition_cap=None, cut="rtol")
    print(f"[engram] eigh -> {len(result.layers)} layers")
    del g_forget, g_total
    torch.cuda.empty_cache()

    def tag(x):
        return f"{x:g}".replace(".", "p")

    conditions = [
        (f"Engram (alpha={args.plain_alpha:g})", f"engram_alpha{tag(args.plain_alpha)}", args.plain_alpha,
         count_ratio(1.0)),
        ("Engram (alpha_W-Norm)", f"engram_wnorm{tag(args.wnorm_alpha)}_p{tag(args.wnorm_power)}", args.wnorm_alpha,
         compose(count_ratio(1.0), weight_norm(args.wnorm_power))),
    ]
    for row, name, alpha, scale in conditions:
        path = os.path.join(args.out_root, name)
        with Stage(f"edit + save {row}", timings):
            edited = editor.apply(result, alpha=alpha, scale=scale).eval()
            edited.save_pretrained(path)
            tok.save_pretrained(path)
            reference = {n: t.detach().to("cpu", copy=True) for n, t in edited.state_dict().items()}
            del edited
            torch.cuda.empty_cache()
        with Stage(f"reload + evaluate {row}", timings):
            again = load(path)
            state = again.state_dict()
            tied = again.get_output_embeddings().weight.data_ptr() == again.get_input_embeddings().weight.data_ptr()
            differ = sorted(n for n in reference if n not in state or not torch.equal(state[n].cpu(), reference[n]))
            if set(state) != set(reference) or differ or not tied:
                raise SystemExit(f"{row}: the reloaded checkpoint differs from the edit (tensors {differ[:3]}, "
                                 f"names equal {set(state) == set(reference)}, lm_head tied {tied})")
            del state, reference
            raw = T.compute_full_tofu(again, tok, D, bs=args.eval_batch_size)
            del again
            torch.cuda.empty_cache()
        scores = T.evaluate_scores(raw, rt, ft)
        ref = T.evaluate_scores(table3[row], rt, ft) if row in table3 else None
        print(f"[control] {row}: reloaded checkpoint equals the edit tensor by tensor; lm_head tied")
        print(f"[result] {row} -> {path}: " + ", ".join(f"{c}={scores[c]:.4f}" for c in COLUMNS))
        if ref is not None:
            print(f"[result] {row} in the Table 3 run: " + ", ".join(f"{c}={ref[c]:.4f}" for c in COLUMNS))
        out["rows"][row] = {"path": path, "alpha": alpha, "scores": scores, "table3_scores": ref, "raw": raw}
        with open(args.out_json, "w") as f:
            json.dump(to_jsonable(out), f, indent=2)
    print(f"[done] {args.out_json}")


if __name__ == "__main__":
    main()
