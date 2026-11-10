"""Screen published unlearning checkpoints on the Table 3 columns that need no generation (Memorization, Privacy, EM,
Forget Quality), as a cheap first pass before the full evaluation of a few (user, 2026-10-29).

For each model id of its share of ``--models`` (one id per line; Slurm array task ``--task`` of ``--tasks``): load the
checkpoint in bfloat16 (as the anchors were scored), compute exactly what ``compute_full_tofu`` computes for those
columns (exact memorization, extraction strength, forget probability, truth ratio and its per-index form, the four MIA
attacks), rescale against the Init Fine. / Retain anchors with ``evaluate_scores`` and keep Memorization, Privacy, EM
and FQ. Utility needs generations (ROUGE, the gibberish classifier) and is left to the full evaluation: the anchors'
Init Fine. values stand in for its inputs here, so the Utility and Overall this script would compute are meaningless
and are not written. Each checkpoint is deleted from the Hugging Face cache after scoring.

Usage (Slurm, ai-engram-gh200 environment): python tofu_checkpoint_screen.py --models <file> --task <i> --tasks <n>
    --anchors_raw <raw_metrics.json> --out_dir <dir>
"""

import argparse
import json
import os
import shutil
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

KEEP = ("Memorization", "Privacy", "EM", "FQ")


def raw_without_generation(T, model, tok, D, bs):
    """The parts of ``compute_full_tofu`` that need no generation, under the same keys."""
    cr = T.Collator(tok, "right")
    qa = lambda d, ak: T.QADataset(d, tok, akey=ak, pwg=False)  # noqa: E731
    fp, ho = D["fp"], D["ho"]
    para = T.prob_sbi(model, qa(fp, "paraphrased_answer"), cr, bs)
    pert = T.prob_sbi(model, qa(fp, "perturbed_answer"), cr, bs)
    out = {"exact_memorization": T.metric_em(model, qa(fp, "answer"), cr, bs),
           "extraction_strength": T.metric_es(model, qa(fp, "answer"), cr, bs),
           "forget_Q_A_Prob": T.agg_prob(T.prob_sbi(model, qa(fp, "answer"), cr, bs)),
           "forget_truth_ratio": T.truth_ratio(para, pert, "closer_to_1_better"),
           "_forget_tr": T.forget_tr_per_index(para, pert), "_mia_fs": {}}
    for atk, key in (("loss", "mia_loss"), ("zlib", "mia_zlib"), ("min_k", "mia_min_k"), ("min_k++", "mia_min_k_plus_plus")):
        auc, fs = T.mia_auc(model, qa(fp, "answer"), qa(ho, "answer"), cr, atk, tok=tok, k=0.4, bs=bs)
        out[key], out["_mia_fs"][atk] = auc, fs
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--tasks", type=int, required=True)
    p.add_argument("--anchors_raw", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset_id", default="locuslab/TOFU")
    p.add_argument("--forget_split", default="forget10")
    p.add_argument("--holdout_split", default="holdout10")
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()
    from engram.benchmarks import _tofu_official as T

    with open(args.models) as fh:
        ids = [line.strip() for line in fh if line.strip()]
    mine = ids[args.task::args.tasks]
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.anchors_raw) as fh:
        anchors = json.load(fh)
    ft, rt = anchors["Init Fine."], anchors["Retain"]
    D = {"fp": load_dataset(args.dataset_id, f"{args.forget_split}_perturbed")["train"],
         "ho": load_dataset(args.dataset_id, args.holdout_split)["train"]}
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    for model_id in mine:
        out = os.path.join(args.out_dir, model_id.replace("/", "__") + ".json")
        if os.path.exists(out):
            print(f"[skip] {model_id}: {out} exists")
            continue
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda().eval()
        t1 = time.time()
        with torch.no_grad():
            raw = raw_without_generation(T, model, tok, D, args.batch_size)
        exp = dict(raw, model_utility=ft["model_utility"], forget_Q_A_gibberish=ft["forget_Q_A_gibberish"],
                   forget_Q_A_ROUGE=ft.get("forget_Q_A_ROUGE"))
        scores = {k: v for k, v in T.evaluate_scores(exp, rt, ft).items() if k in KEEP}
        raw_out = {k: v for k, v in raw.items() if not k.startswith("_")}
        with open(out, "w") as fh:
            json.dump({"model_id": model_id, "scores": scores, "raw": raw_out,
                       "seconds": {"load": t1 - t0, "score": time.time() - t1}}, fh, indent=1)
        print(f"[screen] {model_id}: " + " / ".join(f"{k} {scores[k]:.4f}" for k in KEEP)
              + f" ({t1 - t0:.0f} s load, {time.time() - t1:.0f} s score)")
        del model
        torch.cuda.empty_cache()
        cached = os.path.join(hub, "models--" + model_id.replace("/", "--"))
        if "/unlearn_" in model_id and os.path.isdir(cached):  # never the anchors, which other runs load
            shutil.rmtree(cached)


if __name__ == "__main__":
    main()
