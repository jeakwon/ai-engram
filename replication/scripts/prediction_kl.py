"""How far other models' next-token predictions are from a reference model's (Retain), on the answer tokens.

For every model of ``--model_ids`` and every split, the script compares the next-token distributions of the
reference P and the model Q at each answer token (teacher-forced, the positions TOFU's answer NLL uses) and reports
the token-weighted mean of KL(P || Q) and KL(Q || P) in nats, the median and the 90th / 99th percentiles of
KL(P || Q), the share of tokens where both put their top-1 on the same token, and the answer NLL of both models on
the same tokens. Splits: retain90 (phase two's data), forget10, holdout10 (authors neither model saw), real authors
and world facts.

Positive control: the reference against itself gives KL 0 and top-1 agreement 1 on every split, or the script stops.

Usage (GH200 node, ai-engram-gh200 environment, offline cache): see prediction_kl_gh200.sh.
"""

import argparse
import json
import os
import platform

from table3_tofu import Stage, git_info, to_jsonable

SPLITS = {"retain": None, "forget": None, "holdout": None, "real_authors": "real_authors_perturbed",
          "world_facts": "world_facts_perturbed"}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model_ids", nargs="+", required=True)
    for k in ("--reference_id", "--tokenizer_id", "--dataset_id", "--forget_split", "--retain_split", "--holdout_split",
              "--attn_implementation", "--out_json"):
        p.add_argument(k, required=True)
    p.add_argument("--batch_size", type=int, required=True)
    args = p.parse_args()

    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from engram.benchmarks import _tofu_official as T

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    timings = {}
    out = {"args": vars(args), "git": git_info(repo), "host": platform.node(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "timings_s": timings, "models": {}}
    device = "cuda"
    names = {"retain": args.retain_split, "forget": args.forget_split, "holdout": args.holdout_split}
    data = {s: load_dataset(args.dataset_id, cfg or names[s])["train"] for s, cfg in SPLITS.items()}
    print(f"[data] {({s: len(d) for s, d in data.items()})}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def load(model_id):
        return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32,
                                                    attn_implementation=args.attn_implementation).to(device).eval()

    def batches(ds):
        loader = DataLoader(T.QADataset(ds, tok, akey="answer"), batch_size=args.batch_size,
                            collate_fn=T.Collator(tok, "right"))
        for b in loader:
            yield {k: b[k].to(device) for k in ("input_ids", "attention_mask", "labels")}

    reference = load(args.reference_id)

    @torch.no_grad()
    def compare(model, ds):
        """Per answer token: KL(P || Q), KL(Q || P), top-1 agreement and both NLLs (fp32 log-softmax)."""
        kl_pq, kl_qp, agree, nll_p, nll_q = [], [], [], [], []
        for x in batches(ds):
            labels = x["labels"][:, 1:]
            mask = labels != T.IGNORE_INDEX
            lp = F.log_softmax(reference(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).logits[:, :-1].float(), -1)[mask]
            lq = F.log_softmax(model(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).logits[:, :-1].float(), -1)[mask]
            target = labels[mask]
            kl_pq.append((lp.exp() * (lp - lq)).sum(-1).cpu())
            kl_qp.append((lq.exp() * (lq - lp)).sum(-1).cpu())
            agree.append((lp.argmax(-1) == lq.argmax(-1)).float().cpu())
            nll_p.append(-lp.gather(1, target[:, None])[:, 0].cpu())
            nll_q.append(-lq.gather(1, target[:, None])[:, 0].cpu())
        kl_pq, kl_qp, agree = torch.cat(kl_pq).double(), torch.cat(kl_qp).double(), torch.cat(agree).double()
        nll_p, nll_q = torch.cat(nll_p).double(), torch.cat(nll_q).double()
        q = torch.quantile(kl_pq.float(), torch.tensor([0.5, 0.9, 0.99])).tolist()
        return {"tokens": int(kl_pq.numel()), "kl_ref_model": float(kl_pq.mean()), "kl_model_ref": float(kl_qp.mean()),
                "kl_ref_model_median": q[0], "kl_ref_model_p90": q[1], "kl_ref_model_p99": q[2],
                "top1_agreement": float(agree.mean()), "nll_reference": float(nll_p.mean()), "nll_model": float(nll_q.mean())}

    for model_id in [args.reference_id] + [m for m in args.model_ids if m != args.reference_id]:
        model = reference if model_id == args.reference_id else load(model_id)
        with Stage(f"compare {model_id}", timings):
            res = {s: compare(model, ds) for s, ds in data.items()}
        if model is not reference:
            del model
            torch.cuda.empty_cache()
        if model_id == args.reference_id:
            # Positive control: the reference against itself.
            bad = [s for s, r in res.items() if abs(r["kl_ref_model"]) > 1e-6 or r["top1_agreement"] != 1.0]
            if bad:
                raise SystemExit(f"reference against itself is not KL 0 / agreement 1 on {bad}")
            print("[control] reference against itself: KL 0 and top-1 agreement 1 on every split")
        out["models"][model_id] = res
        for s, r in res.items():
            print(f"[kl] {model_id} | {s}: KL(ref||model) {r['kl_ref_model']:.4f} (median {r['kl_ref_model_median']:.4f}, "
                  f"p90 {r['kl_ref_model_p90']:.4f}, p99 {r['kl_ref_model_p99']:.4f}), KL(model||ref) "
                  f"{r['kl_model_ref']:.4f}, top-1 agreement {r['top1_agreement']:.4f}, NLL ref {r['nll_reference']:.4f} "
                  f"model {r['nll_model']:.4f}, tokens {r['tokens']}")
        with open(args.out_json, "w") as f:
            json.dump(to_jsonable(out), f, indent=2)
    print(f"[done] {args.out_json}")


if __name__ == "__main__":
    main()
