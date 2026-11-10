"""Model Utility sub-metrics of the Table 3 anchors (Init Fine. and Retain).

``engram.benchmarks._tofu_official.compute_full_tofu`` keeps only the harmonic mean of the nine
Model Utility sub-metrics (retain, real authors, world facts x probability, ROUGE, truth ratio).
ReCAST runs now store the nine values; this script gives them for the anchors, so a run's
sub-metrics can be read relative to Init Fine. Settings as the Table 3 run (24952059): base
tokenizer for both models, bfloat16, sdpa, batch 16.
The greedy generations behind the three ROUGE scores are kept too (``tofu_capture.py``).
Positive control: each harmonic mean must equal the model_utility of the Table 3 raw metrics to
1e-4, or the script stops.

Usage (GH200 node, ai-engram-gh200 environment, offline cache): every argument is required.
"""

import argparse
import json
import os
import platform

from table3_tofu import Stage, git_info, to_jsonable
from tofu_capture import TofuCapture


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base_id", required=True, help="Init Fine. model; its tokenizer serves both models (as Table 3)")
    p.add_argument("--retain_id", required=True, help="Retain model")
    p.add_argument("--dataset_id", required=True)
    p.add_argument("--model_dtype", required=True, choices=["bfloat16"])
    p.add_argument("--attn_implementation", required=True)
    p.add_argument("--eval_batch_size", type=int, required=True)
    p.add_argument("--anchors_raw", required=True, help="raw_metrics.json of the Table 3 run")
    p.add_argument("--out_json", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from engram.benchmarks import _tofu_official as T

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    timings = {}
    out = {"args": vars(args), "git": git_info(repo), "host": platform.node(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "timings_s": timings, "models": {}}
    print(f"[run] args={json.dumps(vars(args))}")
    with open(args.anchors_raw) as f:
        anchors = json.load(f)
    with Stage("load datasets", timings):
        rp = load_dataset(args.dataset_id, "retain_perturbed")["train"]
        ra = load_dataset(args.dataset_id, "real_authors_perturbed")["train"]
        wf = load_dataset(args.dataset_id, "world_facts_perturbed")["train"]
    print(f"[data] retain_perturbed={len(rp)} real_authors_perturbed={len(ra)} world_facts_perturbed={len(wf)}")
    tok = AutoTokenizer.from_pretrained(args.base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        print(f"[override] tokenizer pad_token was None -> eos_token {tok.eos_token!r} (as the Table 3 run)")

    for row, model_id in (("Init Fine.", args.base_id), ("Retain", args.retain_id)):
        model = (AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                      attn_implementation=args.attn_implementation)
                 .to("cuda").eval())
        with TofuCapture(T, ("retain", "real_authors", "world_facts")) as capture:
            with Stage(f"model utility {row}", timings):
                mu, _ = T.compute_model_utility(model, tok, rp, ra, wf, bs=args.eval_batch_size)
        subs, generations = capture.check(mu)
        ref = anchors[row]["model_utility"]
        print(f"[control] {row}: model_utility {mu:.10f}, Table 3 raw {ref:.10f}, difference {mu - ref:.2e}; "
              f"capture consistent")
        if abs(mu - ref) > 1e-4:
            raise SystemExit(f"{row}: model utility differs from the Table 3 run by {mu - ref:.2e}")
        print(f"[result] {row}: {json.dumps({k: round(v, 4) for k, v in subs.items()})}")
        out["models"][row] = {"model_id": model_id, "model_utility": mu, "table3_model_utility": ref, "subs": subs,
                              "generations": generations}
        del model
        torch.cuda.empty_cache()
    with open(args.out_json, "w") as f:
        json.dump(to_jsonable(out), f, indent=2)
    print(f"[done] {args.out_json}")


if __name__ == "__main__":
    main()
