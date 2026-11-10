"""The signed-shrink table's diagnostics for other models as updates of Init Fine.: Retain, Engram edits, ...

For every model of ``--model_ids`` (same architecture as Init Fine.: Retain, a saved Engram edit or a model saved
after phase two), with Delta = theta_model - theta_Init, the script reports the Frobenius norm of Delta and the first-order loss changes <G_f, Delta>, <G_r, Delta> with the Init
Fine. mean gradients (example-weighted means of each batch's mean answer NLL, as recast_tofu.py), over the target
profile (``--targets``, the 48 matrices of down_up_o) and over all parameters, next to the actual forget / retain
loss of Init Fine. and of the model. Init Fine. itself is the zero update.

Positive control: the Init Fine. loss equals the one recomputed after the models are loaded and freed (1e-6),
or the script stops.

Usage (GH200 node, ai-engram-gh200 environment, offline cache):
    python replication/scripts/anchor_first_order.py --base_id <Init Fine.> --model_ids <Retain> ... --dataset_id ... \
        --forget_split forget10 --retain_split retain90 --targets down_up_o --attn_implementation sdpa \
        --grad_batch_size 8 --out_json <file>
"""

import argparse
import json
import os
import platform

from recast_core import select_targets
from table3_tofu import Stage, git_info, to_jsonable


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model_ids", nargs="+", required=True)
    for k in ("--base_id", "--dataset_id", "--forget_split", "--retain_split", "--targets",
              "--attn_implementation", "--out_json"):
        p.add_argument(k, required=True)
    p.add_argument("--grad_batch_size", type=int, required=True)
    args = p.parse_args()

    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from engram.benchmarks import _tofu_official as T

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    timings = {}
    out = {"args": vars(args), "git": git_info(repo), "host": platform.node(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "timings_s": timings}
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    data = {"forget": load_dataset(args.dataset_id, args.forget_split)["train"],
            "retain": load_dataset(args.dataset_id, args.retain_split)["train"]}

    def load(model_id):
        return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32,
                                                    attn_implementation=args.attn_implementation).to(device)

    def batches(ds):
        loader = DataLoader(T.QADataset(ds, tok, akey="answer"), batch_size=args.grad_batch_size,
                            collate_fn=T.Collator(tok, "right"))
        for b in loader:
            yield {k: b[k].to(device) for k in ("input_ids", "attention_mask", "labels")}

    @torch.no_grad()
    def mean_loss(model, ds):
        model.eval()
        total, count = 0.0, 0
        for x in batches(ds):
            size = int(x["input_ids"].shape[0])
            total += float(model(**x).loss) * size
            count += size
        return total / count

    def mean_gradient(model, ds):
        """Example-weighted mean of the batch-mean-loss gradients of every parameter (eval mode, as the runs)."""
        model.eval()
        params = dict(model.named_parameters())
        sums = {n: torch.zeros_like(t, dtype=torch.float32) for n, t in params.items()}
        count = 0
        for x in batches(ds):
            size = int(x["input_ids"].shape[0])
            model.zero_grad(set_to_none=True)
            model(**x).loss.backward()
            for n, t in params.items():
                if t.grad is not None:
                    sums[n].add_(t.grad.float(), alpha=float(size))
            count += size
        model.zero_grad(set_to_none=True)
        return {n: s / count for n, s in sums.items()}

    init = load(args.base_id)
    with Stage("Init Fine. losses and gradients", timings):
        loss_init = {side: mean_loss(init, ds) for side, ds in data.items()}
        grads = {side: mean_gradient(init, ds) for side, ds in data.items()}
    init_params = {n: t.detach().clone() for n, t in init.named_parameters()}
    target_params = {n + ".weight" for n in select_targets(init, args.targets)}
    del init
    torch.cuda.empty_cache()
    models = {}
    for model_id in args.model_ids:
        other = load(model_id)
        with Stage(f"{model_id}: losses and the update", timings):
            loss = {side: mean_loss(other, ds) for side, ds in data.items()}
            o_params = dict(other.named_parameters())
            if set(o_params) != set(init_params):
                raise SystemExit(f"Init Fine. and {model_id} have different parameter names")
            update = {}
            for scope, names in (("targets", sorted(target_params)), ("all", sorted(init_params))):
                sq, dot_f, dot_r = 0.0, 0.0, 0.0
                for n in names:
                    d = (o_params[n].detach().float() - init_params[n].float()).double()
                    sq += float(d.square().sum())
                    dot_f += float((grads["forget"][n].double() * d).sum())
                    dot_r += float((grads["retain"][n].double() * d).sum())
                update[scope] = {"tensors": len(names), "update_norm": sq ** 0.5, "predicted_forget_change": dot_f,
                                 "predicted_retain_change": dot_r}
        del other, o_params
        torch.cuda.empty_cache()
        models[model_id] = {"loss": loss, "actual_change": {side: loss[side] - loss_init[side] for side in data},
                            "update": update}
    init = load(args.base_id)
    again = mean_loss(init, data["forget"])
    if abs(again - loss_init["forget"]) > 1e-6 * max(1.0, loss_init["forget"]):
        raise SystemExit(f"Init Fine. forget loss not reproduced: {again} vs {loss_init['forget']}")
    print(f"[control] Init Fine. forget loss reproduced ({again:.6f})")
    print(f"[loss] Init Fine. forget {loss_init['forget']:.4f} retain {loss_init['retain']:.4f}")
    for model_id, m in models.items():
        print(f"[loss] {model_id}: forget {m['loss']['forget']:.4f} retain {m['loss']['retain']:.4f}; actual change "
              f"forget {m['actual_change']['forget']:+.4f} retain {m['actual_change']['retain']:+.4f}")
        for scope, r in m["update"].items():
            print(f"[update] {model_id} - Init Fine. over {scope} ({r['tensors']} tensors): ||Delta|| "
                  f"{r['update_norm']:.3f}; predicted forget {r['predicted_forget_change']:+.4f}, retain "
                  f"{r['predicted_retain_change']:+.4f}")
    out.update({"loss_init": loss_init, "models": models})
    with open(args.out_json, "w") as f:
        json.dump(to_jsonable(out), f, indent=2)
    print(f"[done] {args.out_json}")


if __name__ == "__main__":
    main()
