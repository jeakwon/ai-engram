"""Where the restriction bases of ReCAST sit relative to the forget and retain activations, and
what the warm-start shrink does to each set's layer outputs - no training, no evaluation.

Hypothesis tested: whitening (basis C) favours input directions with little retain energy, where
the forget activations are weak too, so its shrink changes forget outputs little; basis D aims at
directions where the forget activations are large. For every target and basis (plain, C, D with
the gradient output side, all at the same explicit rank budget) the script reports:

- e_f, e_r: mean forget / retain input energy per covector of the basis, relative to an average
  direction (tr(Vh C Vh^T) / r / (tr(C) / d));
- q_f, q_r: the warm-start change of the layer outputs on forget / retain answer tokens,
  E||D x||^2 / E||W x||^2 = tr(D C D^T) / tr(W C W^T), with D = W* - W;
- the selectivity q_f / q_r and the relative weight change ||D|| / ||W||;
- for an lm_head tied to the input embedding (``--untie_lm_head no``), also the change of the
  embedding rows the forget / retain inputs read, sum_t p(t) ||D_t||^2 / sum_t p(t) ||W_t||^2 over
  the token frequencies p of all non-padding input tokens (q_emb_f, q_emb_r and their ratio); with
  ``--untie_lm_head yes`` the head gets its own copy first and the embedding is not edited.

Positive control: equal ranks for the three bases at every target, or the script stops.

Usage (GH200 node, ai-engram-gh200 environment, offline cache): every argument is required.
"""

import argparse
import json
import os
import platform

from recast_core import (collect_input_covariances, collect_mean_gradients, parse_blocks, select_targets,
                         untie_output_embedding, warm_start)
from table3_tofu import Stage, git_info, to_jsonable

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base_id", required=True)
    p.add_argument("--dataset_id", required=True)
    p.add_argument("--forget_split", required=True)
    p.add_argument("--retain_split", required=True)
    p.add_argument("--targets", required=True)
    p.add_argument("--target_blocks", required=True)
    p.add_argument("--attn_implementation", required=True)
    p.add_argument("--grad_batch_size", type=int, required=True)
    p.add_argument("--s_min", type=float, required=True)
    p.add_argument("--restriction_evr", type=float, required=True)
    p.add_argument("--rho", type=float, required=True, help="whitening / representation regulariser, as in the runs")
    p.add_argument("--rank_multiplier", type=float, required=True, help="explicit budget m, as in the runs")
    p.add_argument("--untie_lm_head", required=True, choices=["yes", "no"],
                   help="yes: the head gets its own weight copy before anything is measured")
    p.add_argument("--out_json", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    import torch.nn as nn
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.utils.data import DataLoader

    from engram.benchmarks import _tofu_official as T

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    timings = {}
    out = {"args": vars(args), "git": git_info(repo), "host": platform.node(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "timings_s": timings}
    print(f"[run] args={json.dumps(vars(args))}")
    device = "cuda"
    forget = load_dataset(args.dataset_id, args.forget_split)["train"]
    retain = load_dataset(args.dataset_id, args.retain_split)["train"]
    tok = AutoTokenizer.from_pretrained(args.base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = (AutoModelForCausalLM.from_pretrained(args.base_id, torch_dtype=torch.float32,
                                                  attn_implementation=args.attn_implementation).to(device))
    if args.untie_lm_head == "yes":
        print(f"[untie] {json.dumps(untie_output_embedding(model))}")
    tied = model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
    print(f"[model] lm_head shares the input embedding tensor: {tied}")
    targets = select_targets(model, args.targets, blocks=parse_blocks(args.target_blocks))

    def feats(b):
        return {k: b[k].to(device) for k in ("input_ids", "attention_mask", "labels")}

    def loader(ds):
        return DataLoader(T.QADataset(ds, tok, akey="answer"), batch_size=args.grad_batch_size,
                          collate_fn=T.Collator(tok, "right"))

    with Stage("gradients", timings):
        g_r, _ = collect_mean_gradients(model, loader(retain), targets, feats)
        g_f, _ = collect_mean_gradients(model, loader(forget), targets, feats)
    with Stage("covariances", timings):
        c_r, n_r = collect_input_covariances(model, loader(retain), targets, feats)
        c_f, n_f = collect_input_covariances(model, loader(forget), targets, feats)
    print(f"[data] answer tokens retain {n_r}, forget {n_f}; targets {len(targets)}")
    token_freq = None
    if tied and "lm_head" in targets:  # the head edit also rewrites the embedding rows the inputs read
        vocab = targets["lm_head"].out_features
        token_freq, totals = {}, {}
        for side, ds in (("f", forget), ("r", retain)):
            counts = torch.zeros(vocab, dtype=torch.float64, device=device)
            for b in loader(ds):
                ids = b["input_ids"][b["attention_mask"].bool()].to(device)
                counts.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float64))
            totals[side] = int(counts.sum())
            token_freq[side] = counts / counts.sum()
        print(f"[data] input tokens (non-padding) forget {totals['f']}, retain {totals['r']}; distinct "
              f"{int((token_freq['f'] > 0).sum())} / {int((token_freq['r'] > 0).sum())}")

    def copies():
        made = {}
        for n, m in targets.items():
            lin = nn.Linear(m.in_features, m.out_features, bias=False, device=device, dtype=torch.float32)
            with torch.no_grad():
                lin.weight.copy_(m.weight)
            made[n] = lin
        return made

    common = dict(rank_multiplier=args.rank_multiplier)
    kinds = {
        "plain": dict(),
        "C": dict(whitening=c_r, whitening_rho=args.rho),
        "D": dict(representation_output="gradient", forget_covariances=c_f, retain_covariances=c_r,
                  whitening_rho=args.rho),
    }
    per_target, ranks = {}, {}
    for kind, extra in kinds.items():
        with Stage(f"basis {kind}", timings):
            work = copies()
            restrictions, _ = warm_start(work, g_r, args.s_min, g_f, args.restriction_evr, **common, **extra)
        for n, r in restrictions.items():
            if not r.basis.active:
                continue
            w = targets[n].weight.detach().float()
            d = r.baseline.float() - w
            vh = r.basis.vh.float()
            row = {}
            for side, c in (("f", c_f[n]), ("r", c_r[n])):
                avg = float(torch.trace(c)) / c.shape[0]
                row[f"e_{side}"] = float(torch.trace(vh @ c @ vh.T)) / vh.shape[0] / avg
                # tr(D C D^T) as a sum of an elementwise product: no out x out matrix (the head has 128k rows)
                row[f"q_{side}"] = float(((d @ c) * d).sum() / ((w @ c) * w).sum())
            row["selectivity"] = row["q_f"] / row["q_r"]
            if token_freq is not None and n == "lm_head":
                d_rows, w_rows = d.double().square().sum(1), w.double().square().sum(1)
                for side in ("f", "r"):
                    row[f"q_emb_{side}"] = float((token_freq[side] * d_rows).sum() / (token_freq[side] * w_rows).sum())
                row["selectivity_emb"] = row["q_emb_f"] / row["q_emb_r"]
                print(f"[result] lm_head {kind:5s} embedding rows: q_emb_f {row['q_emb_f']:.4f}  q_emb_r "
                      f"{row['q_emb_r']:.4f}  selectivity_emb {row['selectivity_emb']:.4f}")
            row["weight_change"] = float(d.norm() / w.norm())
            row["rank"] = r.basis.rank
            per_target.setdefault(n, {})[kind] = row
            ranks.setdefault(n, {})[kind] = r.basis.rank
        del work, restrictions
        torch.cuda.empty_cache()
    bad = [n for n, rk in ranks.items() if len(set(rk.values())) != 1]
    if bad:
        raise SystemExit(f"bases differ in rank at {bad[:3]}")
    print("[control] the three bases have equal ranks at every target")

    keys = ("e_f", "e_r", "q_f", "q_r", "selectivity", "weight_change")
    summary = {}
    for proj in PROJECTIONS + ("all",):
        names = [n for n in per_target if proj == "all" or n.endswith(proj)]
        if not names:
            continue
        summary[proj] = {kind: {k: sum(per_target[n][kind][k] for n in names) / len(names) for k in keys}
                         for kind in kinds}
    for proj, kinds_row in summary.items():
        for kind, s in kinds_row.items():
            print(f"[result] {proj:9s} {kind:5s} " + "  ".join(f"{k} {s[k]:.4f}" for k in keys))
    out.update({"answer_tokens": {"retain": n_r, "forget": n_f}, "summary": summary, "per_target": per_target})
    with open(args.out_json, "w") as f:
        json.dump(to_jsonable(out), f, indent=2)
    print(f"[done] {args.out_json}")


if __name__ == "__main__":
    main()
