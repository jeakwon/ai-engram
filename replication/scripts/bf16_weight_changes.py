"""Which weights a training with bf16 weights at lr 1e-5 could move: bitwise comparison of two checkpoints from one base.

Init Fine. (TOFU full) and Retain (retain90) were both fine-tuned from Llama-3.2-1B-Instruct and saved in bf16. If the
weights themselves were bf16 during training (no fp32 master copy), an update smaller than half the bf16 spacing at a
weight's value rounds back to the old value, so a weight whose every update was lost keeps its base value in both
checkpoints and the two agree on it bit for bit; a weight that moved in either differs. In the binade [2^e, 2^(e+1)) of
|w| the spacing is 2^(e-7), and a step of the peak lr survives only below the binade where half the spacing exceeds it:
bf16 weights give a sharp step there, fp32 master weights cast to bf16 at save time a gradual decline instead.

Per parameter group (projection type, embedding, norms) and per binade of |w| in the first checkpoint, the script
reports the number of entries, the share bitwise equal in both checkpoints, the share for which one step of the peak lr
would be lost (bf16(w + lr) == w and bf16(w - lr) == w), and for the unequal entries the distribution of |a - b| in
units of the spacing. ``--lr`` is the configured peak lr; the script checks it against the lr logged in each
checkpoint's trainer_state.json (logged every few steps, so the logged maximum may fall slightly below the peak).

Positive controls: two hand-picked values round as the spacing predicts, and the first checkpoint compared with itself
is bitwise equal everywhere, or the script stops.

Usage (GH200 node, ai-engram-gh200 environment, offline cache): see bf16_weight_changes_gh200.sh.
"""

import argparse
import json
import os
import platform
from collections import defaultdict

from table3_tofu import git_info, to_jsonable

E_LO, E_HI = -14, 0  # binades below 2^-13 and from 2^0 up are pooled
ULP_EDGES = [1, 2, 4, 8, 16, 64]  # |a - b| / spacing buckets: <= 1, (1, 2], ..., > 64


def bin_label(e):
    return "zero" if e is None else f"<2^{E_LO + 1}" if e <= E_LO else f">=2^{E_HI}" if e >= E_HI else f"2^{e}"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for k in ("--model_a", "--model_b", "--out_json"):
        p.add_argument(k, required=True)
    p.add_argument("--lr", type=float, required=True)
    args = p.parse_args()

    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = {"args": vars(args), "git": git_info(repo), "host": platform.node(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "checkpoints": {}}
    device = "cuda"

    weights = {}
    for key, model_id in (("a", args.model_a), ("b", args.model_b)):
        d = snapshot_download(model_id, local_files_only=True)
        with open(os.path.join(d, "trainer_state.json")) as f:
            state = json.load(f)
        lrs = [h["learning_rate"] for h in state["log_history"] if "learning_rate" in h]
        weights[key] = load_file(os.path.join(d, "model.safetensors"))
        dtypes = sorted({str(t.dtype) for t in weights[key].values()})
        out["checkpoints"][key] = {"model_id": model_id, "global_step": state["global_step"],
                                   "train_batch_size": state["train_batch_size"], "logged_max_lr": max(lrs),
                                   "n_tensors": len(weights[key]), "dtypes": dtypes}
        print(f"[load] {key} {model_id}: {out['checkpoints'][key]}")
        if dtypes != ["torch.bfloat16"]:
            raise RuntimeError(f"{model_id}: expected bf16 tensors only, found {dtypes}")
        if not 0.98 * args.lr <= max(lrs) <= args.lr:
            raise RuntimeError(f"{model_id}: logged max lr {max(lrs)} does not match --lr {args.lr}")
    if weights["a"].keys() != weights["b"].keys():
        raise RuntimeError("the two checkpoints hold different tensors")
    lr = args.lr

    # Rounding control: in [2^-10, 2^-9) the spacing is 2^-17 (half 3.8e-6 < 1e-5), in [2^-6, 2^-5) it is 2^-13 (half
    # 6.1e-5 > 1e-5), so a step of 1e-5 survives at 0.001 and is lost at 0.02.
    small, large = torch.tensor([0.001, 0.02], dtype=torch.bfloat16)
    if (small.float() + 1e-5).to(torch.bfloat16) == small or (large.float() + 1e-5).to(torch.bfloat16) != large:
        raise RuntimeError("bf16 rounding control failed")

    def group_of(name):
        return name.rsplit(".", 2)[-2]  # q_proj, ..., down_proj, embed_tokens, input_layernorm, ..., norm

    def compare(a, b, acc, group):
        a, b = a.to(device), b.to(device)
        if a.shape != b.shape:
            raise RuntimeError(f"{group}: shapes differ")
        equal = a.view(torch.int16) == b.view(torch.int16)
        a32 = a.float()
        _, exp = torch.frexp(a32)  # |a| = m 2^exp, m in [0.5, 1): binade e = exp - 1, spacing 2^(e - 7)
        e = (exp - 1).clamp(E_LO, E_HI)
        zero = a32 == 0
        lost = ((a32 + lr).to(torch.bfloat16) == a) & ((a32 - lr).to(torch.bfloat16) == a)
        ulps = (b.float() - a32).abs() / torch.ldexp(torch.ones_like(a32), exp - 8)
        bucket = torch.bucketize(ulps, torch.tensor(ULP_EDGES, device=device, dtype=torch.float32))
        for value in torch.unique(e).tolist():
            for is_zero in (False, True):
                m = (e == value) & (zero if is_zero else ~zero)
                n = int(m.sum())
                if not n:
                    continue
                row = acc[group][bin_label(None if is_zero else value)]
                row["n"] += n
                row["equal"] += int((equal & m).sum())
                row["lost_one_step"] += int((lost & m).sum())
                row["equal_and_lost"] += int((equal & lost & m).sum())
                if not is_zero:
                    counts = torch.bincount(bucket[m & ~equal], minlength=len(ULP_EDGES) + 1).tolist()
                    row["ulp_buckets"] = [x + y for x, y in zip(row["ulp_buckets"], counts)]

    def new_acc():
        return defaultdict(lambda: defaultdict(lambda: {"n": 0, "equal": 0, "lost_one_step": 0, "equal_and_lost": 0,
                                                         "ulp_buckets": [0] * (len(ULP_EDGES) + 1)}))

    control = new_acc()
    for name, t in weights["a"].items():
        compare(t, t, control, group_of(name))
    if any(r["equal"] != r["n"] for g in control.values() for r in g.values()):
        raise RuntimeError("self-comparison control failed")
    print("[control] bf16 rounding and self-comparison passed")

    acc = new_acc()
    for name in weights["a"]:
        compare(weights["a"][name], weights["b"][name], acc, group_of(name))

    order = ["zero", f"<2^{E_LO + 1}"] + [f"2^{e}" for e in range(E_LO + 1, E_HI)] + [f">=2^{E_HI}"]
    matrices = [g for g in acc if not g.endswith("norm")]
    pooled = {"matrices": matrices, "norms": [g for g in acc if g.endswith("norm")]}
    for label, members in pooled.items():
        for g in members:
            for b, r in acc[g].items():
                row = acc[label][b]
                for k in ("n", "equal", "lost_one_step", "equal_and_lost"):
                    row[k] += r[k]
                row["ulp_buckets"] = [x + y for x, y in zip(row["ulp_buckets"], r["ulp_buckets"])]

    def share(x, n):
        return x / n if n else None

    table = {}
    for g in list(acc):
        table[g] = {}
        for b in order:
            if b not in acc[g]:
                continue
            r = acc[g][b]
            unequal = r["n"] - r["equal"]
            table[g][b] = {"n": r["n"], "equal_share": share(r["equal"], r["n"]),
                           "lost_one_step_share": share(r["lost_one_step"], r["n"]),
                           "ulp_bucket_shares_of_unequal": [share(x, unequal) for x in r["ulp_buckets"]]}
        tot = {k: sum(acc[g][b][k] for b in acc[g]) for k in ("n", "equal", "lost_one_step", "equal_and_lost")}
        lost_n, kept_n = tot["lost_one_step"], tot["n"] - tot["lost_one_step"]
        table[g]["total"] = {"n": tot["n"], "equal_share": share(tot["equal"], tot["n"]),
                             "lost_one_step_share": share(lost_n, tot["n"]),
                             "equal_share_where_step_lost": share(tot["equal_and_lost"], lost_n),
                             "equal_share_where_step_kept": share(tot["equal"] - tot["equal_and_lost"], kept_n)}
    out["ulp_edges"] = ULP_EDGES
    out["groups"] = table

    for g in ["matrices", "norms"] + sorted(matrices):
        print(f"\n[{g}] binade of |w|: n, share equal in both, share where one step of lr {lr:g} is lost, "
              f"unequal |a-b| / spacing <= 1 / <= 2 / > 2")
        for b, r in table[g].items():
            if b == "total":
                print(f"  total: n {r['n']}, equal {r['equal_share']:.4f}, step lost {r['lost_one_step_share']:.4f}, "
                      f"equal where lost {r['equal_share_where_step_lost']}, "
                      f"equal where kept {r['equal_share_where_step_kept']}")
                continue
            u = r["ulp_bucket_shares_of_unequal"]
            tail = "" if u[0] is None or b == "zero" else f", {u[0]:.3f} / {u[0] + u[1]:.3f} / {1 - u[0] - u[1]:.3f}"
            print(f"  {b:>8}: {r['n']:>11} {r['equal_share']:.4f} {r['lost_one_step_share']:.4f}{tail}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(to_jsonable(out), f, indent=1)
    print(f"\n[write] {args.out_json}")


if __name__ == "__main__":
    main()
