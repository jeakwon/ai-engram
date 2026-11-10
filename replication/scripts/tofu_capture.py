"""Keep what engram.benchmarks._tofu_official computes and discards: the nine Model Utility
sub-metrics and the greedy generations behind each ROUGE score.

``TofuCapture`` patches three functions of the module for the duration of a ``with`` block and
calls the originals unchanged, so no metric changes. ROUGE calls are labelled by their order:
``compute_full_tofu`` scores forget, then (inside ``compute_model_utility``) retain, real
authors and world facts. ``check`` is the positive control: one Model Utility call whose
harmonic mean is the reported value, and one ROUGE call per split whose per-item mean equals
the reported ROUGE.
"""

ROUGE_KEYS = {"retain": "retain_Q_A_ROUGE", "real_authors": "ra_Q_A_ROUGE", "world_facts": "wf_Q_A_ROUGE"}


class TofuCapture:
    def __init__(self, T, rouge_splits):
        self.T, self.rouge_splits = T, list(rouge_splits)
        self.mu_subs, self.generations = [], []

    def __enter__(self):
        T, orig = self.T, {n: getattr(self.T, n) for n in ("compute_model_utility", "metric_rouge",
                                                            "eval_text_similarity")}
        self._orig = orig

        def compute_model_utility(*a, **k):
            mu, subs = orig["compute_model_utility"](*a, **k)
            self.mu_subs.append(subs)
            return mu, subs

        def metric_rouge(*a, **k):
            self.generations.append([])
            return orig["metric_rouge"](*a, **k)

        def eval_text_similarity(model, tokenizer, batch, generation_args):
            outs, generate = [], model.generate

            def recording_generate(*ga, **gk):
                out = generate(*ga, **gk)
                outs.append(out)
                return out

            model.generate = recording_generate
            try:
                scores = orig["eval_text_similarity"](model, tokenizer, batch, generation_args)
            finally:
                del model.generate
            # Decoded as in eval_text_similarity.
            ids, labels = batch["input_ids"], batch["labels"]
            prompts = tokenizer.batch_decode(ids, skip_special_tokens=True)
            full = tokenizer.batch_decode([l[l != T.IGNORE_INDEX] for l in labels], skip_special_tokens=True)
            gens = tokenizer.batch_decode(outs[0][:, ids.shape[-1]:], skip_special_tokens=True)
            eos = tokenizer.decode([tokenizer.eos_token_id])
            for p, f, g, s in zip(prompts, full, gens, scores):
                if eos and eos in g:
                    g = g.split(eos)[0]
                self.generations[-1].append({"prompt": p, "answer": f.replace(p, "").strip(),
                                             "generation": g.strip(), "rougeL_recall": s["rougeL_recall"]})
            return scores

        T.compute_model_utility, T.metric_rouge, T.eval_text_similarity = (
            compute_model_utility, metric_rouge, eval_text_similarity)
        return self

    def __exit__(self, *exc):
        for name, fn in self._orig.items():
            setattr(self.T, name, fn)
        return False

    def check(self, model_utility, forget_rouge=None):
        """Positive control; returns (nine sub-metrics, generations by split)."""
        import numpy as np
        import scipy.stats

        if len(self.mu_subs) != 1:
            raise RuntimeError(f"expected one Model Utility call, got {len(self.mu_subs)}")
        subs = self.mu_subs[0]
        if abs(float(scipy.stats.hmean(list(subs.values()))) - model_utility) > 1e-12:
            raise RuntimeError("captured sub-metrics do not give the reported Model Utility")
        if len(self.generations) != len(self.rouge_splits):
            raise RuntimeError(f"expected ROUGE calls {self.rouge_splits}, got {len(self.generations)}")
        reported = {"forget": forget_rouge, **{s: subs[k] for s, k in ROUGE_KEYS.items()}}
        for split, rows in zip(self.rouge_splits, self.generations):
            mean = float(np.mean([r["rougeL_recall"] for r in rows]))
            if abs(mean - reported[split]) > 1e-9:
                raise RuntimeError(f"{split}: captured ROUGE mean {mean} differs from reported {reported[split]}")
        return subs, dict(zip(self.rouge_splits, self.generations))
