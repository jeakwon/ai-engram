"""Generations of ReCAST runs next to Init Fine. and Retain on the Model Utility splits.

Reads the anchors' generations (mu_submetrics_anchors.py output) and each run's generations.json
(recast_tofu.py) and writes a Markdown report: per split and model the mean ROUGE-L recall, the
mean length in words of generation and reference answer, and the items where the run loses the
most ROUGE against Init Fine., question, reference, Init and run side by side.
Positive control: every model saw the same prompts in the same order, or the script stops.

Usage (compute node; plain Python):
    python replication/scripts/compare_generations.py --anchors_json <file> \
        --run "label=<generations.json>" [--run ...] --splits real_authors world_facts --examples 6 --out_md <file>
"""

import argparse
import json
import statistics


def question(prompt):
    return prompt.rsplit("user\n\n", 1)[-1].rsplit("assistant", 1)[0].strip()


def words(text):
    return len(text.split())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--anchors_json", required=True)
    p.add_argument("--run", action="append", required=True, help="label=path to a run's generations.json")
    p.add_argument("--splits", nargs="+", required=True)
    p.add_argument("--examples", type=int, required=True)
    p.add_argument("--out_md", required=True)
    args = p.parse_args()

    with open(args.anchors_json) as f:
        anchors = json.load(f)["models"]
    models = {name: anchors[name]["generations"] for name in ("Init Fine.", "Retain")}
    for spec in args.run:
        label, _, path = spec.partition("=")
        with open(path) as f:
            models[label] = json.load(f)

    lines = [f"# Generations: {', '.join(models)}", ""]
    for split in args.splits:
        rows = {name: gens[split] for name, gens in models.items()}
        prompts = [r["prompt"] for r in rows["Init Fine."]]
        for name, rs in rows.items():
            if [r["prompt"] for r in rs] != prompts:
                raise SystemExit(f"{split}: {name} saw different prompts than Init Fine.")
        lines += [f"## {split} ({len(prompts)} items)", "",
                  "| Model | ROUGE-L recall | words generated | words reference | generated > 2x reference |",
                  "|---|---|---|---|---|"]
        for name, rs in rows.items():
            longer = sum(words(r["generation"]) > 2 * words(r["answer"]) for r in rs) / len(rs)
            lines.append(f"| {name} | {statistics.mean(r['rougeL_recall'] for r in rs):.4f} | "
                         f"{statistics.mean(words(r['generation']) for r in rs):.1f} | "
                         f"{statistics.mean(words(r['answer']) for r in rs):.1f} | {longer:.2f} |")
        lines.append("")
        for label in list(models)[2:]:
            drop = sorted(range(len(prompts)), key=lambda i: rows[label][i]["rougeL_recall"]
                          - rows["Init Fine."][i]["rougeL_recall"])[:args.examples]
            lines += [f"### {split}: largest ROUGE losses of {label} against Init Fine.", ""]
            for i in drop:
                init, run = rows["Init Fine."][i], rows[label][i]
                lines += [f"- **Q:** {question(prompts[i])}", f"  - reference: {init['answer']}",
                          f"  - Init Fine. ({init['rougeL_recall']:.2f}): {init['generation']}",
                          f"  - {label} ({run['rougeL_recall']:.2f}): {run['generation']}", ""]
    print("[control] every model saw the same prompts in the same order")
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
