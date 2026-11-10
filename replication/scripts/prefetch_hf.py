"""Download the Hugging Face models and dataset configs of a run into HF_HOME and record their revisions.

Usage (on a GH200 node, inside the ai-engram-gh200 environment):
    python replication/scripts/prefetch_hf.py --model_ids org/model [...] \
        --dataset_id locuslab/TOFU --dataset_configs full holdout10 [...] --manifest out.json
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python replication/scripts/prefetch_hf.py --check_offline \
        --model_ids org/model [...] --dataset_id locuslab/TOFU --dataset_configs full holdout10 [...]

The second form downloads nothing: it loads every config and tokenizer and every dataset
config from the cache alone, the way the evaluation will.
"""

import argparse
import json
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Prefetch Hugging Face models and dataset configs")
    parser.add_argument("--model_ids", nargs="+", required=True, help="Model repositories")
    parser.add_argument("--allow_patterns", nargs="+", required=True, help="Model files to download")
    parser.add_argument("--ignore_patterns", nargs="*", default=[], help="Model files to skip")
    parser.add_argument("--dataset_id", type=str, required=True, help="Dataset repository")
    parser.add_argument("--dataset_configs", nargs="+", required=True, help="Dataset configs to load")
    parser.add_argument("--max_workers", type=int, default=8, help="Parallel file downloads")
    parser.add_argument("--manifest", type=str, help="JSON file to write repo -> revision (download mode)")
    parser.add_argument("--check_offline", action="store_true", help="Load everything from the cache only")
    return parser.parse_args()


def tree(path):
    files = []
    for root, _, names in os.walk(path):
        files += [os.path.relpath(os.path.join(root, n), path) for n in names]
    return sorted(files)


def size_gb(path, files):
    # Snapshot entries are symlinks into blobs/; getsize follows them.
    return sum(os.path.getsize(os.path.join(path, f)) for f in files) / 1e9


def download(args):
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    manifest = {"models": {}, "dataset": {}}
    for repo_id in args.model_ids:
        start = time.time()
        path = snapshot_download(repo_id=repo_id, allow_patterns=args.allow_patterns,
                                 ignore_patterns=args.ignore_patterns or None, max_workers=args.max_workers)
        files = tree(path)
        gb = size_gb(path, files)
        print(f"[prefetch] {repo_id} revision={os.path.basename(path)} files={len(files)} "
              f"size_gb={gb:.2f} seconds={time.time() - start:.0f}")
        manifest["models"][repo_id] = {"revision": os.path.basename(path), "path": path,
                                       "files": files, "size_gb": round(gb, 3)}

    start = time.time()
    path = snapshot_download(repo_id=args.dataset_id, repo_type="dataset",
                             allow_patterns=["README.md"] + [f"{c}.json" for c in args.dataset_configs])
    print(f"[prefetch] {args.dataset_id} revision={os.path.basename(path)} files={tree(path)} "
          f"seconds={time.time() - start:.0f}")
    rows = {}
    for config in args.dataset_configs:
        rows[config] = load_dataset(args.dataset_id, config)["train"].num_rows
        print(f"[prefetch] {args.dataset_id}/{config} rows={rows[config]}")
    manifest["dataset"] = {"id": args.dataset_id, "revision": os.path.basename(path), "rows": rows}

    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[prefetch] manifest written to {args.manifest}")


def check_offline(args):
    for var in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(var) != "1":
            raise SystemExit(f"--check_offline needs {var}=1 in the environment")
    from datasets import load_dataset
    from transformers import AutoConfig, AutoTokenizer

    for repo_id in args.model_ids:
        config = AutoConfig.from_pretrained(repo_id)
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        print(f"[offline] {repo_id} config={type(config).__name__} "
              f"architectures={config.architectures} tokenizer={type(tokenizer).__name__}")
    for config in args.dataset_configs:
        rows = load_dataset(args.dataset_id, config)["train"].num_rows
        print(f"[offline] {args.dataset_id}/{config} rows={rows}")
    print("[offline] every model config, tokenizer and dataset config loaded from the cache")


def main():
    args = parse_args()
    print(f"[prefetch] HF_HOME={os.environ.get('HF_HOME')} check_offline={args.check_offline}")
    if args.check_offline:
        check_offline(args)
    else:
        if not args.manifest:
            raise SystemExit("--manifest is required in download mode")
        download(args)


if __name__ == "__main__":
    main()
