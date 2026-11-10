# Cluster Environment Reference

This is an operational reference for the policy in the repository-root
`AGENTS.md`. Paths may change with cluster maintenance; verify a failing path
against the launchers before changing code or configuration.

## Locations

| Purpose | Path |
|---|---|
| Repository root | `/net/home/plgrid/plgkrzepk/GitHub/ai-engram` |
| Shared project storage | `/net/storage/pr3/plgrid/plggdnnp` |
| Miniforge, GH200 / `aarch64` | `/net/storage/pr3/plgrid/plggdnnp/apps/miniforge3-gh200` |
| Shared Conda environments | `/net/storage/pr3/plgrid/plggdnnp/conda_envs` |
| Project environment (GH200) | `/net/storage/pr3/plgrid/plggdnnp/conda_envs/ai-engram-gh200` |
| Hugging Face cache (`HF_HOME`) | `/net/storage/pr3/plgrid/plggdnnp/cache/plgkrzepk/huggingface` |
| Slurm and job logs | `/net/storage/pr3/plgrid/plggdnnp/archive/plgkrzepk/ai-engram_logs` (symlinked as `replication/logs`) |
| Experiment results | `/net/storage/pr3/plgrid/plggdnnp/experiment_results/ai-engram/` |
| Artifacts cited results rest on (e.g. collected statistics, model manifests) | `/net/storage/pr3/plgrid/plggdnnp/runtime_artifacts/ai-engram/` |

Repository-relative commands assume the repository root as the working
directory unless stated otherwise.

## Slurm Account

| Item | Value |
|---|---|
| Account | `plgreprunlearn-gpu-gh200` |
| Partition | `plgrid-gpu-gh200` (time limit 2 days) |
| Node architecture | `aarch64` (GH200); login nodes are `x86_64` |

## Python Environment

The project has its own environment, `ai-engram-gh200`. It is not shared
with `reload-llm-gh200` or the environments of other repositories.

- Spec: `replication/environment/ai_engram_gh200.yaml` (Conda: Python 3.11,
  the version of the upstream CI, and pip) and
  `replication/environment/requirements_gh200.txt` (pip pins, installed after
  torch).
- PyTorch: `torch==2.6.0` from `https://download.pytorch.org/whl/cu126`. It
  is the version already running on these nodes in `reload-llm-gh200`, and
  one of the two versions upstream verified bit-for-bit (`CHANGELOG.md`,
  0.8.0).
- ai-engram itself is installed with `pip install --no-deps -e .` from the
  repository root. The environment therefore runs the checked-out working
  tree: switching branches changes the code every later job runs.
- Build: `sbatch replication/scripts/bash/environment/build_env_gh200.sh`
  from the repository root. The job refuses to overwrite an existing prefix
  unless submitted with `--export=ALL,AI_ENGRAM_ENV_REPAIR=1`, which updates
  the prefix to the spec instead. It writes `pip freeze` into the prefix and
  ends with a CUDA and import smoke test.
- Use on an allocated GH200 node:

  ```bash
  ENV=/net/storage/pr3/plgrid/plggdnnp/conda_envs/ai-engram-gh200
  export LD_LIBRARY_PATH="$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  "$ENV/bin/python" ...
  ```

- The environment cannot run on a login node (`aarch64` binaries). Login-node
  checks are structural: `/usr/bin/python3.9 -m py_compile` and `bash -n`.

## Hugging Face Cache and Downloads

- Launchers export `HF_HOME` explicitly. `~/.bash_env` sets `XDG_CACHE_HOME`
  on storage, which `huggingface_hub` falls back to, but only when that file
  is sourced; never rely on it in a job.
- The cache is shared with `reload-unlearning`. Table 3 needs:
  - the dataset `locuslab/TOFU` (plain JSON files, no loading script). The
    cache holds the forget, retain, perturbed, real-authors and world-facts
    files; `full.json` (the reference set) and `holdout10.json` (the
    membership-inference holdout) are not cached yet;
  - the models `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` (Init Fine.)
    and `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90` (Retain);
  - the classifier `madhurjindal/autonlp-Gibberish-Detector-492513457`,
    loaded by the forget-fluency metric in
    `engram.benchmarks._tofu_official.gibberish_score`.
- Downloads require explicit user authorization. Jobs after the download run
  with `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`.

## API Keys

- W&B: `~/.bash_env` sets `WANDB_ENTITY`. Upstream ai-engram does not log to
  W&B; replication launchers set `WANDB_PROJECT` explicitly and export
  `WANDB_DIR` to the logs directory on storage, so run files land in
  `archive/plgkrzepk/ai-engram_logs/wandb/`.
- Hugging Face: no token is needed for public repositories. A token, when one
  is needed, is read from the environment only.

## Helios SSH Fallback

Copied from the `reload-unlearning` repository guide (originally from
`unlearning-svd`); binding with the same authority as `AGENTS.md`.

- Try the configured `helios` SSH alias first.
- If the alias fails with a DNS resolution error for
  `login01.helios.cyfronet.pl`, resolve the current IPv4 address with
  `dig +short login01.helios.cyfronet.pl A`, verify that the result is
  non-empty, and connect with the resolved address while preserving the
  existing host identity:

  ```bash
  HELIOS_IPV4="$(dig +short login01.helios.cyfronet.pl A | tail -n 1)"
  test -n "${HELIOS_IPV4}"
  ssh -o HostName="${HELIOS_IPV4}" \
      -o HostKeyAlias=login01.helios.cyfronet.pl \
      helios
  ```

- `HostKeyAlias=login01.helios.cyfronet.pl` makes SSH use the existing
  `known_hosts` identity for the login host when connecting by address.
  Never disable host-key verification or accept an unverified replacement
  key as a DNS workaround.
- When a direct IPv4 connection is used, report the resolved repository root
  and active branch before any remote edit or experiment launch, exactly as
  for a normal SSH session.

## `$HOME` Limits

Binding with the same authority as `AGENTS.md`.

- `$HOME` is capped at 10 GiB **and 100k files**, and the file cap is the one
  that bites first. Slurm refuses submissions with a bare
  "Batch job submission failed: Unspecified error" once either is near its
  limit; `hpc-fs` prints the real reason and its file column updates lazily.
- Environments, caches, model checkpoints, logs and results therefore never
  live in `$HOME`; they go to the storage paths above.
- Before moving a log tree, check `squeue` is empty. Moving a directory a
  running job is reading or writing breaks that job.
