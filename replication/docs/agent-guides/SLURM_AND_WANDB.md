# Slurm and W&B Reference

The repository-root `AGENTS.md` defines authorization and monitoring policy.
This document is the operational checklist for experiments.

## Submission Check

Before submission, derive the time limit from evidence rather than a generic
default:

1. inspect `Elapsed`, `Timelimit`, and terminal state for completed jobs that
   match the model, forget split, workflow, and approximate task size
   (`sacct -u $USER -o JobID,JobName,State,Elapsed,Timelimit`);
2. confirm in their logs whether the recorded elapsed time includes the same
   phases (for example model download plus statistics collection plus
   evaluation);
3. scale the estimate for changed models, splits, conditions or sample counts;
4. add a modest safety margin and record both the estimate and chosen limit in
   the launch summary.

When no materially comparable job exists, first use the smallest representative
pilot that can measure the whole path. A conservative initial estimate is
acceptable only when its assumptions are stated. Avoid hour-scale limits for
jobs supported by evidence to finish in a few minutes. Upstream documents the
full TOFU evaluation of the anchors plus two Engram conditions as ~24 min on
one A100 (`docs/tofu.md`); that is a starting assumption, not GH200 evidence.

After submitting a job or array, record and verify once:

- job ID and array range/throttle,
- command and resolved arguments,
- partition, account, GPUs, CPUs, memory, and time limit,
- dependency expression,
- current state,
- stdout/stderr and durable output paths.

A `PENDING` job has not validated runtime imports, checkpoint loading, dataset
access, CUDA, or W&B. Report those items as not yet verified rather than
polling.

For a job already running during the one-time check, inspect the initial task
log for immediate setup failures and W&B initialization. Short blocking
validation jobs may be followed until completion when their result is required
before a dependent submission.

## W&B

Long-running experiments use online W&B with an explicit project. Upstream
ai-engram does not log to W&B; replication launchers set `WANDB_PROJECT`
explicitly (`ai-engram-replication`), and `WANDB_ENTITY` comes from
`~/.bash_env`. Expected exceptions are tests, dry-runs, explicitly
local/offline workflows, and an explicit user request to disable logging.

Tags encode every axis needed to distinguish runs:

- model and forget split,
- Engram condition (`alpha_uniform`, `alpha_wnorm`) or anchor (`init_fine`,
  `retain`) and campaign type,
- alpha and the alpha_W-Norm exponent,
- edited layers,
- reference-set size,
- seed.

Use stable filesystem-safe values such as `alpha_0p6`. A flag alone is not
evidence that online logging works; confirmation comes from the started task
log and its project/run URL.

## Campaign Types

Binding with the same authority as `AGENTS.md`.

- **`method-faithful`** -- the replication: paper settings (Appendix E.3),
  upstream code, and every deviation recorded in `replication/docs/WORK_LOG.md`
  as `Deviation:`. Campaign tag: `engram_tofu_table3_faithful`.
- **Matched** -- any comparison where methods share one protocol (same model,
  split, reference set, evaluation code). Only explicit method-specific
  mechanisms and declared sweep axes may differ. It gets its own campaign tag
  and protocol name. The ReCAST port on TOFU is one: campaign tag
  `recast_tofu_comparison`, same model, splits, anchors and evaluation code as
  Table 3, phase two on the OpenUnlearning TOFU trainer recipe; its sweep axes
  are the variant and the shrink floor.

Never place results of both types in one table without clearly marking the
protocol mismatch. Before submission, report both the shared settings and
every method-specific deviation.

## Time Limits

Binding with the same authority as `AGENTS.md`.

- Choose every Slurm time limit adaptively before submission. Inspect elapsed
  times of completed, materially comparable jobs (prefer `sacct` plus their
  logs), estimate the requested job's work from model, split, phases and task
  count, then add a modest explicit safety margin. Record the evidence,
  estimate, and selected limit in the launch report. Do not use a generic
  one-hour limit for a job expected to take only minutes. If no comparable run
  exists, submit the smallest representative pilot first or state the
  conservative assumption used.
