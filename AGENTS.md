# AGENTS Instructions for ai-engram (replication)

## Scope and Sources of Truth

- This is the only `AGENTS.md` in the repository. Do not add nested instruction
  files; keep repository-wide agent policy here. `CLAUDE.md` at the repository
  root only imports this file for Claude Code; put no policy there.
- The repository is a clone of the upstream `jeakwon/ai-engram`, the reference
  implementation of "AI Engram: In Search of Memory Traces in Artificial
  Intelligence" (Kwon et al., ICML 2026, arXiv 2606.14997). It is used for a
  replication of the paper's language-model experiment, **Table 3**:
  Llama-3.2-1B-Instruct unlearning on TOFU `forget10` (setup in Appendix
  E.1-E.3, metric definitions in Appendix E.2; Table 6 reports the Overall
  score of the same runs).
- Replication scope (user decision, 2026-10-21): the rows Init Fine. (the
  TOFU fine-tuned model before unlearning), Retain (the gold model trained
  without the forget set), Engram (alpha = 0.6) and Engram (alpha_W-Norm), in
  the columns Memorization, Utility, Privacy, Exact Memorization and log10
  Forget Quality. The eight trained baselines (GradDiff, IdkNLL, UNDIAL, RMU,
  AltPO, NPO, SimNPO, IdkDPO) are not replicated; where a report shows them,
  their numbers are quoted from the paper and marked as not replicated.
- Second campaign (user decision, 2026-10-22): **ReCAST on the same TOFU
  setup**, a port of the ReCAST method of the user's `unlearning-svd`
  repository (`SEMU/Classification`, unrestricted and restricted with the SEMU
  adapter backend) to Llama-3.2-1B-Instruct. Its specification is
  `SEMU/Classification/docs/RECAST_PORTING_CONTRACT.md` in that repository,
  with the Classification code as the behavioral reference. Campaign tag
  `recast_tofu_comparison`; its rows are scored against the Table 3 anchors
  but reported separately from the method-faithful replication, never merged
  into Table 3 without explicit stratification.
- `origin` is the author's repository. Never push to it.
- When the user asks to create or switch to a branch and/or worktree, treat that
  branch-worktree pair as the persistent active workspace. Before subsequent
  edits, verify that the current directory and checked-out branch still match it.
  Continue working there until the user explicitly names a different branch or
  worktree; do not infer a switch from adjacent discussion, audits, or port plans.
- When implementing functionality on a new branch/worktree, guide the user
  continuously through its lifecycle until the implementation is explicitly
  closed. The checklist and handoff requirements are in `WORKFLOW.md`.
- Reference guides provide operational facts and commands, but this file is
  authoritative when a conflict exists. Guide sections marked as binding carry
  the same authority as this file.
- Read the relevant guide before acting (all under `replication/docs/agent-guides/`):

| Task | Reference |
|---|---|
| Cluster paths, Python environment, Hugging Face cache, API keys, SSH fallback, where logs and results live | `CLUSTER_ENVIRONMENT.md` |
| Slurm submission, time limits, W&B, campaign types | `SLURM_AND_WANDB.md` |
| Branch lifecycle, naming, audits, patch fallback | `WORKFLOW.md` |
| The incidents behind the configuration, destructive-operation and ablation rules | `RULE_RATIONALE.md` |

- Try the configured `helios` SSH alias first. If it fails on DNS, use the IPv4
  fallback in `CLUSTER_ENVIRONMENT.md`. Never disable host-key verification or
  accept an unverified replacement key as a DNS workaround.

## Repository-Wide Safety

- Prefer minimal changes. Preserve scientific protocols, checkpoint
  compatibility, naming, and existing abstractions.
- Preserve unrelated user changes in both the working tree and index.
- Never commit credentials, tokens, datasets, checkpoints, logs, generated
  reports, caches, experiment outputs, backups, notebooks, or debug dumps.
- Never write an API key or token (W&B, Hugging Face) into source, configs,
  launchers or logs; read it from the environment.
- Do not delete datasets, checkpoints, logs, reports, experiment outputs, or
  cached artifacts unless explicitly requested.
- Do not launch training, large Slurm arrays, dataset or model downloads, or
  GPU test gates unless explicitly requested.
- Any call to an external hosted API other than Hugging Face downloads and
  W&B logging needs explicit user authorization.
- Do not modify shell startup files, environment files, or cluster configuration
  unless explicitly requested.
- Replication files live under `replication/`; do not add loose files at the
  repository root or into upstream directories.
- In user-facing reports and scientific documentation, spell out every method,
  metric, protocol, split, and training-component name on first use before using
  an acronym or short identifier. Internal filesystem-safe identifiers may stay
  concise, but their displayed meaning must be unambiguous.
- When encountering an incomplete, misleading, inconsistent, or incorrect name,
  explicitly tell the user, explain the ambiguity, propose the corrected full
  name, and ask whether persisted/public naming should be changed when the fix is
  not already unambiguous and behavior-preserving.
- Final evaluation reports the same metrics for the Original model (Init Fine.,
  before unlearning), the Retain reference (trained without the forget set) and
  every unlearned model, computed by the same evaluation code.
- Do not copy instructions or conventions from unrelated repositories.
- Do not create new Markdown documentation files unless the user asks for them;
  update the existing ones instead.
- Communicate with the user in Polish. Code, comments, docstrings and these
  documents stay in English.

## Upstream Fidelity

- This repository is a **method-faithful replication**. The upstream package
  and its files -- `src/engram/`, `tests/`, `examples/`, `docs/` (the upstream
  MkDocs site), `mkdocs.yml`, `pyproject.toml`, `README.md`, `CHANGELOG.md`,
  `.github/` -- stay unmodified. Replication code calls the package; it does
  not copy or patch it.
- Every deviation from the paper or from upstream code -- a substituted model,
  a package version, a fixed upstream bug, a changed default -- is recorded in
  `replication/docs/WORK_LOG.md` as an entry starting with `Deviation:`, with
  the reason. `grep -n 'Deviation:' replication/docs/WORK_LOG.md` lists them.
- Where the paper and the code disagree, report both and ask which one to
  follow; do not silently pick one.
- Do not silently fix upstream bugs. A fix changes what is being replicated:
  propose it, and record it as a deviation once the user agrees.
- The ReCAST port follows the porting contract and the Classification code;
  each language-model adaptation and each phase-two choice that differs from
  the canonical Classification protocol is a `Deviation:` entry. A port is
  called ReCAST only while the contract's parity tests
  (`replication/tests/test_recast_core.py`) pass.
- The upstream `tofu` extra in `pyproject.toml` omits `scikit-learn`, which
  `engram.benchmarks._tofu_official` imports. Install the environment from the
  spec in `replication/environment/`, never from the upstream extras alone.

## Replication Layout and Wiring

- New files go to: Conda and pip specs in `replication/environment/`, Slurm
  launchers in `replication/scripts/bash/<workflow>/`, Python commands in
  `replication/scripts/`, agent guides and the work log in `replication/docs/`.
  `replication/logs` is a symlink to storage (`CLUSTER_ENVIRONMENT.md`).
  The ReCAST port is the library `replication/scripts/recast_core.py`, the
  command `replication/scripts/recast_tofu.py` and the tests
  `replication/tests/test_recast_core.py`. Its extensions beyond the porting
  contract (their modules, flags, launchers, tests and results) are described
  in `replication/docs/RECAST_EXTENSIONS.md`; keep it current when an
  extension is added or changed.
- The package is installed into the environment editable from this checkout
  without dependencies, so a job runs the checked-out code with the pinned
  dependencies. Every run records the git commit and branch it ran.
- The upstream reproduction of these numbers is `tests/test_tofu_evaluate.py`
  (with `tests/_tofu_evaluate.py`) and `engram.benchmarks.tofu`. Replication
  scripts reuse that code path instead of reimplementing metrics.

## Scientific Invariants

- Anchors: Init Fine. is `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`,
  Retain is `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90`. Splits come
  from `locuslab/TOFU` and stay paired: forget `forget10`, retain `retain90`,
  holdout `holdout10` (for the membership-inference attacks).
- Metrics are the raw OpenUnlearning TOFU metrics as ported in
  `engram.benchmarks._tofu_official`, rescaled against Init Fine. and Retain,
  both evaluated in the same run by the same code. Memorization, Utility and
  Privacy are harmonic means of their sub-metrics; Overall is the harmonic mean
  of the three (Appendix E.2). Exact Memorization is
  `|E(model) - E(Retain)| / |E(Init Fine.) - E(Retain)|` clipped to [0, 1].
  Forget Quality is reported as log10 of the Kolmogorov-Smirnov test p-value
  between the truth-ratio distributions of the evaluated model and Retain on
  the forget split.
- Engram conditions (Appendix E.3): alpha = 0.6 with uniform per-layer
  strength, and alpha_W-Norm with the per-layer strength set by the ratio
  `||W+|| / ||W||` rescaled by its maximum over layers. Their implementation,
  covariance sets, answer-token masking and edited layers follow
  `tests/test_tofu_evaluate.py`; every value is stated explicitly in the launch.
- Every random step (for example the shuffle that selects the reference set)
  uses an explicit seed recorded with the run.

## Experiment Configuration

The incidents behind these rules are in `RULE_RATIONALE.md`.

- Launch commands and generated run configs must state every value that defines
  the experiment explicitly. Never rely on an argparse or function default to
  supply one.
- When a generator derives configs from a template, it inherits the template's
  omissions. Adding a key to the template is the fix; adding it only to the
  derived set is not.
- A value that the runtime overrides unconditionally must not be written into
  the config as though it were in force. Record the override in the work log
  instead.
- Parsed arguments (W&B config, argument dumps) record what was requested, not
  what executed. Before stating what a run executed, or copying a finished
  run's setting into a new config, cite the run's own log lines.
- A finding that contradicts something the user stated or that is already
  recorded (these rules, memory, docs) is reported as unverified until runtime
  evidence confirms it.
- Never add a silent runtime override. Any value the code sets on `args` at
  runtime must also be printed to stdout when applied.

## Destructive Operations

The incidents behind these rules are in `RULE_RATIONALE.md`.

- **Print the list before acting.** Never compute a deletion list and delete in
  the same command. Show what will go and what will stay, with a reason per
  entry.
- **Establish liveness from `/proc`, not from `ps`.** Check `/proc/*/exe`,
  `/proc/*/fd` and `/proc/*/cwd`.
- **Compute your own ancestor chain and assert no overlap** with the targets,
  aborting on a match. One loop over `/proc/$p/ppid`.
- **Prefer `mv` to an archive over `rm`.** Moves are reversible. Delete only an
  empty directory or an application's own trash.
- **A move is as destructive as a delete for a running job.** Before touching a
  directory, check `squeue`, and check whether any queued or running job reads
  or writes that path.
- **Ask before deleting anything in `$HOME` outside the scratch directory**, no
  exception for "it is only a cache".

## Verifying an Ablation

The incidents behind these rules are in `RULE_RATIONALE.md`.

- **A null result is suspect until the manipulation is proven to have
  happened.** "No difference" has two causes -- the effect is absent, or the
  change never took effect -- and the second is the common one. Rule out the
  second first.
- **Demand evidence from the runtime, not the config.** A config records a
  request.
- **Check for overrides, not just the value.** Mechanically:
  `grep -n "args\.<key> *="` over the runtime path.
- **Every ablation needs a positive control, checked before the result.** Give
  the manipulation an observable signature and read it first.
- **The control goes to stdout, not only to W&B.**
- **After a second failure of the same thing, read the contract before
  patching again.** Two consecutive fixes to one file without reading the code
  that validates it is forbidden.

## Where Logs, Results and Generated Artifacts Live

- Slurm logs, experiment results, collected covariance statistics and Hugging
  Face caches live on project storage; locations are in
  `CLUSTER_ENVIRONMENT.md`. A repository path that holds them is a symlink to
  storage. Never restore a symlinked tree into `$HOME`.
- Experiment results never go into git: tables, data JSON, figures, reports,
  statistics files and audit outputs.
- What stays in git: code, launchers, environment specs, protocol and procedure
  documents, and the index documents that point at results.
- Never add a broad `.gitignore` pattern for a directory that mixes results
  with procedure documents. Symlinks to storage are registered in
  `.git/info/exclude`, since the upstream `.gitignore` stays unmodified.
- Before moving a log tree, check `squeue` is empty.

## Cluster Execution and Validation

- Login nodes are for orchestration, inspection, configuration, syntax checks,
  and lightweight pure-logic tests only.
- Never run model construction, non-trivial tensor operations, dataset access,
  DataLoader iteration, statistics, transformations, or heavy tests on a login
  node. Use an allocated compute node.
- Never run anything that imports PyTorch, transformers or datasets on a login
  node; this includes the upstream unit tests under `tests/`. The project
  environment is `aarch64` and runs only on GH200 nodes. Login-node checks are
  structural only: `/usr/bin/python3.9 -m py_compile <file>.py` and
  `bash -n <launcher>.sh` (the system `python3` is 3.6). Do not present a
  structural check as a numerical or integration test.
- Submitting any Slurm job, including validation and smoke tests, requires
  explicit user authorization.
- Behavior-affecting changes are non-trivial and require targeted validation.
  Bug fixes should include a regression check when practical. Docs, comments,
  formatting, and behavior-preserving renames are trivial.
- Never claim a test or gate passed unless its output and exit status were
  reviewed. A successful `sbatch` proves only submission. Report skipped
  validation and its reason.
- Do not hardcode host-specific dataset, model or cache paths in runtime code
  or portable configs; use configuration and environment variables. Launchers
  may set cluster defaults.

## Experiments

- Long-running experiments require explicit user authorization and online W&B
  logging with an explicit project, except for tests, dry-runs, explicitly
  local/offline work, or an explicit user request to disable logging.
- The replication is one `method-faithful` campaign: paper settings and
  upstream code, with every deviation recorded. A re-tuned or matched-protocol
  comparison is a separate campaign with its own tag; never combine both in
  one table without explicit stratification.
- Use the paper's Engram settings (Appendix E.3). Any re-tuning, such as an
  alpha sweep, is a separate, declared campaign; never tune on the metrics that
  are then reported as the replication result.
- Sweep tags must cover every distinguishing axis, including model, forget
  split, Engram condition, alpha, the alpha_W-Norm exponent, edited layers,
  reference-set size and seed. Use stable filesystem-safe values.
- Every run of one campaign also carries one identical, filesystem-safe W&B
  tag that identifies the campaign as a whole. Keep it stable across
  conditions, anchor evaluations, evaluation-only jobs and follow-up metric
  jobs.
- Choose every Slurm time limit adaptively before submission. Record the
  evidence, estimate, and selected limit in the launch report; the procedure is
  in `SLURM_AND_WANDB.md`.
- After submission, perform one validation of job ID, arguments, state, time
  limit, dependency, resources, and log path, then yield. Do not poll unless the
  user requests monitoring or a blocking short validation is required.
- If a task is already running, inspect its initial log and verify W&B
  initialization. If it is pending, report W&B as not yet verifiable.

## Audits

- Treat an audit or review request as read-only unless the user also asks for
  fixes. Do not silently modify the audited implementation.
- State the selected audit level at the start: `lightweight` or
  `scientific/release`, as defined in `WORKFLOW.md`, which also lists what an
  audit must inspect and report.

## Git, Documentation, and Reporting

- Before remote edits, inspect the target and Git status. Apply a narrow patch,
  then inspect the focused diff and run `git diff --check`.
- Keep commits focused. When diagnosing staging or commit issues, inspect both
  `git diff` and `git diff --cached`; review and re-stage pre-commit autofixes.
- Append to `replication/docs/WORK_LOG.md` only after substantial
  implementation, experiment submission, or completed scientific analysis. Do
  not update it for planning, review, inconclusive diagnostics, or trivial
  work. Granularity:
  - One entry per experiment, updated in place: add `Result (HH:MM): ...` and
    flip `[ ]` to `[x]` instead of writing a new "done" entry. A result on a
    later day: flip the old box, end its first line with `-> result DD HH:MM`,
    and write the result at the end of the log.
  - A separate entry only for: an experiment start (design, set, jobs); a code
    or tool change with its gate result or the skipped gate; a user decision,
    quoted; a deviation (`Deviation:`); a correction that touches results,
    data, W&B or an earlier claim (mark the corrected entry
    `SUPERSEDED -> DD HH:MM: ...`); a completed analysis with numbers; a
    deletion (what, whose consent, `/proc`).
  - Never a separate entry: the next stage of a running experiment, a resubmit,
    a cancel of a job that never started, a time limit, a gate submission,
    "COMPLETED"/"running", a dry run, "Next:"/TODO, planning, `$HOME` cleanup.
    At most a clause in the experiment's entry.
  - Always keep: the job IDs that produced reported numbers; the IDs of jobs
    that started and left files (logs, outputs, W&B), even when cancelled;
    corrections; quoted decisions; provenance; gate results; open follow-ups.
  - `[ ]` means still open, so `grep -n '^- \[ \]'` lists open work. One
    `## YYYY-MM-DD` heading per day.
- Numbers cited in `replication/docs/WORK_LOG.md` or in reports come only from
  scripts in `replication/scripts/`, saved there before the numbers are cited.
  A session scratchpad holds one-off work only; data a cited result rests on
  goes to storage
  (`/net/storage/pr3/plgrid/plggdnnp/runtime_artifacts/ai-engram/...`), never
  only to a scratchpad.
- For non-trivial work, report the outcome, changed files, validation results,
  skipped gates, and unresolved limitations. Add requirement-to-test
  traceability for multi-part or high-risk changes.
