# Workflow Reference

Rules copied from the `reload-unlearning` repository guide (originally from
`unlearning-svd`) and adapted to this repository. They are binding with the
same authority as `AGENTS.md`.

## Branch and worktree lifecycle

When implementing functionality on a new branch/worktree:

- Maintain a concise, current checklist as open `[ ]` items in
  `replication/docs/WORK_LOG.md`. At each substantial milestone and in every
  implementation/status handoff, state the active branch/worktree, what is
  complete, what remains (including blockers or decisions), and the next action.
- Update the checklist when scope changes or an audit reveals new work; do not
  make the user reconstruct outstanding tasks from conversation history.
- Distinguish implementation, validation, integration, commits/publication, and
  source-branch closure; a finished job or passing test alone is not closure.
- Close the work explicitly only after accounting for remaining source changes,
  configuration, documentation, tests, and artifacts. List any deferred research
  or residual risks separately from completion requirements, with the reason
  for deferral. Do not imply that integration is the final step while known
  required work remains. This reporting rule does not authorize extra jobs,
  branch switches, commits, pushes, or deletion of worktrees/artifacts.

## Naming

- Proactively flag misleading names of flags, functions, metrics and persisted
  fields even outside naming reviews. Check their actual scope, propose an
  accurate name and discuss compatibility before public/persisted renames.

## `apply_patch` fallback

- If the filesystem `apply_patch` helper fails specifically with `bwrap` and
  `ENOSPC`, first verify disk/inodes and `/proc/sys/user/max_user_namespaces`. If
  disk/inodes are healthy and user namespaces are disabled, do not repeatedly
  retry the helper. Build an explicit unified diff in the session scratch
  directory, apply it with `git apply` (or POSIX `patch` for an untracked file),
  restrict it to named workspace files, then inspect the focused diff and run
  `git diff --check`. This is a patch-only fallback, not permission for
  arbitrary scripted rewrites.

## Audit levels

- Use `lightweight` for a narrow, low-risk review: inspect the focused diff,
  nearby call sites, configuration and documentation; run the smallest relevant
  regression checks and `git diff --check`; report findings and skipped gates.
- Use `scientific/release` for scientific code, experiment protocols, persisted
  schemas, release-facing behavior, or whenever the user asks for a full audit.
  In addition to the lightweight checks, build requirement-to-implementation-to-
  test traceability; independently verify formulas, estimators, split/sample
  alignment, leakage, numerical edge cases, reference identity, provenance,
  determinism, backward compatibility, and resource/performance risks. Run
  targeted adversarial/property and synthetic integration checks. Compute/GPU or
  Slurm gates still require their normal explicit authorization.
- If the user does not choose a level, default to `scientific/release` for
  scientific or experiment-affecting code and to `lightweight` otherwise. State
  the selected level at the start of the audit.

## Audit requests

- Inspect implementation, configuration, documentation, public interfaces,
  call sites, persisted schemas, and tests. For scientific code, verify the
  mathematical definition, estimators, data alignment, reference selection,
  leakage risks, and reproducibility metadata independently of test status.
- For this repository, also compare the implementation with the paper
  (arXiv 2606.14997) and list every disagreement with page or equation.
- Distinguish observed behavior, behavior covered by tests, and assumptions
  that could not be verified. Existing passing tests are evidence of regression
  stability, not proof of scientific or functional correctness.
- Run the smallest relevant permitted regression checks plus targeted boundary,
  adversarial, or property-style checks for likely failure modes. Report exact
  commands, exit status, reviewed results, and every skipped gate with its
  reason; never broaden compute or Slurm authorization implicitly.
- Present findings by severity with file-and-line evidence, user-visible impact,
  and a concrete remediation. Also identify unnecessary complexity,
  duplication, dead paths, API/CLI gaps, stale documentation, and opportunities
  to reuse existing abstractions.
- End with a concise verdict, residual risks, and an ordered remediation plan.
  Preserve unrelated working-tree and index changes throughout the audit.
