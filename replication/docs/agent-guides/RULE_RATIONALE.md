# Why the Rules Exist

The incidents behind the configuration, destructive-operation and ablation
rules in the repository-root `AGENTS.md`. They happened in the sibling
repository `unlearning-svd` (same user, same cluster) and are condensed from
its guide. The rules themselves stay in `AGENTS.md`; these passages are
binding with the same authority.

## Experiment configuration

- A default is a property of the code at the moment of the run, so a config
  that omits a key silently changes meaning whenever that default is edited,
  and the artifacts it produced stop being reproducible.
- This is not hypothetical: a whole campaign of configs omitted one key, and
  reading the effective value back took a source-level audit plus dedicated
  probe runs.
- The runtime overwrote that key for every run regardless of what was passed,
  so the configs recorded something that never ran. On 2026-09-30 the
  argument dump of finished runs (the parser default of their date) was
  reported to the user as what ran, and a control set was generated pinning
  it -- a bit-identical duplicate with a false config value and W&B tag,
  although every run's own log line stated the executed value.
- Hence the rules: state every value, never record an overridden value as in
  force, print every runtime override to stdout, and cite runtime log lines
  for what executed.
- Relevance here: upstream functions in this repository carry defaults that
  define the experiment. `engram.benchmarks.tofu.run` defaults the scaling
  condition (`scale="adaptive"`) and the reference-set size
  (`n_total=4000`), `load_splits` the shuffle seed, `search` the alpha grid,
  and `EditorConfig` the storage device and bias absorption; a launch that
  leaves them implicit does not record what ran.

## Destructive operations

Every rule here exists because it was broken on 2026-09-23, in one session.

- Print the list before acting. A `grep` in the "keep this one" branch matched
  an extra line, the condition never fired, and `rm -rf` took the running
  VS Code server with it.
- Liveness from `/proc`. `ps` shows a command line, which says nothing about
  what a process currently holds open. An old extension looked superseded by
  its version number and was the one actually running.
- A move is as destructive as a delete. Moving a checkpoint tree out from under
  a report array cost 7 of 12 outputs.

## Verifying an ablation

- A null result. Over 60 probes were reported as showing "the two arms are
  bit-identical" while both arms were running the same code path.
- Evidence from the runtime. Two keys were overwritten by the runtime for
  every run; neither override announced itself.
- The override list. This was missed twice in one session, an hour apart, for
  two different keys.
- A positive control. For the manipulation in question it was one number,
  the fraction of entries at a floor -- 0.51-0.71 with the change, exactly
  1.000 without it.
- The control on stdout. The fractions were already computed and only ever
  logged to W&B, which the probes ran with disabled. The code knew and did
  not say.
- Reading the contract. Four attempts at one set of records went that way;
  three were patching symptoms, and reading the validating code gave the
  cause in one step.
- **Argument dumps record what was requested.** They are written before runtime
  overrides. For what executed, read the runtime's own log lines.
