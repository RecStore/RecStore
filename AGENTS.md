# RecStore Agent Guidelines

These instructions apply repository-wide unless the current conversation gives a
more specific instruction.

## Language

- Write `AGENTS.md` and agent-facing operating instructions in English.
- Reply to the user in Chinese by default.
- Write project documentation in English by default unless requested otherwise.
- Write code comments in English by default.

## Task-Specific Guides

Read the relevant current guide before doing specialized work. Prefer the
task-specific skills for benchmark execution details; keep root guidance focused
on repository-wide rules.

- End-to-end RecStore/TorchRec benchmarks: `.agents/skills/rs-benchmark-e2e/SKILL.md`
- Parameter Server, transport, or RDMA benchmarks:
  `.agents/skills/rs-benchmark-ps/SKILL.md`
- KVEngine and storage-only benchmarks:
  `.agents/skills/rs-benchmark-kvengine/SKILL.md`
- RecStore/TorchRec loss alignment:
  `.agents/skills/rs-loss-aligned/SKILL.md`
- General performance interpretation and layer-labeling background:
  `docs/agent/perf.md`

When a skill and `docs/agent/perf.md` disagree, treat the matching skill as the
source of truth for commands, defaults, current benchmark lanes, and report
format.

## Git Rules

- Default commit messages must be English Conventional Commits, for example
  `feat(scope): ...`, `fix(scope): ...`, `docs(scope): ...`.
- Do not amend commits unless explicitly requested.
- Never use destructive commands such as `git reset --hard` or
  `git checkout --` unless explicitly requested.
- Assume the worktree may be dirty. Do not revert unrelated user changes.
- Do not commit transient AI planning files or scratch artifacts such as
  `docs/superpowers/specs/*`.
- Do not commit generated benchmark outputs, temporary runtime directories, or
  large local result artifacts unless explicitly requested.

## Development Workflow

For feature work or non-trivial bug fixes:

1. Understand the local context first.
2. Follow existing design or propose a small design when needed.
3. Implement in small, reviewable increments.
4. Verify with the narrowest useful tests first, then broader checks when risk
   or blast radius requires it.

Do not claim completion before running verification that actually exercises the
changed behavior.

## Experimental Results

Put one-shot script and experiment outputs under `results/`. Do not write them
into the repo root, `src/`, skill directories, or another run's directory.

Each run gets its own directory:

```text
results/<topic>_<MMDDHHMM>/
```

- `<topic>` is a short snake_case label for the experiment or script, for
  example `e2e`, `benchmark_ps`, `benchmark_kvengine`, `loss_aligned`,
  `criteo_kaggle_e2e`. Prefer the matching skill default when one exists.
  Optional tags go in the topic, for example `e2e_rdma` or
  `criteo_kaggle_local_shm_gpu01`.
- `<MMDDHHMM>` is local time from `date +%m%d%H%M`.
- Never reuse an existing directory for a new run. Never use `latest`, `tmp`,
  a leading underscore, or a name without a timestamp.
- If the user already gave `--output-dir` or an output path, use that path
  as-is.

Keep logs, CSVs, plots, and `summary.md` inside that run directory. Nested
lane or repeat subdirectories are fine. `results/` is gitignored; do not
commit generated artifacts unless explicitly requested.

## Review Focus

Prioritize correctness before performance. Pay special attention to:

- sparse update visibility across training steps
- prefetch and read-after-write ordering
- tensor device, dtype, and shape mismatches
- fallback correctness when optimized paths are unavailable
- background thread lifecycle, shutdown, and exception propagation
- consistency between Python wrappers and backend behavior

## Coding Rules

- Follow existing repository patterns before introducing abstractions.
- Prefer readable, local changes over clever or broad refactors.
- Use ASCII by default unless the file already requires non-ASCII content.
- Add comments only for non-obvious intent or invariants.
- In Python, make submission, wait, and consumption semantics explicit.
- In C++, preserve surrounding ownership and synchronization style.
