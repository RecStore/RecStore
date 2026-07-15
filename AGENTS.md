# RecStore Agent Guidelines

These instructions apply repository-wide unless the current conversation gives a
more specific instruction.

## Language

- Write `AGENTS.md` and agent-facing operating instructions in English.
- Reply to the user in Chinese by default.
- Write project documentation in Chinese by default unless requested otherwise.
- Write code comments in English by default.

## Task-Specific Guides

Read the relevant current guide before doing specialized work. Prefer the
task-specific skills for benchmark execution details; keep root guidance focused
on repository-wide rules.

- End-to-end RecStore/TorchRec benchmarks: `.agents/skills/benchmark-e2e/SKILL.md`
- Parameter Server, transport, or RDMA benchmarks:
  `.agents/skills/benchmark-ps/SKILL.md`
- KVEngine and storage-only benchmarks:
  `.agents/skills/benchmark-kvengine/SKILL.md`
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

## Architecture Boundaries

Keep these boundaries explicit:

- storage and server behavior
- Python client protocol and semantics
- model integration glue
- training-loop scheduling and optimization logic

Prefer explicit context passing, obvious synchronization points, and loud
failures when invariants are violated. Avoid hidden shared mutable state,
misleading async APIs, and broad refactors that obscure behavior changes.

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

## Verified Lessons

- Do not trust README claims without checking the actual code path.
- Distributed RecStore routing must follow `distributed_client`; treat
  `hash_method`, `num_shards`, and `servers` as separate fields.
- Do not assume `shard == sorted_index`; route by explicit shard id.
- Python wrappers that recreate backend routing must match backend semantics or
  fail loudly.
- Async-looking APIs are not automatically safe; verify handle uniqueness and
  visibility semantics.
- For correctness, prefer explicit submit, wait, and consume boundaries until an
  async path has proven handle uniqueness, ordering, and visibility semantics.
- Treat cross-resource or cross-layer benchmark comparisons as lane observations,
  not architecture-level proof. Use the benchmark skills for current lane names,
  ablation definitions, and reporting requirements.

## Testing

- Run the most relevant targeted tests for the changed area.
- Add tests when behavior is not already covered.
- If tests cannot run in the current environment, say why.

Useful verification layers:

- Python unit tests under `src/python/pytorch/recstore/unittest`
- model-zoo integration checks under `model_zoo/torchrec_dlrm`
- `model_zoo/rs_demo` smoke and benchmark runs
- compiled targets in `build/`
- server/client smoke tests against `ps_server`
- benchmark-specific tests and preflight commands from the matching skill

## PyTorch Client Verification

When asked to validate baseline PyTorch client operability:

1. Confirm `build/` exists.
2. Run `make -j` inside `build/`.
3. Start `./build/bin/ps_server --config_path ./recstore_config.json`.
4. Confirm shards listen on the ports in `recstore_config.json`.
5. Inspect available tests with `ctest -N | rg pytorch_client`.
6. Run the narrow matching `pytorch_client` test with `ctest -R ... -VV`.
7. Stop the manually started server.

## Editing Safety

- Read relevant code before editing.
- Never overwrite or revert user changes just to simplify your patch.
- Work with unexpected changes unless they directly block correctness.
- Ask only when conflicting changes make the right resolution unclear.
- Keep patches scoped to the task.
- Do not present hypothetical fixes as completed work.
