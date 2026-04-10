# RecStore Repository Guidelines for AI Agents

This document applies repository-wide unless the current conversation gives a more specific instruction.

## Language and Output Rules

- Write this repository's `AGENTS.md` and agent-facing operating instructions in English.
- Default user-facing conversation output must be in Chinese.
- Default project documentation must be in Chinese unless the user explicitly asks otherwise.
- Default code comments must be in English.

## Commit and Git Rules

- Default commit messages must be in English.
- Prefer Conventional Commits:
  - `feat(scope): ...`
  - `fix(scope): ...`
  - `docs(scope): ...`
  - `refactor(scope): ...`
  - `test(scope): ...`
  - `ci(scope): ...`
- Do not amend commits unless the user explicitly asks for it.
- Never use destructive git commands such as `git reset --hard` or `git checkout --` unless explicitly requested.
- Assume the worktree may be dirty. Do not revert unrelated user changes.

## Default Development Workflow

For feature work or non-trivial bug fix, follow this order unless the user explicitly asks to skip part of it:

1. Understand the local context first.
2. Confirm or propose the design before implementation.
3. For multi-step work, write or follow a plan.
4. Implement in small, reviewable increments.
5. Verify with the narrowest useful tests first, then broader checks as needed.

- Do not jump straight into code changes without understanding the existing implementation and nearby constraints.
- Prefer minimal, local changes over broad refactors.
- Preserve established interfaces and behavior unless the task explicitly changes them.
- If a design doc or plan already exists for the task, execute against it rather than improvising a new architecture.
- Do not claim success before running verification that actually exercises the changed behavior.

## Core Boundaries

Preserve clear boundaries between:

- Storage and server behavior
- Python client protocol and semantics
- Model integration glue
- Training-loop scheduling and optimization logic

Prefer:

- Passing context explicitly across steps or stages
- Making synchronization points obvious
- Isolating performance optimizations from correctness-critical paths
- Failing loudly when invariants are violated

Avoid:

- Hidden shared mutable state across batches or steps
- Async naming that does not match real execution semantics
- Mixing correctness changes with speculative performance tuning in the same patch unless tightly coupled
- Large refactors that obscure the behavioral change being made

## Review Focus

- Prioritize correctness before performance.
- Sparse update visibility across training steps
- Prefetch and read-after-write ordering
- Implicit state carried between batches
- Tensor device, dtype, and shape mismatches
- Fallback path correctness when optimized paths are unavailable
- Background thread lifecycle, shutdown, and exception propagation
- Consistency between Python wrappers and backend behavior

## Coding Rules

- Follow existing repository patterns before introducing new abstractions.
- Keep functions focused and boundaries explicit.
- Prefer readable code over clever code.
- Use ASCII by default unless the file already requires non-ASCII content.
- Add comments only where intent or invariants are non-obvious.
- In Python, make submission, wait, and consumption semantics explicit.
- In C++, preserve surrounding ownership and synchronization style.

## Recently Verified Lessons

- Do not trust README claims without checking the code path. In `rs_demo`, the documented prefetch read path had drifted from the actual runner behavior.
- In distributed RecStore paths, routing semantics must follow `distributed_client`, not whichever config block is easiest to read.
- Treat `hash_method`, `num_shards`, and `servers` as separate fields. If one is missing, fall back field-by-field instead of replacing the whole config block.
- Do not assume `shard == sorted_index`. Route by explicit shard id.
- If a Python wrapper recreates backend routing behavior, it must match backend semantics exactly or fail loudly.
- Async-looking APIs are not automatically safe. Verify whether handles are globally unique or only unique per shard / per client context.
- If different shards can return colliding local prefetch handles, the wrapper must use its own opaque handle layer instead of exposing backend-local ids directly.
- If correctness is the goal, a stable `prefetch + immediate wait` path is better than a misleading async API that can return wrong data.
- Do not describe a correctness-first fallback as proof of async overlap benefit.
- For local benchmark bring-up, first get a stable closed loop. Do not insist on a default allocator or fast path if it crashes the server before any result is produced.
- In the current local `rs_demo` benchmark path, `R2ShmMalloc` can trigger `ps_server` instability. If the task is to obtain a runnable baseline, `PersistLoopShmMalloc` is an acceptable temporary choice, but it must be stated explicitly in the report.
- A result comparing `TorchRec-Local-HBM` with `RecStore-Local-RPC` is only an engineering observation for that lane. It is not enough to claim architecture-level superiority.

## Testing and Verification

- Run the most relevant targeted tests for the changed area.
- If the behavior is not already covered, add tests.
- If tests cannot be run in the current environment, say so explicitly and explain why.

Useful verification layers in this repository include:

- focused Python unit tests under `src/python/pytorch/recstore/unittest`
- model-zoo integration checks under `model_zoo/torchrec_dlrm`
- `model_zoo/rs_demo` smoke and benchmark runs
- compiled test targets in `build/`
- server/client smoke tests against a running `ps_server`

### Default PyTorch Client Verification Flow

When the user asks to validate baseline repository operability, or explicitly asks for a PyTorch client integration check, use this default sequence:

1. Confirm `build/` exists at the repository root.
2. Run `make -j` inside `build/`.
3. Return to the repository root and start:
   `./build/bin/ps_server --config_path ./recstore_config.json`
4. Confirm logs include lines similar to:
   - `bRPC Server shard 0 listening on 127.0.0.1:15123`
   - `bRPC Server shard 1 listening on 127.0.0.1:15124`
5. Run `ctest -R pytorch_client_test -VV` inside `build/`.
6. Stop the manually started `ps_server` after the test completes.

Notes:

- `pytorch_client_test` maps to `src/framework/pytorch/python_client/client_test.py`.
- The test connects to the ports defined in `recstore_config.json`, currently `15123` and `15124`.
- If a usable `ps_server` is already running on those ports, the test may reuse it.
- If the environment blocks local socket binding, start the server and run the client tests in an environment without that restriction.
- If `build/` does not exist or the project is not configured, complete the build setup first.

## Editing Safety

- Read the relevant code before editing it.
- Never overwrite user changes just to make your patch simpler.
- If you encounter unexpected modifications in the same files, work with them unless they directly block the task.
- If a conflict affects correctness and the right resolution is unclear, stop and ask.
- Keep patches scoped to the task at hand.

For agent behavior:

- Do not present hypothetical fixes as completed work.
- Do not say tests pass unless you ran them.
- Do not hide uncertainty. State assumptions and remaining risks clearly.
