# Project workflow

This repository serves Russian speech-to-text with Sber GigaAM v3. Read `README.md` before changing supported API, configuration, or deployment behavior.

## Behavior changes

Use RED-GREEN-REFACTOR for every feature and bug fix:

1. Add a focused test for the observable behavior or regression and run it to confirm it fails for the intended reason.
2. Implement the smallest change that makes the test pass.
3. Refactor without changing behavior, then run the full test suite.
4. Update `README.md` when the API, supported parameters, environment variables, run commands, or deployment behavior changes.

Do not alter compatibility behavior, error envelopes, upload cleanup, SSE cancellation, timestamps, model aliases, or GPU serialization without tests specifying the contract.

## Verification

Use the repository's documented command:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-test.txt
.venv/bin/python -m unittest discover -s tests -v
```

Reuse an existing `.venv`. Tests mock ASR/GPU dependencies and do not verify real checkpoint loading, CUDA/XPU execution, arbitrary media decoding, Docker startup, or model downloads. The CI workflow builds/pushes Docker images; it does not run unit tests. No Python linter or formatter is configured, so do not invent pytest, Ruff, Black, or pre-commit gates.

## Security and operations

- Never commit `.env`, API keys, registry credentials, model weights, user audio, or generated caches.
- Preserve upload-size enforcement and temporary-file cleanup, including stream disconnect/error paths.
- Keep internal decoder, GPU, and exception details in server logs, not client responses.
- Treat uploaded audio and external model artifacts as untrusted. Document operational impact before adding network fetches or credentials.
- Keep the pinned GigaAM upstream reference and Docker publication behavior deliberate; the engine depends on upstream internals.

## Rule synchronization

Keep shared policy in root `AGENTS.md`; update paired Cursor and Claude Code rules together with equivalent bodies and matching activation scope. `CLAUDE.md` is a compatibility symlink to `AGENTS.md`. Codex's project hook reads `.cursor/rules/*.mdc` directly; do not copy rule bodies into `.codex` or create `.codex/rules.md`. The hook is fail-open and not a security boundary. Its scoped delivery runs only for recognized edit tools and paths; shell-based file edits may not attach scoped rules, so keep critical constraints in `AGENTS.md`, tests, code, or CI. Codex users must approve it with `/hooks` once per clone and after either hook file changes.
