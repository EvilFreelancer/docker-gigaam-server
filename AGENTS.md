# Project instructions

## Project and architecture

This repository packages a Russian speech-to-text HTTP service around Sber's GigaAM v3 models. Read `README.md` for the supported API, model aliases, runtime settings, deployment modes, and known compatibility limitations.

The Python application lives in `gigaam/`:

- `gigaam/main.py` is the FastAPI transport layer. It validates OpenAI-compatible inputs, checks optional bearer authentication, translates errors, formats responses, stores uploads, and handles SSE lifecycle and cleanup. Keep model inference and VAD logic out of route handlers.
- `gigaam/asr.py` owns model loading, checkpoint aliases, audio decoding, VAD chunk planning, batched inference, timestamps, and serialized access to the engine. It uses pinned upstream GigaAM internals; review both Dockerfiles before changing those assumptions.
- `tests/` specifies observable API and ASR behavior. Tests replace the real ASR/GPU dependencies and must not download model weights.
- `docker-compose*.yml`, `.env.dist`, `gigaam/Dockerfile*`, and `.github/workflows/` define NVIDIA CUDA and Intel XPU deployment/build paths.

Keep dependency direction simple: HTTP concerns belong in `main.py`; inference and audio processing belong in `asr.py`. When a change crosses the boundary, keep the interface explicit and test it.

## Behavior changes

Use RED-GREEN-REFACTOR for features and bug fixes:

1. Add a focused test for the observable behavior or regression and run it to confirm it fails for the intended reason.
2. Implement the smallest change that makes that test pass.
3. Refactor without changing the behavior, then run the complete test suite.
4. Update `README.md` when an API contract, supported parameter, environment variable, run command, or deployment behavior changes.

Do not change existing API compatibility behavior, error envelopes, upload cleanup, SSE cancellation, timestamps, model aliases, or GPU serialization without tests that specify the new contract.

## Verification

The documented test command is:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-test.txt
.venv/bin/python -m unittest discover -s tests -v
```

If `.venv` already exists, activate or use it rather than recreating it. Tests use mocked ASR; they do not prove actual checkpoint loading, CUDA/XPU execution, ffmpeg decoding of arbitrary media, Docker startup, or model-download availability. State these limits instead of claiming an end-to-end GPU check.

There is no configured Python linter or formatter in the repository. Do not invent a required `pytest`, `ruff`, `black`, or `pre-commit` command. The CI workflow builds and pushes `linux/amd64` Docker images for CUDA and XPU on version tags or manual dispatch; it does not run unit tests.

## Security and operations

- Never commit `.env`, API keys, Docker Hub credentials, model weights, user audio, or generated caches. Keep secrets in environment configuration and GitHub Actions secrets.
- Preserve the configured upload-size limit and temporary-file cleanup paths, including streaming disconnect/error cleanup.
- Do not return internal decoder, GPU, or exception details to API clients; keep diagnostic details in server logs.
- Treat uploaded audio and external model/checkpoint artifacts as untrusted inputs. Do not add new network fetches or credentials without documenting the operational impact.
- Keep the Docker Hub publication and pinned GigaAM upstream reference deliberate; `asr.py` depends on upstream internals.

## Documentation and agent-rule synchronization

Keep the README aligned with user-visible API and deployment behavior. Project rules are maintained in root `AGENTS.md`, Cursor's `.cursor/rules/`, and Claude Code's `.claude/rules/`; update paired rules together and preserve their matching scope. `CLAUDE.md` is a compatibility symlink to this file. Codex's `.codex/hooks/attach_rules.py` reads Cursor rules directly; do not duplicate their bodies or create `.codex/rules.md`.

The Codex hook is a fail-open context adapter, not a security boundary. Critical requirements must remain here and in tests/code/CI. In each Codex clone, approve the hook with `/hooks`; repeat after changing `.codex/hooks.json` or `.codex/hooks/attach_rules.py`.
