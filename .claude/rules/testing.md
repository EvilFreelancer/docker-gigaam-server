---
paths:
  - "tests/**/*.py"
---

# Tests

- The suite uses Python's built-in `unittest` runner. Add behavior tests in `tests/test_*.py`; use descriptive `test_*` names, `unittest.TestCase`, and the existing test support in `tests/_support.py` where useful.
- Keep tests deterministic and independent of GPUs, downloaded checkpoints, external network access, and local audio files. The API test support injects a fake ASR engine; ASR tests stub Torch/GigaAM dependencies.
- Cover observable success and failure behavior, including status, OpenAI error envelope, headers/content type, response fields, and resource cleanup when relevant.
- Run a focused test first, then the full documented suite:

  ```bash
  .venv/bin/python -m unittest tests.test_api_validation -v
  .venv/bin/python -m unittest discover -s tests -v
  ```

  If using a temporary environment instead of `.venv`, use that environment's Python executable.
- There is no coverage target, pytest configuration, or lint/test CI gate in this repository. Do not add invented thresholds or claim mocked tests validate actual GPU inference.
