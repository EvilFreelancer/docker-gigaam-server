import importlib.util
import sys
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


APP_DIR = Path(__file__).resolve().parents[1] / "gigaam"


class FakeAudioDecodeError(RuntimeError):
    pass


def load_api_module():
    fake_asr = types.ModuleType("asr")
    fake_asr.AsrEngine = object
    fake_asr.AudioDecodeError = FakeAudioDecodeError
    fake_asr.Transcription = object
    spec = importlib.util.spec_from_file_location("gigaam_api_under_test", APP_DIR / "main.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"asr": fake_asr}):
        spec.loader.exec_module(module)
    return module


API = load_api_module()
RESULT = SimpleNamespace(
    text="hello world",
    duration=2.0,
    segments=[
        SimpleNamespace(
            start=0.0,
            end=2.0,
            text="hello world",
            words=[
                SimpleNamespace(text="hello", start=0.1, end=0.8),
                SimpleNamespace(text="world", start=1.0, end=1.8),
            ],
        )
    ],
)


class FakeEngine:
    models = ["gigaam-v3-e2e-rnnt"]

    def resolve(self, requested):
        if requested in (None, "", "gigaam", "whisper-1", "v3_e2e_rnnt", self.models[0]):
            return self.models[0]
        return None

    def transcribe(
        self, audio_path, model_name, word_timestamps=False, on_segment=None, cancel_event=None
    ):
        if on_segment:
            on_segment(RESULT.segments[0])
        return RESULT


class ApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = API

    def setUp(self):
        self.engine = FakeEngine()
        self.client = TestClient(self.api.app)

    def post_transcription(self, data=None, files=None, api_patches=None):
        request_data = data if data is not None else {"model": "gigaam"}
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(self.api, "_save_upload", new=AsyncMock(return_value="/tmp/sample.wav"))
            )
            stack.enter_context(patch.object(self.api.os, "unlink"))
            if str(request_data.get("stream", "false")).lower() != "true":
                stack.enter_context(
                    patch.object(self.api.asyncio, "to_thread", new=AsyncMock(return_value=RESULT))
                )
            stack.enter_context(patch.object(self.api, "engine", self.engine))
            for name, value in (api_patches or {}).items():
                target = self.api
                parts = name.split(".")
                for part in parts[:-1]:
                    target = getattr(target, part)
                stack.enter_context(patch.object(target, parts[-1], value))
            return self.client.post(
                "/v1/audio/transcriptions",
                data=request_data,
                files=files if files is not None else {
                    "file": ("sample.wav", b"fake audio", "audio/wav")
                },
            )
