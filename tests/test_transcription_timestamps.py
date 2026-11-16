import asyncio
import importlib.util
import io
import json
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import UploadFile


APP_DIR = Path(__file__).resolve().parents[1] / "gigaam"


def load_api_module():
    fake_asr = types.ModuleType("asr")
    fake_asr.AsrEngine = object
    fake_asr.Transcription = object
    spec = importlib.util.spec_from_file_location("gigaam_main_under_test", APP_DIR / "main.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"asr": fake_asr}):
        spec.loader.exec_module(module)
    return module


class FakeEngine:
    def __init__(self):
        self.word_timestamps = None

    def resolve(self, requested):
        return requested or "gigaam-v3-e2e-rnnt"

    def transcribe(self, audio_path, model_name, word_timestamps):
        self.word_timestamps = word_timestamps
        return SimpleNamespace(
            text="hello world",
            duration=2.0,
            segments=[
                SimpleNamespace(
                    id=0,
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


class TranscriptionTimestampTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = load_api_module()

    def request(self, response_format="verbose_json", granularities=None):
        engine = FakeEngine()
        fields = dict(
            file=UploadFile(file=io.BytesIO(b"audio"), filename="sample.wav"),
            model="gigaam-v3-e2e-rnnt",
            response_format=response_format,
            granularities=None,
            granularities_plain=None,
            stream=False,
        )
        if granularities is not None:
            fields["granularities"] = granularities
        with (
            patch.object(self.api, "_require_engine", return_value=engine),
            patch.object(self.api, "_save_upload", new=AsyncMock(return_value="/tmp/sample.wav")),
            patch.object(self.api.os, "unlink"),
        ):
            response = asyncio.run(self.api.transcriptions(**fields))
        return json.loads(response.body), engine

    def test_verbose_json_defaults_to_segments_and_words(self):
        payload, engine = self.request()

        self.assertTrue(engine.word_timestamps)
        self.assertEqual(payload["segments"][0]["text"], "hello world")
        self.assertEqual(payload["words"][0], {"word": "hello", "start": 0.1, "end": 0.8})

    def test_explicit_segment_granularity_does_not_compute_words(self):
        payload, engine = self.request(granularities=["segment"])

        self.assertFalse(engine.word_timestamps)
        self.assertIn("segments", payload)
        self.assertNotIn("words", payload)

    def test_explicit_word_and_segment_granularities_return_both(self):
        payload, engine = self.request(granularities=["segment", "word"])

        self.assertTrue(engine.word_timestamps)
        self.assertIn("segments", payload)
        self.assertIn("words", payload)

    def test_plain_json_keeps_compact_response_without_timestamps(self):
        payload, engine = self.request(response_format="json")

        self.assertFalse(engine.word_timestamps)
        self.assertEqual(set(payload), {"text", "usage"})


if __name__ == "__main__":
    unittest.main()
