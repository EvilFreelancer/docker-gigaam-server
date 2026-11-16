import asyncio
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import UploadFile

from tests._support import ApiTestCase


class TranscriptionTimestampTests(ApiTestCase):
    def request(self, response_format="verbose_json", granularities=None):
        engine = FakeTimestampEngine()
        fields = dict(
            file=UploadFile(file=io.BytesIO(b"audio"), filename="sample.wav"),
            model="gigaam-v3-e2e-rnnt",
            response_format=response_format,
            granularities=None,
            granularities_plain=None,
            stream=False,
            temperature=None,
            include=None,
            include_plain=None,
            languages=None,
            languages_plain=None,
            keywords=None,
            keywords_plain=None,
            chunking_strategy=None,
            known_speaker_names=None,
            known_speaker_names_plain=None,
            known_speaker_references=None,
            known_speaker_references_plain=None,
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

    def test_verbose_json_defaults_to_openai_segment_granularity(self):
        payload, engine = self.request()
        self.assertFalse(engine.word_timestamps)
        self.assertEqual(payload["segments"][0]["text"], "hello world")
        self.assertNotIn("words", payload)

    def test_timestamp_rounding_carries_across_minute_boundary(self):
        self.assertEqual(self.api._timestamp(59.9996), "00:01:00,000")

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


class FakeTimestampEngine:
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
