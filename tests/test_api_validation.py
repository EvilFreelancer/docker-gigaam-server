from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from tests._support import ApiTestCase


class ApiValidationTests(ApiTestCase):
    def test_transcription_requires_configured_bearer_key(self):
        with patch.object(self.api, "API_KEY", "secret"):
            response = self.post_transcription()
        self.assertEqual(response.status_code, 401)
        self.assertIn("error", response.json())

    def test_invalid_response_format_is_openai_bad_request(self):
        response = self.post_transcription({"model": "gigaam", "response_format": "xml"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["param"], "response_format")

    def test_diarized_json_is_rejected_as_unsupported_not_unknown_format(self):
        response = self.post_transcription({"model": "gigaam", "response_format": "diarized_json"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["param"], "response_format")
        self.assertIn("not supported", response.json()["error"]["message"].lower())

    def test_stream_rejects_non_json_response_formats(self):
        response = self.post_transcription({
            "model": "gigaam", "stream": "true", "response_format": "srt"
        })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["param"], "response_format")

    def test_timestamp_granularities_require_verbose_json(self):
        response = self.post_transcription({
            "model": "gigaam", "response_format": "json",
            "timestamp_granularities[]": "segment",
        })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["param"], "timestamp_granularities[]")

    def test_unknown_timestamp_granularity_is_rejected(self):
        response = self.post_transcription({
            "model": "gigaam", "response_format": "verbose_json",
            "timestamp_granularities[]": "sentence",
        })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["param"], "timestamp_granularities[]")

    def test_logprobs_include_is_rejected_instead_of_silently_faked(self):
        response = self.post_transcription({"model": "gigaam", "include[]": "logprobs"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["param"], "include[]")

    def test_unknown_include_value_is_rejected(self):
        response = self.post_transcription({"model": "gigaam", "include[]": "tokens"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["param"], "include[]")

    def test_auto_chunking_strategy_uses_server_vad(self):
        response = self.post_transcription({"model": "gigaam", "chunking_strategy": "auto"})
        self.assertEqual(response.status_code, 200)

    def test_unsupported_language_and_keyword_hints_are_rejected(self):
        cases = [
            ({"languages[]": "ru"}, "languages[]"),
            ({"keywords[]": "GigaAM"}, "keywords[]"),
            ({"known_speaker_names[]": "agent"}, "known_speaker_names[]"),
            ({"known_speaker_names": "agent"}, "known_speaker_names[]"),
            ({"known_speaker_references": "data:audio/wav;base64,AA=="}, "known_speaker_references[]"),
        ]
        for fields, expected_param in cases:
            with self.subTest(fields=fields):
                response = self.post_transcription({"model": "gigaam", **fields})
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["error"]["param"], expected_param)

    def test_language_and_keyword_arrays_without_bracket_suffix_are_rejected(self):
        cases = [
            ({"languages": "ru"}, "languages[]"),
            ({"keywords": "GigaAM"}, "keywords[]"),
        ]
        for fields, expected_param in cases:
            with self.subTest(fields=fields):
                response = self.post_transcription({"model": "gigaam", **fields})
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["error"]["param"], expected_param)

    def test_temperature_above_openai_range_is_rejected(self):
        response = self.post_transcription({"model": "gigaam", "temperature": "1.1"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["param"], "temperature")

    def test_temperature_at_openai_maximum_is_accepted(self):
        response = self.post_transcription({"model": "gigaam", "temperature": "1"})
        self.assertEqual(response.status_code, 200)

    def test_unbracketed_known_speaker_parameter_is_rejected(self):
        response = self.post_transcription({"model": "gigaam", "known_speaker_names": "speaker"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["param"], "known_speaker_names[]")

    def test_speaker_reference_error_names_the_supplied_field(self):
        response = self.post_transcription({
            "model": "gigaam", "known_speaker_references": "data:audio/wav;base64,AA=="
        })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["param"], "known_speaker_references[]")

    def test_missing_file_uses_openai_error_envelope(self):
        response = self.client.post("/v1/audio/transcriptions", data={"model": "gigaam"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_translation_is_rejected_with_openai_error_envelope(self):
        response = self.client.post(
            "/v1/audio/translations",
            files={"file": ("sample.wav", b"audio", "audio/wav")},
            data={"model": "gigaam"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["param"], "model")

    def test_audio_decode_error_is_reported_as_bad_request(self):
        response = self.post_transcription(
            api_patches={
                "asyncio.to_thread": AsyncMock(
                    side_effect=self.api.AudioDecodeError("ffmpeg failed")
                )
            }
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["param"], "file")

    def test_unexpected_transcription_error_uses_openai_server_error_envelope(self):
        client = TestClient(self.api.app, raise_server_exceptions=False)
        with (
            patch.object(self.api, "engine", self.engine),
            patch.object(self.api, "_save_upload", new=AsyncMock(return_value="/tmp/sample.wav")),
            patch.object(self.api.os, "unlink"),
            patch.object(
                self.api.asyncio,
                "to_thread",
                new=AsyncMock(side_effect=RuntimeError("gpu out of memory")),
            ),
        ):
            response = client.post(
                "/v1/audio/transcriptions",
                data={"model": "gigaam"},
                files={"file": ("sample.wav", b"fake audio", "audio/wav")},
            )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["type"], "server_error")
