from tests._support import ApiTestCase


class TranscriptionResponseTests(ApiTestCase):
    def test_default_json_returns_text_and_duration_usage(self):
        response = self.post_transcription()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "text": "hello world", "usage": {"type": "duration", "seconds": 2}
        })
        self.assertEqual(response.headers["content-type"], "application/json")

    def test_text_format_returns_plain_text(self):
        response = self.post_transcription({"model": "gigaam", "response_format": "text"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "hello world")
        self.assertTrue(response.headers["content-type"].startswith("text/plain"))

    def test_srt_format_returns_numbered_subtitles(self):
        response = self.post_transcription({"model": "gigaam", "response_format": "srt"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "1\n00:00:00,000 --> 00:00:02,000\nhello world\n")
        self.assertEqual(response.headers["content-type"], "application/x-subrip")

    def test_vtt_format_returns_header_and_dot_milliseconds(self):
        response = self.post_transcription({"model": "gigaam", "response_format": "vtt"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nhello world\n")
        self.assertEqual(response.headers["content-type"], "text/vtt; charset=utf-8")

    def test_verbose_json_includes_required_fields_and_requested_timestamp_fields(self):
        response = self.post_transcription({
            "model": "gigaam", "response_format": "verbose_json",
            "timestamp_granularities[]": ["segment", "word"],
        })
        payload = response.json()
        self.assertTrue({"task", "language", "duration", "text", "segments", "words", "usage"} <= payload.keys())
        self.assertEqual(payload["duration"], 2.0)
        self.assertEqual(payload["segments"][0]["text"], "hello world")
        self.assertEqual(payload["words"][0], {"word": "hello", "start": 0.1, "end": 0.8})
        self.assertEqual(set(payload["segments"][0]), {
            "id", "seek", "start", "end", "text", "tokens", "temperature",
            "avg_logprob", "compression_ratio", "no_speech_prob",
        })

    def test_verbose_json_can_return_words_without_segments(self):
        response = self.post_transcription({
            "model": "gigaam", "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        })
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("segments", response.json())
        self.assertEqual(response.json()["words"][0]["word"], "hello")
