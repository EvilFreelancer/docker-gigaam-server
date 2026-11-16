import asyncio
import io
from unittest.mock import patch

from fastapi import UploadFile

from tests._support import ApiTestCase


class UploadTests(ApiTestCase):
    def test_empty_upload_is_rejected(self):
        with self.assertRaises(self.api.ApiError) as raised:
            asyncio.run(self.api._save_upload(UploadFile(file=io.BytesIO(b""), filename="empty.wav")))
        self.assertEqual(raised.exception.status_code, 400)

    def test_upload_size_limit_is_enforced(self):
        with patch.object(self.api, "MAX_FILE_MB", 0):
            with self.assertRaises(self.api.ApiError) as raised:
                asyncio.run(self.api._save_upload(UploadFile(file=io.BytesIO(b"x"), filename="large.wav")))
        self.assertEqual(raised.exception.status_code, 413)
