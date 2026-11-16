import importlib.util
import sys
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ASR_PATH = Path(__file__).resolve().parents[1] / "gigaam" / "asr.py"


class FakeTensor:
    def to(self, *_args):
        return self


class FakeWave:
    def numel(self):
        return 16000

    def __getitem__(self, _slice):
        return FakeAudioArray()


class FakeAudioArray:
    size = 1

    def numpy(self):
        return self


class FakeDataLoader:
    def __init__(self, *_args, **_kwargs):
        pass

    def __iter__(self):
        yield FakeTensor(), FakeTensor()


class FakeAudioDataset:
    def __init__(self, items, tokenizer=None):
        self.items = items

    @staticmethod
    def collate(batch):
        return batch


class FakeModel:
    _device = "cpu"
    _dtype = "float32"

    def forward(self, wav_pad, wav_lens):
        return "encoded", "encoded_len"

    def _decode(self, *_args):
        return [
            ("first", [SimpleNamespace(text="first", start=0.1, end=0.4)]),
            ("second", [SimpleNamespace(text="second", start=0.2, end=0.5)]),
        ]


def load_asr_module():
    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.nn = SimpleNamespace(Module=object)
    torch.inference_mode = lambda: (lambda function: function)
    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")
    torch_utils_data.DataLoader = FakeDataLoader

    gigaam = types.ModuleType("gigaam")
    gigaam.load_model = lambda *_args, **_kwargs: None
    preprocess = types.ModuleType("gigaam.preprocess")
    preprocess.SAMPLE_RATE = 16000
    preprocess.load_audio = lambda *_args: FakeWave()
    utils = types.ModuleType("gigaam.utils")
    utils.AudioDataset = FakeAudioDataset

    modules = {
        "torch": torch,
        "torch.utils": torch_utils,
        "torch.utils.data": torch_utils_data,
        "gigaam": gigaam,
        "gigaam.preprocess": preprocess,
        "gigaam.utils": utils,
    }
    spec = importlib.util.spec_from_file_location("gigaam_asr_under_test", ASR_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**modules, spec.name: module}):
        spec.loader.exec_module(module)
    return module


class AsrStreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.asr = load_asr_module()

    def test_plan_chunks_merges_close_regions_within_limit(self):
        chunks = self.asr.plan_chunks(
            [(0.0, 1.0), (1.2, 2.0), (5.0, 6.0)], max_chunk_s=3.0, max_gap_s=0.5
        )
        self.assertEqual(chunks, [(0.0, 2.0), (5.0, 6.0)])

    def test_plan_chunks_splits_regions_longer_than_limit(self):
        chunks = self.asr.plan_chunks([(0.0, 7.0)], max_chunk_s=3.0, max_gap_s=0.5)
        self.assertEqual(chunks, [(0.0, 3.0), (3.0, 6.0), (6.0, 7.0)])

    def test_decode_callback_receives_each_segment_as_batch_decodes(self):
        engine = self.asr.AsrEngine.__new__(self.asr.AsrEngine)
        engine.batch_size = 2
        segments = [(0.0, 1.0), (1.0, 2.0)]
        callbacks = []
        with patch.object(self.asr, "DataLoader", FakeDataLoader):
            decoded = engine._decode_chunks(
                FakeModel(), FakeWave(), segments, word_timestamps=True, on_segment=callbacks.append
            )
        self.assertEqual([segment.text for segment in decoded], ["first", "second"])
        self.assertEqual(callbacks, decoded)
        self.assertEqual([segment.start for segment in decoded], [0.0, 1.0])
        self.assertEqual(decoded[0].words[0].start, 0.1)
        self.assertEqual(decoded[1].words[0].start, 1.2)

    def test_decode_stops_sending_segments_when_cancelled(self):
        engine = self.asr.AsrEngine.__new__(self.asr.AsrEngine)
        engine.batch_size = 2
        cancel_event = threading.Event()
        callbacks = []

        def on_segment(segment):
            callbacks.append(segment)
            cancel_event.set()

        with patch.object(self.asr, "DataLoader", FakeDataLoader):
            decoded = engine._decode_chunks(
                FakeModel(),
                FakeWave(),
                [(0.0, 1.0), (1.0, 2.0)],
                word_timestamps=True,
                on_segment=on_segment,
                cancel_event=cancel_event,
            )
        self.assertEqual([segment.text for segment in decoded], ["first"])
        self.assertEqual(callbacks, decoded)

    def test_audio_load_runtime_error_is_marked_as_decode_error(self):
        engine = self.asr.AsrEngine.__new__(self.asr.AsrEngine)
        engine.models = {"gigaam-v3-e2e-rnnt": FakeModel()}
        engine.checkpoints = {"gigaam-v3-e2e-rnnt": "v3_e2e_rnnt"}
        engine.default_model = "gigaam-v3-e2e-rnnt"
        with patch.object(self.asr, "load_audio", side_effect=RuntimeError("ffmpeg failed")):
            with self.assertRaises(self.asr.AudioDecodeError):
                engine.transcribe("bad-audio.wav", "gigaam-v3-e2e-rnnt")

    def test_cancelled_request_skips_vad_after_waiting_for_engine_lock(self):
        engine = self.asr.AsrEngine.__new__(self.asr.AsrEngine)
        engine.models = {"gigaam-v3-e2e-rnnt": FakeModel()}
        engine.checkpoints = {"gigaam-v3-e2e-rnnt": "v3_e2e_rnnt"}
        engine.default_model = "gigaam-v3-e2e-rnnt"
        engine._lock = threading.Lock()
        engine._lock.acquire()
        engine._plan = Mock()
        cancel_event = threading.Event()
        audio_loaded = threading.Event()
        finish_audio_load = threading.Event()
        results = []

        def load_audio(_path):
            audio_loaded.set()
            self.assertTrue(finish_audio_load.wait(timeout=1))
            return FakeWave()

        def transcribe():
            results.append(
                engine.transcribe(
                    "sample.wav", "gigaam-v3-e2e-rnnt", cancel_event=cancel_event
                )
            )

        with patch.object(self.asr, "load_audio", side_effect=load_audio):
            request = threading.Thread(target=transcribe)
            request.start()
            self.assertTrue(audio_loaded.wait(timeout=1))
            cancel_event.set()
            finish_audio_load.set()
            request.join(timeout=1)
            returned_before_lock_release = not request.is_alive()
            engine._lock.release()
            request.join(timeout=1)
        self.assertTrue(returned_before_lock_release)
        self.assertFalse(request.is_alive())
        engine._plan.assert_not_called()
        self.assertEqual(results[0].duration, 1.0)
        self.assertEqual(results[0].segments, [])
