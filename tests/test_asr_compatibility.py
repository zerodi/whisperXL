"""Exercise the real Transformers batching path without downloading models."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from whisperx.asr import FasterWhisperPipeline, _resolve_model_name, load_model
from whisperx.vads import Vad


class FakeVad(Vad):
    def __init__(self, chunks):
        super().__init__(0.5)
        self.chunks = chunks

    def __call__(self, audio):
        return self.chunks

    @staticmethod
    def preprocess_audio(audio):
        return audio

    @staticmethod
    def merge_chunks(segments, *args, **kwargs):
        return segments


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate_segment_batched(self, features, tokenizer, options, previous_batch_context_tokens):
        ids = features[:, 0].int().tolist()
        self.calls.append((ids, [list(t) for t in previous_batch_context_tokens]))
        return {
            "text": [f"chunk {i}" for i in ids],
            "avg_logprob": [-0.1] * len(ids),
            "tokens": [[i + 1] for i in ids],
        }


class FakePipeline(FasterWhisperPipeline):
    def preprocess(self, audio):
        return {"inputs": torch.tensor([float(audio["inputs"][0])])}


def make_pipeline(count, language="en"):
    chunks = [{"start": float(i), "end": float(i + 1)} for i in range(count)]
    model = FakeModel()
    pipeline = FakePipeline(
        model, FakeVad(chunks), {"vad_onset": 0.5, "vad_offset": 0.363},
        SimpleNamespace(initial_prompt=None),
        tokenizer=SimpleNamespace(language_code=language, task="transcribe"),
        language=language,
    )
    audio = np.repeat(np.arange(max(count, 1), dtype=np.float32), 16000)
    return pipeline, model, audio


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_batches_preserve_segments_and_incomplete_last_batch(batch_size):
    pipeline, model, audio = make_pipeline(9)
    result = pipeline.transcribe(audio, batch_size=batch_size)
    assert [s["text"] for s in result["segments"]] == [f"chunk {i}" for i in range(9)]
    assert [s["start"] for s in result["segments"]] == list(range(9))
    assert result["language"] == "en"
    assert [len(ids) for ids, _ in model.calls][-1] == 1
    assert all(not tokens for _, contexts in model.calls for tokens in contexts)


def test_interleaved_context_and_wraparound_with_partial_batch():
    pipeline, model, audio = make_pipeline(9)
    result = pipeline.transcribe(audio, batch_size=4, interleaved_context=True)
    assert [s["text"] for s in result["segments"]] == [f"chunk {i}" for i in range(9)]
    assert model.calls[0] == ([0, 3, 5, 7], [[], [], [], []])
    assert model.calls[1] == ([1, 4, 6, 8], [[1], [4], [6], [8]])
    assert model.calls[2] == ([2], [[1, 2]])
    assert model.calls[3] == ([0, 3, 5, 7], [[], [1, 2, 3], [4, 5], [6, 7]])


@pytest.mark.parametrize("interleaved", [False, True])
def test_empty_vad_returns_language_without_running_pipeline(interleaved):
    pipeline, model, audio = make_pipeline(0, language="ru")
    progress = Mock()
    assert pipeline.transcribe(audio, batch_size=4, interleaved_context=interleaved,
                               progress_callback=progress) == {"segments": [], "language": "ru"}
    assert model.calls == []
    progress.assert_called_once_with(100.0)


def test_empty_vad_autodetects_language_without_creating_tokenizer():
    pipeline, model, audio = make_pipeline(0)
    pipeline.preset_language = None
    pipeline.tokenizer = None
    pipeline.detect_language = Mock(return_value="ru")
    assert pipeline.transcribe(audio)["language"] == "ru"
    pipeline.detect_language.assert_called_once_with(audio)
    assert pipeline.tokenizer is None
    assert model.calls == []


@pytest.mark.parametrize("name,expected", [
    ("large", "large-v3"),
    ("large-v1", "Systran/faster-whisper-large-v1"),
    ("large-v2", "Systran/faster-whisper-large-v2"),
    ("distil-large-v2", "Systran/faster-distil-whisper-large-v2"),
    ("small", "small"),
    ("organization/custom-model", "organization/custom-model"),
])
def test_load_model_resolves_alias_and_constructs_backend_options(monkeypatch, name, expected):
    backend = SimpleNamespace(model=SimpleNamespace(is_multilingual=True))
    constructor = Mock(return_value=backend)
    monkeypatch.setattr("whisperx.asr.WhisperModel", constructor)
    pipeline = load_model(name, "cpu", vad_model=FakeVad([]), local_files_only=True)
    assert constructor.call_args.args == (expected,)
    assert constructor.call_args.kwargs["local_files_only"] is True
    assert pipeline.options.beam_size == 5


def test_local_directory_takes_precedence_over_legacy_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "large-v2").mkdir()
    assert _resolve_model_name("large-v2") == "large-v2"
