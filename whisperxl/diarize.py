"""Speaker diarization API under the whisperxl namespace."""

from whisperx.diarize import DiarizationPipeline, IntervalTree, Segment, assign_word_speakers

__all__ = ["DiarizationPipeline", "IntervalTree", "Segment", "assign_word_speakers"]
