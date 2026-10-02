# whisperxl

Speech transcription with word-level timestamps, voice activity detection, and speaker diarization. Based on [WhisperX](https://github.com/m-bain/whisperX), using `faster-whisper2`, CTranslate2, wav2vec2, and pyannote-audio.

## Installation

Requires Python 3.12–3.13 (default: 3.13) and FFmpeg on `PATH`. NVIDIA GPU inference uses PyTorch `cu128` and CUDA 12 cuBLAS for CTranslate2. See [Windows setup](WINDOWS_SETUP.md) for runtime installation and checks.

```bash
git clone https://github.com/zerodi/whisperxl.git
cd whisperxl
uv sync --locked
uv run --locked python -m nltk.downloader punkt_tab
```

Use the lockfile to install compatible dependencies. The backend distribution is `faster-whisper2`; it provides the `faster_whisper` module.

## Command line

```bash
uv run --locked whisperxl audio.wav --model small --language en --device cuda --compute_type float16
uv run --locked whisperxl audio.wav --model small --language ru --batch_size 4
uv run --locked whisperxl audio.wav --device cpu --compute_type int8
uv run --locked whisperxl --help
```

Outputs include JSON, SRT, VTT, TXT, and TSV. Select formats with `--output_format` and the destination with `--output_dir`. Use `--highlight_words True` for word highlighting and `--interleaved_context` to preserve context between batches.

For speaker diarization, accept the conditions of [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) and supply a Hugging Face read token:

```bash
uv run --locked whisperxl audio.wav --diarize --hf_token YOUR_TOKEN
```

Use `--min_speakers` and `--max_speakers` when the speaker count is known. See [language examples](EXAMPLES.md).

## Python API

```python
import whisperxl

model = whisperxl.load_model("small", "cuda", compute_type="float16")
audio = whisperxl.load_audio("audio.wav")
result = model.transcribe(audio, batch_size=4)

align_model, metadata = whisperxl.load_align_model(
    language_code=result["language"], device="cuda"
)
result = whisperxl.align(
    result["segments"], align_model, metadata, audio, "cuda"
)
print(result["segments"])
```

The `whisperx` command and imports remain available for compatibility. Diarization is available through `whisperxl.diarize.DiarizationPipeline`.

## Development

```bash
uv sync --locked --extra dev
uv run --locked python -m nltk.downloader punkt_tab
uv run --locked pytest tests -q
uv build
```

CI checks Windows with Python 3.12 and 3.13. GPU inference requires local validation.

## License and citations

[BSD-2-Clause](LICENSE). Original WhisperX authors and contributors retain their attribution. Speaker diarization uses the community model licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

For research using WhisperX or interleaved context, cite the corresponding papers:

```bibtex
@article{bain2022whisperx,
  title={WhisperX: Time-Accurate Speech Transcription of Long-Form Audio},
  author={Bain, Max and Huh, Jaesung and Han, Tengda and Zisserman, Andrew},
  journal={INTERSPEECH 2023},
  year={2023}
}
```

Please also cite the paper below if you enabled interleaved context:

```bibtex
@article{bain2026context,
  title={Context-Aware Interleaved Batching for WhisperX},
  author={Bain, Carlos and Bain, Max},
  journal={arXiv preprint arXiv:2608.31170v1},
  year={2026}
}
```
