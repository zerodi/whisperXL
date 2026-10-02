# Windows installation and validation

This checkout targets Windows x64, NVIDIA GPUs, and Python 3.12–3.13. The default
interpreter is Python 3.13; the lockfile fixes the complete dependency graph.

## Install

Install `uv`, a compatible NVIDIA driver, and FFmpeg. The `ffmpeg` executable
must be on `PATH`: whisperxl itself decodes files with that executable, even though
its ASR backend also includes PyAV. For TorchCodec audio decoding, install the
**full-shared** build from [Gyan's FFmpeg builds](https://www.gyan.dev/ffmpeg/builds/),
which includes the `avcodec`, `avformat`, and other DLLs. A static `ffmpeg.exe`
alone works for whisperxl's file loader but does not satisfy TorchCodec.

```powershell
uv sync --locked --extra dev
uv run --locked python -m nltk.downloader punkt_tab
uv pip check
uv run --locked whisperxl --help
```

Only install `faster-whisper2`, not the original `faster-whisper`, in this
environment. Both distributions provide the `faster_whisper` import namespace.
The supported stack uses Torch 2.11.0/cu128, TorchAudio 2.11, TorchVision 0.26,
TorchCodec 0.17, Transformers 5.18, and pyannote-audio 4.0.7. TorchAudio 2.11 uses
the stable PyTorch ABI and does not need to match Torch's minor version.

## GPU runtimes

PyTorch's CUDA 12.8 libraries are supplied by its wheel. CTranslate2 4.8 uses
CUDA 12 cuBLAS and no longer needs cuDNN. Install the
[CUDA 12.8 toolkit/runtime](https://developer.nvidia.com/cuda-12-8-1-download-archive)
alongside the PyTorch runtime; its `bin` directory must be available to the
process through `PATH`. Do not rename CUDA 13 DLLs to CUDA 12 DLL names.

Verify both engines separately:

```powershell
uv run --locked python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.ones(2, device='cuda').tolist())"
uv run --locked python -c "import ctranslate2; print(ctranslate2.get_supported_compute_types('cuda'))"
uv run --locked python -c "import torch, ctypes; ctypes.WinDLL('cublas64_12.dll'); print('CUDA 12 cuBLAS loads')"
uv run --locked python -c "import whisperx.asr, whisperx.alignment, whisperx.diarize, torchcodec; print('Core imports OK')"
```

Device enumeration does not execute a CTranslate2 model. A real transcription
below is the final check for its CUDA kernels and cuBLAS. On machines without a
GPU, use `--device cpu --compute_type int8` for a functional check.

For direct TorchCodec decoding on Windows, register the shared FFmpeg `bin`
directory before decoding and keep the directory handle alive:

```python
import os
from torchcodec.decoders import AudioDecoder

ffmpeg_dll_directory = os.add_dll_directory(r"C:\tools\ffmpeg-shared\bin")
samples = AudioDecoder("audio-en.wav").get_all_samples()
print(samples.data.shape, samples.sample_rate)
```

Replace the example directory with your shared build's location. This checks
the actual native decoder; importing `torchcodec` alone does not load it.

## Tests and real inference

```powershell
uv run --locked pytest tests -q
uv run --locked whisperxl audio-en.wav --model small --language en --device cuda --compute_type float16 --batch_size 1 --output_dir output-en-1
uv run --locked whisperxl audio-en.wav --model small --language en --device cuda --compute_type float16 --batch_size 4 --interleaved_context --output_dir output-en-4
uv run --locked whisperxl audio-ru.wav --model small --language ru --device cuda --compute_type float16 --batch_size 4 --output_dir output-ru
```

Repeat with `--vad_method silero` to exercise the second VAD. Check that JSON and
SRT contain the expected transcript and that word times are finite, ordered, and
within the recording. Short recordings may not contain enough VAD chunks to
exercise context; use a longer recording for that scenario.

For diarization, accept the conditions of
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
and supply a read token through the existing `--hf_token` option. Use a recording
with two speakers and `--diarize --min_speakers 2 --max_speakers 2`. Model access and a valid token are required for diarization.

## Model cache

Run the same transcription twice using `--model_dir .validation/models`, then
repeat with `--model_cache_only True`. Test a missing cache directory separately:
it must fail rather than silently download an ASR model. For a full offline
check, cache the VAD and NLTK assets too, then set `HF_HUB_OFFLINE=1` and disable
network access. `--model_cache_only` alone does not cover every VAD or TorchAudio
download path.

## Validate an update without replacing the active environment

```powershell
$env:UV_PROJECT_ENVIRONMENT = '.venv-validation'
uv sync --locked --extra dev
uv pip check --python .venv-validation\Scripts\python.exe
uv run --locked pytest tests -q
Remove-Item Env:UV_PROJECT_ENVIRONMENT
```

Keep the active environment until the isolated environment passes unit tests
and real GPU inference. Then sync the active environment with the same lockfile.
CI checks Windows/Python 3.12 and 3.13; real GPU inference remains a local check.
