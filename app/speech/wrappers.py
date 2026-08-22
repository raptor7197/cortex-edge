"""Offline speech wrappers (PLAN Phase 9): whisper.cpp STT + Piper TTS.

Both invoke local binaries (whisper-cli, piper) and raise a clear
error when they are not installed — they are optional runtime
dependencies for the Raspberry Pi deployment.
"""

import subprocess
import tempfile
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def transcribe(audio_path: str | Path, model: str = "tiny.en") -> str:
    """Speech-to-text via whisper.cpp. Accepts any ffmpeg-readable file."""
    binary = _find_binary("whisper-cli", "whisper")
    model_path = MODELS_DIR / "stt" / f"ggml-{model}.bin"
    if not model_path.exists():
        raise FileNotFoundError(
            f"STT model not found: {model_path}\n"
            "Download: `wget -O models/stt/ggml-tiny.en.bin "
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin`"
        )
    result = subprocess.run(
        [str(binary), "-m", str(model_path), "-f", str(audio_path), "-otxt"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"whisper.cpp failed: {result.stderr[:300]}")
    txt_path = f"{audio_path}.txt"
    text = Path(txt_path).read_text(encoding="utf-8") if Path(txt_path).exists() else ""
    return text.strip()


def synthesize(text: str, voice: str = "en_US-lessac-medium", out_wav: str | None = None) -> str:
    """Text-to-speech via Piper. Returns path to the generated WAV."""
    binary = _find_binary("piper")
    model_dir = MODELS_DIR / "tts"
    out = out_wav or str(Path(tempfile.mktemp(suffix=".wav")))
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
        text_file = f.name
    result = subprocess.run(
        [str(binary), "-m", str(model_dir / voice), "-f", text_file, "-o", out],
        capture_output=True, text=True, timeout=120,
    )
    Path(text_file).unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"piper failed: {result.stderr[:300]}")
    return out


def _find_binary(*names: str) -> Path:
    import shutil

    for name in names:
        path = shutil.which(name)
        if path:
            return Path(path)
    raise FileNotFoundError(
        f"Required binary not found: {'/'.join(names)}. "
        "Install it on the device (see PLAN Phase 9 / README)."
    )
