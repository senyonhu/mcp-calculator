"""Microphone listener MCP server that records in shared mode.

The tool starts a background recording session that captures microphone
input in shared mode so other applications can continue using the
microphone. Recording stops automatically after the specified duration
(40 minutes by default).
"""
import datetime
import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from fastmcp import FastMCP

logger = logging.getLogger("MicrophoneListener")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

mcp = FastMCP("MicrophoneListener")

_recording_thread: threading.Thread | None = None
_stop_event = threading.Event()


def _resolve_input_device(device: int | str | None) -> int | str | None:
    """Pick a valid input device, preferring the provided one if available."""

    try:
        if device is not None:
            sd.check_input_settings(device=device)
            return device

        # Try the default input device first
        default_device = sd.default.device[0]
        if default_device not in (None, -1):
            sd.check_input_settings(device=default_device)
            return default_device

        # Fall back to the first device that supports input
        devices = sd.query_devices()
        for idx, info in enumerate(devices):
            if info.get("max_input_channels", 0) > 0:
                try:
                    sd.check_input_settings(device=idx)
                    logger.info("Using fallback input device '%s' (index %d)", info.get("name"), idx)
                    return idx
                except Exception:
                    continue
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to validate audio device: %s", exc)
        raise

    raise RuntimeError("No input-capable audio device found. Check microphone settings.")


def _record_audio(
    duration_seconds: float,
    output_path: Path,
    samplerate: int,
    channels: int,
    device: int | str | None,
) -> None:
    """Internal helper that streams microphone audio to a WAV file."""

    q: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(indata, frames, time_info, status):
        if status:
            logger.warning("Input status: %s", status)
        q.put(indata.copy())

    logger.info(
        "Opening shared input stream at %s Hz (%s channels) using device %s",
        samplerate,
        channels,
        device,
    )
    with sf.SoundFile(
        output_path,
        mode="x",
        samplerate=samplerate,
        channels=channels,
        subtype="PCM_16",
        format="WAV",
    ) as file:
        try:
            with sd.InputStream(
                samplerate=samplerate,
                channels=channels,
                callback=_callback,
                device=device,
            ):
                start_time = time.monotonic()
                while not _stop_event.is_set():
                    elapsed = time.monotonic() - start_time
                    if elapsed >= duration_seconds:
                        logger.info("Reached duration %.2f seconds, stopping", duration_seconds)
                        break
                    try:
                        data = q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    file.write(data)
        except sd.PortAudioError as exc:
            logger.error("Audio device error during recording: %s", exc)
            raise

    logger.info("Recording finished: %s", output_path)


@mcp.tool()
def start_microphone_listener(
    duration_minutes: int = 40,
    samplerate: int = 48000,
    channels: int = 1,
    output_folder: str | None = None,
    device: int | str | None = None,
) -> dict:
    """Start a non-exclusive microphone recording that stops automatically.

    The recording runs in the background and writes a WAV file. It uses the
    audio driver's shared mode so the microphone remains available to other
    applications while data is being captured.

    Args:
        duration_minutes: Length of the recording in minutes (default 40).
        samplerate: Sample rate in Hz (default 48000).
        channels: Number of channels to capture (default mono).
        output_folder: Optional folder for the output WAV file. Defaults to
            the current working directory.
    """

    global _recording_thread

    if duration_minutes <= 0:
        raise ValueError("duration_minutes must be positive")

    if _recording_thread and _recording_thread.is_alive():
        return {
            "success": False,
            "message": "A recording is already in progress.",
        }

    output_dir = Path(output_folder) if output_folder else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"microphone_capture_{timestamp}.wav"

    duration_seconds = float(duration_minutes) * 60.0

    try:
        resolved_device = _resolve_input_device(device)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not start recording: %s", exc)
        return {
            "success": False,
            "message": f"Failed to open input device: {exc}",
        }

    _stop_event.clear()
    _recording_thread = threading.Thread(
        target=_record_audio,
        args=(duration_seconds, output_file, samplerate, channels, resolved_device),
        daemon=True,
    )
    _recording_thread.start()

    logger.info(
        "Started microphone listener: %s (duration %d minutes, %d Hz, %d channels)",
        output_file,
        duration_minutes,
        samplerate,
        channels,
    )

    return {
        "success": True,
        "message": "Recording started.",
        "output_file": str(output_file),
        "duration_minutes": duration_minutes,
    }


@mcp.tool()
def stop_microphone_listener() -> dict:
    """Stop the current microphone recording if one is running."""

    global _recording_thread

    if not _recording_thread or not _recording_thread.is_alive():
        return {"success": False, "message": "No active recording."}

    _stop_event.set()
    _recording_thread.join(timeout=10)
    _recording_thread = None

    return {"success": True, "message": "Recording stopped."}


if __name__ == "__main__":
    mcp.run(transport="stdio")
