"""Real-time microphone audio capture using sounddevice."""

import threading
import time
from typing import Callable, List, Optional
import numpy as np
import sounddevice as sd

from .buffer import AudioRingBuffer


class AudioCapture:
    """
    Real-time audio capture from microphone.

    Uses sounddevice for cross-platform audio input.
    Stores audio in a ring buffer and notifies callbacks.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        buffer_duration: float = 60.0,
        device: Optional[int] = None,
        block_size: int = 1024
    ):
        """
        Initialize audio capture.

        Args:
            sample_rate: Sample rate in Hz (16000 recommended for speech)
            channels: Number of audio channels (1 for mono)
            buffer_duration: Ring buffer duration in seconds
            device: Audio device index (None for default)
            block_size: Samples per callback
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.block_size = block_size

        self.ring_buffer = AudioRingBuffer(
            max_duration_sec=buffer_duration,
            sample_rate=sample_rate
        )

        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._callbacks: List[Callable[[np.ndarray], None]] = []
        self._lock = threading.Lock()

        # Statistics
        self._frames_captured = 0
        self._start_time: Optional[float] = None

    def add_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        Add a callback to be notified of new audio frames.

        Args:
            callback: Function that receives audio array (float32, mono)
        """
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Remove a previously added callback."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def start(self) -> None:
        """Start audio capture."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()
        self._frames_captured = 0

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            device=self.device,
            blocksize=self.block_size,
            callback=self._audio_callback
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop audio capture."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags
    ) -> None:
        """Internal callback from sounddevice."""
        if status:
            # Log any issues (overflow, underflow)
            pass

        # Extract mono audio
        audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()

        # Write to ring buffer
        self.ring_buffer.write(audio)
        self._frames_captured += len(audio)

        # Notify callbacks
        with self._lock:
            callbacks = self._callbacks.copy()

        for callback in callbacks:
            try:
                callback(audio)
            except Exception:
                pass  # Don't let callback errors break capture

    def get_recent_audio(self, seconds: float) -> np.ndarray:
        """
        Get recent audio from the buffer.

        Args:
            seconds: Duration to retrieve

        Returns:
            Audio samples as float32 array
        """
        return self.ring_buffer.get_last_n_seconds(seconds)

    @property
    def is_running(self) -> bool:
        """Check if capture is active."""
        return self._running

    @property
    def elapsed_time(self) -> float:
        """Time since capture started (seconds)."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def frames_captured(self) -> int:
        """Total frames captured since start."""
        return self._frames_captured

    @staticmethod
    def list_devices() -> List[dict]:
        """List available audio input devices."""
        devices = sd.query_devices()
        input_devices = []
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                input_devices.append({
                    'index': i,
                    'name': dev['name'],
                    'channels': dev['max_input_channels'],
                    'sample_rate': dev['default_samplerate']
                })
        return input_devices

    @staticmethod
    def get_default_device() -> Optional[dict]:
        """Get the default input device info."""
        try:
            default_idx = sd.default.device[0]
            if default_idx is not None:
                dev = sd.query_devices(default_idx)
                return {
                    'index': default_idx,
                    'name': dev['name'],
                    'channels': dev['max_input_channels'],
                    'sample_rate': dev['default_samplerate']
                }
        except Exception:
            pass
        return None
