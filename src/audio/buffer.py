"""Thread-safe ring buffer for audio capture."""

import threading
import numpy as np
from typing import Optional


class AudioRingBuffer:
    """
    Circular buffer for storing audio samples.

    Thread-safe for concurrent read/write operations.
    Maintains a fixed-size window of the most recent audio.
    """

    def __init__(self, max_duration_sec: float = 60.0, sample_rate: int = 16000):
        """
        Initialize ring buffer.

        Args:
            max_duration_sec: Maximum duration of audio to store (seconds)
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_sec * sample_rate)
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.write_pos = 0
        self.total_written = 0
        self._lock = threading.Lock()

    def write(self, audio: np.ndarray) -> None:
        """
        Write audio samples to the buffer.

        Args:
            audio: Audio samples as float32 array
        """
        audio = audio.flatten().astype(np.float32)
        n_samples = len(audio)

        with self._lock:
            if n_samples >= self.max_samples:
                # Audio larger than buffer - keep only the last max_samples
                self.buffer[:] = audio[-self.max_samples:]
                self.write_pos = 0
                self.total_written += n_samples
            elif self.write_pos + n_samples <= self.max_samples:
                # Fits without wrapping
                self.buffer[self.write_pos:self.write_pos + n_samples] = audio
                self.write_pos += n_samples
                self.total_written += n_samples
            else:
                # Needs to wrap around
                first_part = self.max_samples - self.write_pos
                self.buffer[self.write_pos:] = audio[:first_part]
                self.buffer[:n_samples - first_part] = audio[first_part:]
                self.write_pos = n_samples - first_part
                self.total_written += n_samples

    def get_last_n_seconds(self, seconds: float) -> np.ndarray:
        """
        Get the most recent N seconds of audio.

        Args:
            seconds: Duration to retrieve

        Returns:
            Audio samples as float32 array
        """
        n_samples = min(int(seconds * self.sample_rate), self.max_samples)

        with self._lock:
            available = min(self.total_written, self.max_samples)
            n_samples = min(n_samples, available)

            if n_samples == 0:
                return np.array([], dtype=np.float32)

            if self.write_pos >= n_samples:
                # No wrap needed
                return self.buffer[self.write_pos - n_samples:self.write_pos].copy()
            else:
                # Need to wrap around
                first_part = n_samples - self.write_pos
                result = np.empty(n_samples, dtype=np.float32)
                result[:first_part] = self.buffer[self.max_samples - first_part:]
                result[first_part:] = self.buffer[:self.write_pos]
                return result

    def get_all(self) -> np.ndarray:
        """Get all available audio in the buffer."""
        with self._lock:
            available = min(self.total_written, self.max_samples)
            if available == 0:
                return np.array([], dtype=np.float32)

            if self.total_written < self.max_samples:
                return self.buffer[:self.write_pos].copy()
            else:
                # Buffer has wrapped - reconstruct in order
                result = np.empty(self.max_samples, dtype=np.float32)
                first_part = self.max_samples - self.write_pos
                result[:first_part] = self.buffer[self.write_pos:]
                result[first_part:] = self.buffer[:self.write_pos]
                return result

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self.buffer.fill(0)
            self.write_pos = 0
            self.total_written = 0

    @property
    def duration_seconds(self) -> float:
        """Current duration of audio in buffer (seconds)."""
        with self._lock:
            available = min(self.total_written, self.max_samples)
            return available / self.sample_rate

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self._lock:
            return self.total_written == 0
