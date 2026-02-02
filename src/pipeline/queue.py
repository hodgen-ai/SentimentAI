"""Thread-safe processing queue for audio chunks."""

import queue
import threading
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class AudioChunk:
    """Represents an audio chunk to be processed."""
    audio: np.ndarray
    sample_rate: int
    start_time: float  # Session time when chunk started
    chunk_id: int
    duration: float = 0.0

    def __post_init__(self):
        if self.duration == 0.0:
            self.duration = len(self.audio) / self.sample_rate


class ProcessingQueue:
    """
    Thread-safe queue for audio chunks awaiting processing.

    Manages backpressure to prevent memory issues during
    high-volume speech.
    """

    def __init__(self, max_size: int = 10):
        """
        Initialize processing queue.

        Args:
            max_size: Maximum chunks to hold (older chunks dropped if exceeded)
        """
        self.max_size = max_size
        self._queue = queue.Queue(maxsize=max_size)
        self._chunk_counter = 0
        self._lock = threading.Lock()
        self._dropped_count = 0

    def put(
        self,
        audio: np.ndarray,
        sample_rate: int,
        start_time: float
    ) -> bool:
        """
        Add audio chunk to queue.

        Args:
            audio: Audio samples
            sample_rate: Sample rate
            start_time: When this chunk started (session time)

        Returns:
            True if added, False if queue full (chunk dropped)
        """
        with self._lock:
            self._chunk_counter += 1
            chunk_id = self._chunk_counter

        chunk = AudioChunk(
            audio=audio.copy(),
            sample_rate=sample_rate,
            start_time=start_time,
            chunk_id=chunk_id
        )

        try:
            self._queue.put_nowait(chunk)
            return True
        except queue.Full:
            with self._lock:
                self._dropped_count += 1
            return False

    def get(self, timeout: float = 1.0) -> Optional[AudioChunk]:
        """
        Get next chunk from queue.

        Args:
            timeout: Seconds to wait for a chunk

        Returns:
            AudioChunk if available, None if timeout
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def task_done(self) -> None:
        """Mark the last retrieved task as done."""
        self._queue.task_done()

    @property
    def pending_count(self) -> int:
        """Number of chunks waiting to be processed."""
        return self._queue.qsize()

    @property
    def dropped_count(self) -> int:
        """Number of chunks dropped due to backpressure."""
        with self._lock:
            return self._dropped_count

    @property
    def total_processed(self) -> int:
        """Total chunks that have been queued."""
        with self._lock:
            return self._chunk_counter

    def clear(self) -> int:
        """Clear the queue, returning number of items cleared."""
        cleared = 0
        while True:
            try:
                self._queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        return cleared

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
