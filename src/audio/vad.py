"""Voice Activity Detection using WebRTC VAD."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

try:
    import webrtcvad
    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False


@dataclass
class SpeechSegment:
    """Represents a detected speech segment."""
    start_time: float  # seconds
    end_time: float    # seconds

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class VoiceActivityDetector:
    """
    Voice Activity Detection wrapper for WebRTC VAD.

    Detects speech segments in audio streams.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        aggressiveness: int = 2,
        frame_duration_ms: int = 30
    ):
        """
        Initialize VAD.

        Args:
            sample_rate: Audio sample rate (must be 8000, 16000, 32000, or 48000)
            aggressiveness: VAD aggressiveness (0-3, higher = more aggressive filtering)
            frame_duration_ms: Frame duration (must be 10, 20, or 30 ms)
        """
        if not HAS_WEBRTCVAD:
            raise ImportError("webrtcvad is required. Install with: pip install webrtcvad")

        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("Sample rate must be 8000, 16000, 32000, or 48000")

        if frame_duration_ms not in (10, 20, 30):
            raise ValueError("Frame duration must be 10, 20, or 30 ms")

        self.sample_rate = sample_rate
        self.aggressiveness = aggressiveness
        self.frame_duration_ms = frame_duration_ms

        self._vad = webrtcvad.Vad(aggressiveness)

        # Calculate samples per frame
        self.samples_per_frame = int(sample_rate * frame_duration_ms / 1000)

    def is_speech(self, audio_frame: np.ndarray) -> bool:
        """
        Check if a single frame contains speech.

        Args:
            audio_frame: Audio samples (float32 or int16)

        Returns:
            True if speech detected
        """
        # Convert to int16 bytes
        audio_bytes = self._to_bytes(audio_frame)

        # Ensure correct frame size
        expected_bytes = self.samples_per_frame * 2  # 2 bytes per int16
        if len(audio_bytes) != expected_bytes:
            # Pad or truncate
            if len(audio_bytes) < expected_bytes:
                audio_bytes += b'\x00' * (expected_bytes - len(audio_bytes))
            else:
                audio_bytes = audio_bytes[:expected_bytes]

        return self._vad.is_speech(audio_bytes, self.sample_rate)

    def process_audio(
        self,
        audio: np.ndarray,
        min_speech_duration: float = 0.1,
        min_silence_duration: float = 0.3
    ) -> List[SpeechSegment]:
        """
        Process audio and detect speech segments.

        Args:
            audio: Audio samples (float32)
            min_speech_duration: Minimum speech segment duration (seconds)
            min_silence_duration: Silence duration to end a segment (seconds)

        Returns:
            List of detected speech segments
        """
        audio = audio.flatten()

        # Process frame by frame
        frame_samples = self.samples_per_frame
        n_frames = len(audio) // frame_samples

        frame_is_speech = []
        for i in range(n_frames):
            frame = audio[i * frame_samples:(i + 1) * frame_samples]
            frame_is_speech.append(self.is_speech(frame))

        # Convert to segments
        segments = self._frames_to_segments(
            frame_is_speech,
            min_speech_duration,
            min_silence_duration
        )

        return segments

    def _frames_to_segments(
        self,
        frame_is_speech: List[bool],
        min_speech_duration: float,
        min_silence_duration: float
    ) -> List[SpeechSegment]:
        """Convert frame-level speech flags to segments."""
        if not frame_is_speech:
            return []

        frame_duration = self.frame_duration_ms / 1000.0
        min_speech_frames = int(min_speech_duration / frame_duration)
        min_silence_frames = int(min_silence_duration / frame_duration)

        segments = []
        in_speech = False
        speech_start = 0
        silence_count = 0
        speech_count = 0

        for i, is_speech in enumerate(frame_is_speech):
            if not in_speech:
                if is_speech:
                    speech_count += 1
                    if speech_count >= min_speech_frames:
                        in_speech = True
                        speech_start = i - speech_count + 1
                        silence_count = 0
                else:
                    speech_count = 0
            else:
                if is_speech:
                    silence_count = 0
                else:
                    silence_count += 1
                    if silence_count >= min_silence_frames:
                        # End segment
                        end_frame = i - silence_count
                        segments.append(SpeechSegment(
                            start_time=speech_start * frame_duration,
                            end_time=end_frame * frame_duration
                        ))
                        in_speech = False
                        speech_count = 0
                        silence_count = 0

        # Handle segment that extends to end
        if in_speech:
            segments.append(SpeechSegment(
                start_time=speech_start * frame_duration,
                end_time=len(frame_is_speech) * frame_duration
            ))

        return segments

    def _to_bytes(self, audio: np.ndarray) -> bytes:
        """Convert audio to int16 bytes."""
        audio = audio.flatten()

        if audio.dtype == np.float32 or audio.dtype == np.float64:
            # Convert float [-1, 1] to int16
            audio = (audio * 32767).astype(np.int16)
        elif audio.dtype != np.int16:
            audio = audio.astype(np.int16)

        return audio.tobytes()


class SpeechChunker:
    """
    Accumulates audio and emits chunks when speech ends.

    Used to collect speech segments for processing.
    """

    def __init__(
        self,
        vad: VoiceActivityDetector,
        silence_threshold_ms: int = 1000,
        min_chunk_duration: float = 0.5,
        max_chunk_duration: float = 30.0
    ):
        """
        Initialize speech chunker.

        Args:
            vad: Voice activity detector instance
            silence_threshold_ms: Silence duration to trigger chunk emission (ms)
            min_chunk_duration: Minimum chunk duration to emit (seconds)
            max_chunk_duration: Maximum chunk duration before forced emit (seconds)
        """
        self.vad = vad
        self.silence_threshold_frames = int(
            silence_threshold_ms / vad.frame_duration_ms
        )
        self.min_chunk_samples = int(min_chunk_duration * vad.sample_rate)
        self.max_chunk_samples = int(max_chunk_duration * vad.sample_rate)

        self._buffer: List[np.ndarray] = []
        self._silence_frames = 0
        self._has_speech = False
        self._total_samples = 0

    def process_frame(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Process an audio frame and potentially emit a chunk.

        Args:
            audio: Audio frame (float32)

        Returns:
            Complete speech chunk if ready, None otherwise
        """
        audio = audio.flatten()
        is_speech = self.vad.is_speech(audio)

        if is_speech:
            self._has_speech = True
            self._silence_frames = 0
            self._buffer.append(audio)
            self._total_samples += len(audio)
        else:
            if self._has_speech:
                self._buffer.append(audio)
                self._total_samples += len(audio)
                self._silence_frames += 1

                # Check if we should emit
                should_emit = (
                    self._silence_frames >= self.silence_threshold_frames or
                    self._total_samples >= self.max_chunk_samples
                )

                if should_emit and self._total_samples >= self.min_chunk_samples:
                    return self._emit_chunk()

        # Force emit if max duration reached
        if self._total_samples >= self.max_chunk_samples:
            return self._emit_chunk()

        return None

    def _emit_chunk(self) -> Optional[np.ndarray]:
        """Emit accumulated audio and reset state."""
        if not self._buffer:
            return None

        chunk = np.concatenate(self._buffer)

        # Trim trailing silence (keep a small amount)
        keep_silence_samples = int(0.1 * self.vad.sample_rate)
        silence_samples = self._silence_frames * self.vad.samples_per_frame
        trim_samples = max(0, silence_samples - keep_silence_samples)
        if trim_samples > 0:
            chunk = chunk[:-trim_samples]

        self._reset()

        if len(chunk) >= self.min_chunk_samples:
            return chunk
        return None

    def _reset(self) -> None:
        """Reset chunker state."""
        self._buffer = []
        self._silence_frames = 0
        self._has_speech = False
        self._total_samples = 0

    def flush(self) -> Optional[np.ndarray]:
        """Flush any remaining audio."""
        if self._buffer and self._total_samples >= self.min_chunk_samples:
            return self._emit_chunk()
        self._reset()
        return None
