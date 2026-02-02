"""Speaker diarization using pyannote.audio."""

from dataclasses import dataclass
from typing import List, Dict, Optional
import logging
import gc
import os

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from pyannote.audio import Pipeline
    HAS_PYANNOTE = True
except ImportError:
    HAS_PYANNOTE = False


logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """A segment of audio attributed to a specific speaker."""
    speaker_id: str
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class DiarizationResult:
    """Result of speaker diarization."""
    segments: List[SpeakerSegment]
    num_speakers: int
    speaker_mapping: Dict[str, str]  # internal_id -> friendly_id


class SpeakerEngine:
    """
    Speaker diarization engine using pyannote.audio.

    Identifies and segments audio by speaker.
    Requires HuggingFace authentication for model access.
    """

    MODEL_ID = "pyannote/speaker-diarization-3.1"

    def __init__(
        self,
        auth_token: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize speaker diarization engine.

        Args:
            auth_token: HuggingFace token (or set HUGGINGFACE_TOKEN env var)
            device: Device to use (auto, cpu, cuda)

        Note:
            You must accept the model terms at:
            https://huggingface.co/pyannote/speaker-diarization-3.1
        """
        if not HAS_PYANNOTE:
            raise ImportError(
                "pyannote.audio is required. Install with: pip install pyannote.audio"
            )

        self.auth_token = auth_token or os.environ.get("HUGGINGFACE_TOKEN")
        if not self.auth_token:
            logger.warning(
                "No HuggingFace token provided. Set HUGGINGFACE_TOKEN env var or pass auth_token. "
                "You must also accept model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1"
            )

        self.device = self._detect_device(device)
        self._pipeline: Optional[Pipeline] = None

        # Speaker registry for consistent IDs across calls
        self._speaker_registry: Dict[str, str] = {}
        self._speaker_counter = 0

        logger.info(f"SpeakerEngine initialized: device={self.device}")

    def _detect_device(self, device: str) -> str:
        """Detect the best available device."""
        if device != "auto":
            return device

        if HAS_TORCH and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @property
    def pipeline(self) -> Pipeline:
        """Lazy-load the diarization pipeline."""
        if self._pipeline is None:
            if not self.auth_token:
                raise ValueError(
                    "HuggingFace token required. Set HUGGINGFACE_TOKEN env var or pass auth_token. "
                    "Get a token at: https://huggingface.co/settings/tokens"
                )

            logger.info(f"Loading pyannote pipeline: {self.MODEL_ID}")
            self._pipeline = Pipeline.from_pretrained(
                self.MODEL_ID,
                use_auth_token=self.auth_token
            )

            if HAS_TORCH:
                self._pipeline.to(torch.device(self.device))

            logger.info("Pyannote pipeline loaded successfully")
        return self._pipeline

    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> DiarizationResult:
        """
        Perform speaker diarization on audio.

        Args:
            audio: Audio samples as float32 array
            sample_rate: Audio sample rate
            min_speakers: Minimum expected speakers (hint)
            max_speakers: Maximum expected speakers (hint)

        Returns:
            DiarizationResult with speaker segments
        """
        audio = audio.flatten().astype(np.float32)

        # Prepare audio for pyannote
        if HAS_TORCH:
            waveform = torch.from_numpy(audio).unsqueeze(0)
        else:
            raise RuntimeError("PyTorch is required for speaker diarization")

        # Run diarization
        diarization_kwargs = {}
        if min_speakers is not None:
            diarization_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarization_kwargs["max_speakers"] = max_speakers

        diarization = self.pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            **diarization_kwargs
        )

        # Parse results
        segments = []
        speakers_in_chunk = set()

        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            # Map to consistent speaker ID
            friendly_id = self._get_or_create_speaker_id(speaker_label)
            speakers_in_chunk.add(friendly_id)

            segments.append(SpeakerSegment(
                speaker_id=friendly_id,
                start_time=turn.start,
                end_time=turn.end
            ))

        # Sort by start time
        segments.sort(key=lambda s: s.start_time)

        return DiarizationResult(
            segments=segments,
            num_speakers=len(speakers_in_chunk),
            speaker_mapping=self._speaker_registry.copy()
        )

    def _get_or_create_speaker_id(self, internal_label: str) -> str:
        """Get or create a friendly speaker ID."""
        if internal_label not in self._speaker_registry:
            self._speaker_counter += 1
            self._speaker_registry[internal_label] = f"Speaker_{self._speaker_counter}"
        return self._speaker_registry[internal_label]

    def reset_speakers(self) -> None:
        """Reset speaker registry (start fresh with new IDs)."""
        self._speaker_registry.clear()
        self._speaker_counter = 0
        logger.info("Speaker registry reset")

    def get_speaker_count(self) -> int:
        """Get total number of unique speakers seen."""
        return len(self._speaker_registry)

    def unload(self) -> None:
        """Unload the pipeline to free memory."""
        if self._pipeline is not None:
            logger.info("Unloading pyannote pipeline")
            del self._pipeline
            self._pipeline = None
            gc.collect()

            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        """Check if the pipeline is currently loaded."""
        return self._pipeline is not None


class SimpleSpeakerTracker:
    """
    Fallback speaker tracking without pyannote.

    Uses simple voice characteristics for basic speaker separation.
    Much less accurate than pyannote but works without authentication.
    """

    def __init__(self):
        self._speaker_embeddings: Dict[str, np.ndarray] = {}
        self._speaker_counter = 0
        self._threshold = 0.7  # Similarity threshold

    def identify_speaker(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Identify speaker from audio using simple features.

        This is a very basic implementation. For production use,
        pyannote or similar is strongly recommended.
        """
        # Extract simple features (RMS energy profile)
        features = self._extract_features(audio)

        # Find best matching speaker
        best_match = None
        best_similarity = 0

        for speaker_id, embedding in self._speaker_embeddings.items():
            similarity = self._compute_similarity(features, embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker_id

        # If no good match, create new speaker
        if best_match is None or best_similarity < self._threshold:
            self._speaker_counter += 1
            speaker_id = f"Speaker_{self._speaker_counter}"
            self._speaker_embeddings[speaker_id] = features
            return speaker_id

        # Update embedding with running average
        self._speaker_embeddings[best_match] = (
            0.9 * self._speaker_embeddings[best_match] + 0.1 * features
        )
        return best_match

    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract simple audio features."""
        audio = audio.flatten()

        # Compute short-time energy profile
        frame_size = 512
        n_frames = len(audio) // frame_size
        energy = np.zeros(min(n_frames, 100))

        for i in range(len(energy)):
            frame = audio[i * frame_size:(i + 1) * frame_size]
            energy[i] = np.sqrt(np.mean(frame ** 2))

        # Normalize
        if energy.max() > 0:
            energy = energy / energy.max()

        return energy

    def _compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute similarity between two feature vectors."""
        # Pad to same length
        max_len = max(len(a), len(b))
        a_padded = np.pad(a, (0, max_len - len(a)))
        b_padded = np.pad(b, (0, max_len - len(b)))

        # Cosine similarity
        dot = np.dot(a_padded, b_padded)
        norm = np.linalg.norm(a_padded) * np.linalg.norm(b_padded)

        if norm == 0:
            return 0

        return dot / norm

    def reset(self) -> None:
        """Reset speaker tracking."""
        self._speaker_embeddings.clear()
        self._speaker_counter = 0
