"""Speech-to-text engine using faster-whisper."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging
import gc

import numpy as np

try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


logger = logging.getLogger(__name__)


@dataclass
class WordTiming:
    """Timing information for a single word."""
    word: str
    start: float
    end: float
    probability: float


@dataclass
class TranscriptionSegment:
    """A transcribed segment of audio."""
    text: str
    start_time: float
    end_time: float
    confidence: float
    words: List[WordTiming] = field(default_factory=list)
    language: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class WhisperEngine:
    """
    Speech-to-text engine using faster-whisper.

    faster-whisper is a reimplementation of Whisper using CTranslate2,
    providing 4x faster inference on CPU and 2x on GPU with the same accuracy.
    """

    # Available model sizes (smallest to largest)
    MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        download_root: Optional[str] = None
    ):
        """
        Initialize Whisper engine.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v3)
            device: Device to use (auto, cpu, cuda)
            compute_type: Compute type (auto, float16, int8, int8_float16)
            download_root: Directory to download models to
        """
        if not HAS_FASTER_WHISPER:
            raise ImportError(
                "faster-whisper is required. Install with: pip install faster-whisper"
            )

        if model_size not in self.MODEL_SIZES:
            raise ValueError(f"Invalid model size. Choose from: {self.MODEL_SIZES}")

        self.model_size = model_size
        self.device = self._detect_device(device)
        self.compute_type = self._detect_compute_type(compute_type)
        self.download_root = download_root

        self._model: Optional[WhisperModel] = None
        logger.info(
            f"WhisperEngine initialized: model={model_size}, "
            f"device={self.device}, compute_type={self.compute_type}"
        )

    def _detect_device(self, device: str) -> str:
        """Detect the best available device."""
        if device != "auto":
            return device

        if HAS_TORCH and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _detect_compute_type(self, compute_type: str) -> str:
        """Detect the best compute type for the device."""
        if compute_type != "auto":
            return compute_type

        if self.device == "cuda":
            return "float16"
        return "int8"

    @property
    def model(self) -> WhisperModel:
        """Lazy-load the model on first access."""
        if self._model is None:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root
            )
            logger.info("Whisper model loaded successfully")
        return self._model

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        word_timestamps: bool = True
    ) -> List[TranscriptionSegment]:
        """
        Transcribe audio to text.

        Args:
            audio: Audio samples as float32 array
            sample_rate: Audio sample rate (will be resampled to 16kHz if different)
            language: Language code (e.g., "en"). None for auto-detection.
            beam_size: Beam size for decoding
            vad_filter: Apply VAD filter to remove silence
            word_timestamps: Include word-level timestamps

        Returns:
            List of transcription segments
        """
        # Ensure correct format
        audio = audio.flatten().astype(np.float32)

        # Resample if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)

        # Transcribe
        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps
        )

        # Convert to our format
        results = []
        for segment in segments:
            words = []
            if word_timestamps and segment.words:
                for w in segment.words:
                    words.append(WordTiming(
                        word=w.word,
                        start=w.start,
                        end=w.end,
                        probability=w.probability
                    ))

            # Convert log probability to confidence (0-1 scale)
            # avg_logprob is typically -0.2 to -1.0, lower = less confident
            confidence = min(1.0, max(0.0, 1.0 + segment.avg_logprob))

            results.append(TranscriptionSegment(
                text=segment.text.strip(),
                start_time=segment.start,
                end_time=segment.end,
                confidence=confidence,
                words=words,
                language=info.language
            ))

        return results

    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Simple linear interpolation as fallback
            duration = len(audio) / orig_sr
            target_length = int(duration * target_sr)
            indices = np.linspace(0, len(audio) - 1, target_length)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            logger.info("Unloading Whisper model")
            del self._model
            self._model = None
            gc.collect()

            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._model is not None

    @staticmethod
    def get_model_info() -> Dict[str, Any]:
        """Get information about available models."""
        return {
            "tiny": {
                "params": "39M",
                "vram": "~1GB",
                "speed": "~32x realtime",
                "quality": "Basic"
            },
            "base": {
                "params": "74M",
                "vram": "~1GB",
                "speed": "~16x realtime",
                "quality": "Good"
            },
            "small": {
                "params": "244M",
                "vram": "~2GB",
                "speed": "~6x realtime",
                "quality": "Better"
            },
            "medium": {
                "params": "769M",
                "vram": "~5GB",
                "speed": "~2x realtime",
                "quality": "Great"
            },
            "large-v3": {
                "params": "1550M",
                "vram": "~10GB",
                "speed": "~1x realtime",
                "quality": "Best"
            }
        }
