"""Main processing pipeline for Vibemaxxing."""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, List
import logging

import numpy as np

from .queue import ProcessingQueue, AudioChunk
from ..transcription import WhisperEngine, TranscriptionSegment
from ..diarization import SpeakerEngine, SpeakerSegment, DiarizationResult
from ..analysis import create_analyzer, SentimentScore
from ..persistence import SentimentRepository


logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result from processing an audio chunk."""
    chunk_id: int
    transcription: List[TranscriptionSegment]
    diarization: Optional[DiarizationResult]
    sentiments: List[SentimentScore]
    processing_time: float
    errors: List[str] = field(default_factory=list)


class ProcessingPipeline:
    """
    Main processing pipeline that orchestrates all components.

    Coordinates:
    - Audio chunk queue
    - Whisper transcription
    - Speaker diarization
    - Sentiment analysis
    - Database persistence
    """

    def __init__(
        self,
        whisper_model: str = "base",
        use_diarization: bool = True,
        use_transformer_sentiment: bool = True,
        db_path: str = "data/vibemaxxing.db",
        device: str = "auto",
        on_result: Optional[Callable[[ProcessingResult], None]] = None
    ):
        """
        Initialize processing pipeline.

        Args:
            whisper_model: Whisper model size
            use_diarization: Enable speaker diarization
            use_transformer_sentiment: Use transformer model for sentiment
            db_path: Database path
            device: Device for ML models
            on_result: Callback when a result is ready
        """
        self.device = device
        self.use_diarization = use_diarization
        self.on_result = on_result

        # Initialize components (lazy loaded)
        self._whisper: Optional[WhisperEngine] = None
        self._speaker: Optional[SpeakerEngine] = None
        self._sentiment = None

        self._whisper_model = whisper_model
        self._use_transformer_sentiment = use_transformer_sentiment

        # Repository
        self._repository = SentimentRepository(db_path)

        # Queue
        self.queue = ProcessingQueue(max_size=10)

        # State
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._session_id: Optional[int] = None

        # Stats
        self._processed_count = 0
        self._total_processing_time = 0.0

        logger.info("ProcessingPipeline initialized")

    def _init_whisper(self) -> WhisperEngine:
        """Lazy-load Whisper engine."""
        if self._whisper is None:
            logger.info(f"Initializing Whisper engine: {self._whisper_model}")
            self._whisper = WhisperEngine(
                model_size=self._whisper_model,
                device=self.device
            )
        return self._whisper

    def _init_speaker(self) -> Optional[SpeakerEngine]:
        """Lazy-load speaker engine."""
        if not self.use_diarization:
            return None

        if self._speaker is None:
            try:
                logger.info("Initializing speaker diarization engine")
                self._speaker = SpeakerEngine(device=self.device)
            except Exception as e:
                logger.warning(f"Could not initialize speaker diarization: {e}")
                self.use_diarization = False
                return None
        return self._speaker

    def _init_sentiment(self):
        """Lazy-load sentiment analyzer."""
        if self._sentiment is None:
            logger.info("Initializing sentiment analyzer")
            self._sentiment = create_analyzer(
                use_transformer=self._use_transformer_sentiment,
                device=self.device
            )
        return self._sentiment

    def start(self) -> int:
        """
        Start the processing pipeline.

        Returns:
            Session ID for this run
        """
        if self._running:
            return self._session_id

        # Create new session
        self._session_id = self._repository.create_session()

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="ProcessingWorker"
        )
        self._worker_thread.start()

        logger.info(f"Pipeline started, session {self._session_id}")
        return self._session_id

    def stop(self) -> None:
        """Stop the processing pipeline."""
        if not self._running:
            return

        logger.info("Stopping pipeline...")
        self._running = False

        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

        # End session
        if self._session_id:
            self._repository.end_session(self._session_id)

        logger.info("Pipeline stopped")

    def submit(self, audio: np.ndarray, sample_rate: int, start_time: float) -> bool:
        """
        Submit an audio chunk for processing.

        Args:
            audio: Audio samples
            sample_rate: Sample rate
            start_time: Session time when chunk started

        Returns:
            True if queued, False if dropped due to backpressure
        """
        return self.queue.put(audio, sample_rate, start_time)

    def _worker_loop(self) -> None:
        """Main worker loop that processes chunks."""
        while self._running:
            chunk = self.queue.get(timeout=1.0)
            if chunk is None:
                continue

            try:
                result = self._process_chunk(chunk)

                if self.on_result:
                    self.on_result(result)

                self._processed_count += 1
                self._total_processing_time += result.processing_time

            except Exception as e:
                logger.error(f"Error processing chunk {chunk.chunk_id}: {e}")

            finally:
                self.queue.task_done()

    def _process_chunk(self, chunk: AudioChunk) -> ProcessingResult:
        """Process a single audio chunk through the full pipeline."""
        start_time = time.time()
        errors = []

        # Step 1: Transcribe
        whisper = self._init_whisper()
        transcription = whisper.transcribe(chunk.audio, chunk.sample_rate)

        if not transcription:
            return ProcessingResult(
                chunk_id=chunk.chunk_id,
                transcription=[],
                diarization=None,
                sentiments=[],
                processing_time=time.time() - start_time,
                errors=["No speech detected"]
            )

        # Step 2: Diarization (optional)
        diarization = None
        if self.use_diarization:
            try:
                speaker = self._init_speaker()
                if speaker:
                    diarization = speaker.diarize(chunk.audio, chunk.sample_rate)
            except Exception as e:
                errors.append(f"Diarization failed: {e}")
                logger.warning(f"Diarization failed: {e}")

        # Step 3: Align transcription with speakers and analyze sentiment
        sentiment_analyzer = self._init_sentiment()
        sentiments = []

        for segment in transcription:
            # Find speaker for this segment
            speaker_id = None
            if diarization:
                speaker_id = self._find_speaker(
                    segment.start_time,
                    segment.end_time,
                    diarization.segments
                )

            # Analyze sentiment
            score = sentiment_analyzer.analyze(segment.text, speaker_id=speaker_id)
            sentiments.append(score)

            # Persist to database
            try:
                self._repository.add_utterance(
                    session_id=self._session_id,
                    text=segment.text,
                    start_time=chunk.start_time + segment.start_time,
                    end_time=chunk.start_time + segment.end_time,
                    sentiment=score.sentiment,
                    emotion=score.emotion.value,
                    confidence=score.confidence,
                    speaker_id=speaker_id
                )
            except Exception as e:
                errors.append(f"Database error: {e}")
                logger.error(f"Failed to persist utterance: {e}")

        return ProcessingResult(
            chunk_id=chunk.chunk_id,
            transcription=transcription,
            diarization=diarization,
            sentiments=sentiments,
            processing_time=time.time() - start_time,
            errors=errors
        )

    def _find_speaker(
        self,
        start: float,
        end: float,
        speaker_segments: List[SpeakerSegment]
    ) -> Optional[str]:
        """Find the speaker who spoke during this time range."""
        best_overlap = 0
        best_speaker = None

        for seg in speaker_segments:
            overlap_start = max(start, seg.start_time)
            overlap_end = min(end, seg.end_time)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = seg.speaker_id

        return best_speaker

    # Stats

    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._running

    @property
    def session_id(self) -> Optional[int]:
        """Current session ID."""
        return self._session_id

    @property
    def processed_count(self) -> int:
        """Number of chunks processed."""
        return self._processed_count

    @property
    def avg_processing_time(self) -> float:
        """Average processing time per chunk."""
        if self._processed_count == 0:
            return 0.0
        return self._total_processing_time / self._processed_count

    @property
    def pending_count(self) -> int:
        """Chunks waiting to be processed."""
        return self.queue.pending_count

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            "is_running": self._running,
            "session_id": self._session_id,
            "processed_count": self._processed_count,
            "pending_count": self.queue.pending_count,
            "dropped_count": self.queue.dropped_count,
            "avg_processing_time": self.avg_processing_time
        }

    # Resource management

    def unload_models(self) -> None:
        """Unload ML models to free memory."""
        if self._whisper:
            self._whisper.unload()
            self._whisper = None

        if self._speaker:
            self._speaker.unload()
            self._speaker = None

        if self._sentiment and hasattr(self._sentiment, 'unload'):
            self._sentiment.unload()
            self._sentiment = None

        logger.info("Models unloaded")
