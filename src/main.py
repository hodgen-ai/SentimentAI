"""Main entry point for Vibemaxxing."""

import signal
import time
import logging
from typing import Optional

import numpy as np

from .audio import AudioCapture, VoiceActivityDetector, SpeechChunker
from .pipeline import ProcessingPipeline, ProcessingResult
from .cli.display import RealtimeDisplay


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Vibemaxxing:
    """
    Main Vibemaxxing application.

    Orchestrates audio capture, VAD, and the processing pipeline.
    """

    def __init__(
        self,
        whisper_model: str = "base",
        use_diarization: bool = True,
        use_transformer_sentiment: bool = True,
        device: str = "auto",
        db_path: str = "data/vibemaxxing.db",
        sample_rate: int = 16000,
        silence_threshold_ms: int = 1000,
        display: Optional[RealtimeDisplay] = None
    ):
        """
        Initialize Vibemaxxing.

        Args:
            whisper_model: Whisper model size
            use_diarization: Enable speaker diarization
            use_transformer_sentiment: Use transformer sentiment model
            device: Device for ML models
            db_path: Database path
            sample_rate: Audio sample rate
            silence_threshold_ms: Silence duration to trigger processing
            display: Display instance for output
        """
        self.sample_rate = sample_rate
        self.display = display

        # Initialize components
        self._audio_capture = AudioCapture(
            sample_rate=sample_rate,
            channels=1,
            buffer_duration=60.0
        )

        self._vad = VoiceActivityDetector(
            sample_rate=sample_rate,
            aggressiveness=2
        )

        self._chunker = SpeechChunker(
            vad=self._vad,
            silence_threshold_ms=silence_threshold_ms,
            min_chunk_duration=0.5,
            max_chunk_duration=30.0
        )

        self._pipeline = ProcessingPipeline(
            whisper_model=whisper_model,
            use_diarization=use_diarization,
            use_transformer_sentiment=use_transformer_sentiment,
            db_path=db_path,
            device=device,
            on_result=self._on_result
        )

        # State
        self._running = False
        self._session_start: Optional[float] = None
        self._sentiments = []

    def start(self) -> None:
        """Start Vibemaxxing."""
        if self._running:
            return

        logger.info("Starting Vibemaxxing...")

        self._running = True
        self._session_start = time.time()
        self._sentiments = []

        # Start pipeline
        session_id = self._pipeline.start()
        logger.info(f"Session {session_id} started")

        # Start display
        if self.display:
            self.display.start()

        # Start audio capture with callback
        self._audio_capture.add_callback(self._on_audio_frame)
        self._audio_capture.start()

        logger.info("Listening for speech...")

        # Main loop
        try:
            while self._running:
                time.sleep(0.1)

                # Update display stats
                if self.display:
                    avg_sentiment = 0.0
                    if self._sentiments:
                        avg_sentiment = np.mean([s.sentiment for s in self._sentiments[-100:]])
                    self.display.update_stats(
                        pending=self._pipeline.pending_count,
                        avg_sentiment=avg_sentiment
                    )
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop Vibemaxxing."""
        if not self._running:
            return

        logger.info("Stopping Vibemaxxing...")

        self._running = False

        # Stop audio capture
        self._audio_capture.remove_callback(self._on_audio_frame)
        self._audio_capture.stop()

        # Flush any remaining audio
        remaining = self._chunker.flush()
        if remaining is not None:
            self._submit_chunk(remaining)

        # Stop pipeline
        self._pipeline.stop()

        # Stop display
        if self.display:
            self.display.stop()

        # Print summary
        self._print_summary()

        logger.info("Vibemaxxing stopped")

    def _on_audio_frame(self, audio: np.ndarray) -> None:
        """Handle incoming audio frame."""
        if not self._running:
            return

        # Process through VAD chunker
        chunk = self._chunker.process_frame(audio)

        if chunk is not None:
            self._submit_chunk(chunk)

    def _submit_chunk(self, audio: np.ndarray) -> None:
        """Submit audio chunk for processing."""
        # Calculate chunk start time
        chunk_duration = len(audio) / self.sample_rate
        elapsed = time.time() - self._session_start
        start_time = elapsed - chunk_duration

        # Submit to pipeline
        success = self._pipeline.submit(audio, self.sample_rate, start_time)

        if not success:
            logger.warning("Chunk dropped due to backpressure")

    def _on_result(self, result: ProcessingResult) -> None:
        """Handle processing result."""
        # Update display
        if self.display:
            self.display.add_result(result)

        # Track sentiments
        self._sentiments.extend(result.sentiments)

        # Log errors
        for error in result.errors:
            logger.warning(f"Processing error: {error}")

    def _print_summary(self) -> None:
        """Print session summary."""
        elapsed = time.time() - self._session_start if self._session_start else 0

        print("\n" + "=" * 40)
        print("Session Summary")
        print("=" * 40)
        print(f"Duration:     {int(elapsed // 60)}m {int(elapsed % 60)}s")
        print(f"Utterances:   {len(self._sentiments)}")

        if self._sentiments:
            sentiments = [s.sentiment for s in self._sentiments]
            print(f"Avg Sentiment: {np.mean(sentiments):+.2f}")
            print(f"Range:         {min(sentiments):.2f} to {max(sentiments):.2f}")

            positive = sum(1 for s in sentiments if s > 0.3)
            negative = sum(1 for s in sentiments if s < -0.3)
            neutral = len(sentiments) - positive - negative

            print(f"Positive:      {positive} ({positive/len(sentiments)*100:.0f}%)")
            print(f"Neutral:       {neutral} ({neutral/len(sentiments)*100:.0f}%)")
            print(f"Negative:      {negative} ({negative/len(sentiments)*100:.0f}%)")

        stats = self._pipeline.get_stats()
        print(f"Dropped:       {stats['dropped_count']}")
        print(f"Avg Process:   {stats['avg_processing_time']:.2f}s")
        print("=" * 40)

    @property
    def is_running(self) -> bool:
        """Check if app is running."""
        return self._running


def main():
    """CLI entry point."""
    from .cli.commands import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
