"""Real-time terminal display for Vibemaxxing."""

import sys
import threading
import time
from typing import List, Optional
from collections import deque

from ..pipeline import ProcessingResult
from ..analysis import Emotion


class RealtimeDisplay:
    """
    Real-time terminal display for sentiment monitoring.

    Shows live transcriptions with sentiment scores and colors.
    """

    # ANSI color codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"

    def __init__(
        self,
        max_lines: int = 15,
        update_interval: float = 0.3,
        show_speaker: bool = True,
        show_emotion: bool = True
    ):
        """
        Initialize display.

        Args:
            max_lines: Maximum transcript lines to show
            update_interval: Refresh rate in seconds
            show_speaker: Show speaker IDs
            show_emotion: Show emotion labels
        """
        self.max_lines = max_lines
        self.update_interval = update_interval
        self.show_speaker = show_speaker
        self.show_emotion = show_emotion

        self._results: deque = deque(maxlen=max_lines)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._start_time: Optional[float] = None
        self._stats = {
            "processed": 0,
            "pending": 0,
            "avg_sentiment": 0.0
        }

    def start(self) -> None:
        """Start the display."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()
        self._clear_screen()
        self._print_header()

        self._thread = threading.Thread(
            target=self._display_loop,
            daemon=True,
            name="Display"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the display."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print(f"\n{self.RESET}")

    def add_result(self, result: ProcessingResult) -> None:
        """Add a processing result to display."""
        with self._lock:
            self._results.append(result)
            self._stats["processed"] += 1

    def update_stats(self, pending: int = 0, avg_sentiment: float = 0.0) -> None:
        """Update display statistics."""
        with self._lock:
            self._stats["pending"] = pending
            self._stats["avg_sentiment"] = avg_sentiment

    def _clear_screen(self) -> None:
        """Clear terminal screen."""
        print("\033[2J\033[H", end="")

    def _move_cursor(self, row: int, col: int = 0) -> None:
        """Move cursor to position."""
        print(f"\033[{row};{col}H", end="")

    def _print_header(self) -> None:
        """Print the header."""
        print(f"{self.BOLD}{self.CYAN}")
        print("=" * 80)
        print("  VIBEMAXXING - Real-time Ambient Sentiment Analysis")
        print("  Press Ctrl+C to stop")
        print("=" * 80)
        print(f"{self.RESET}")

    def _display_loop(self) -> None:
        """Main display loop."""
        while self._running:
            self._refresh_display()
            time.sleep(self.update_interval)

    def _refresh_display(self) -> None:
        """Refresh the display."""
        # Move to transcript area (after header)
        self._move_cursor(7, 0)

        with self._lock:
            results = list(self._results)
            stats = self._stats.copy()

        # Clear transcript area
        for _ in range(self.max_lines + 3):
            print(" " * 80)

        self._move_cursor(7, 0)

        # Print transcripts
        if not results:
            print(f"{self.DIM}  Listening for speech...{self.RESET}")
        else:
            for result in results[-self.max_lines:]:
                for sentiment in result.sentiments:
                    self._print_utterance(sentiment)

        # Print status bar
        self._move_cursor(7 + self.max_lines + 1, 0)
        self._print_status_bar(stats)

    def _print_utterance(self, sentiment) -> None:
        """Print a single utterance with formatting."""
        # Choose color based on sentiment
        if sentiment.sentiment > 0.3:
            color = self.GREEN
            indicator = "+"
        elif sentiment.sentiment < -0.3:
            color = self.RED
            indicator = "-"
        else:
            color = self.YELLOW
            indicator = "~"

        # Emotion emoji
        emotion_emoji = {
            Emotion.JOY: "+",
            Emotion.SADNESS: "-",
            Emotion.ANGER: "!",
            Emotion.FEAR: "?",
            Emotion.SURPRISE: "*",
            Emotion.NEUTRAL: "~"
        }.get(sentiment.emotion, "~")

        # Format speaker
        speaker = sentiment.speaker_id or "Unknown"
        if len(speaker) > 12:
            speaker = speaker[:12]

        # Format text
        text = sentiment.text
        max_text_len = 50
        if len(text) > max_text_len:
            text = text[:max_text_len] + "..."

        # Build output
        output = f"{color}[{indicator}] {sentiment.sentiment:+.2f}"

        if self.show_speaker:
            output += f" | {speaker:12s}"

        if self.show_emotion:
            output += f" | {emotion_emoji}"

        output += f" | {text}{self.RESET}"

        print(output)

    def _print_status_bar(self, stats: dict) -> None:
        """Print the status bar."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"

        # Overall sentiment indicator
        avg = stats.get("avg_sentiment", 0)
        if avg > 0.2:
            vibe = f"{self.GREEN}GOOD VIBES{self.RESET}"
        elif avg < -0.2:
            vibe = f"{self.RED}LOW VIBES{self.RESET}"
        else:
            vibe = f"{self.YELLOW}NEUTRAL{self.RESET}"

        print(f"{self.DIM}─" * 80 + f"{self.RESET}")
        print(
            f"  {self.BOLD}Time:{self.RESET} {elapsed_str}  |  "
            f"{self.BOLD}Processed:{self.RESET} {stats.get('processed', 0)}  |  "
            f"{self.BOLD}Pending:{self.RESET} {stats.get('pending', 0)}  |  "
            f"{self.BOLD}Vibe:{self.RESET} {vibe}"
        )


class MinimalDisplay:
    """Minimal display that just prints utterances."""

    def __init__(self):
        self._lock = threading.Lock()

    def start(self) -> None:
        print("Vibemaxxing started. Listening...")
        print("-" * 40)

    def stop(self) -> None:
        print("-" * 40)
        print("Stopped.")

    def add_result(self, result: ProcessingResult) -> None:
        with self._lock:
            for sentiment in result.sentiments:
                indicator = "+" if sentiment.sentiment > 0.3 else "-" if sentiment.sentiment < -0.3 else "~"
                speaker = sentiment.speaker_id or "?"
                print(f"[{indicator}] {sentiment.sentiment:+.2f} | {speaker}: {sentiment.text}")

    def update_stats(self, **kwargs) -> None:
        pass
