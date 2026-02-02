"""CLI commands for Vibemaxxing."""

import argparse
import signal
import sys
import time
from typing import Optional

from .display import RealtimeDisplay, MinimalDisplay


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="vibemaxxing",
        description="Vibemaxxing - Real-time Ambient Sentiment Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vibemaxxing start                    Start monitoring with default settings
  vibemaxxing start --model small      Use larger Whisper model
  vibemaxxing start --no-diarization   Disable speaker identification
  vibemaxxing stats                    Show sentiment statistics
  vibemaxxing stats --speaker Speaker_1  Show stats for specific speaker
  vibemaxxing insights                 Generate insights from data
  vibemaxxing export -o data.csv       Export data to CSV
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start ambient monitoring")
    start_parser.add_argument(
        "--model", "-m",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        default="base",
        help="Whisper model size (default: base)"
    )
    start_parser.add_argument(
        "--display",
        choices=["live", "minimal", "none"],
        default="live",
        help="Display mode (default: live)"
    )
    start_parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Disable speaker diarization"
    )
    start_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for ML models (default: auto)"
    )
    start_parser.add_argument(
        "--db",
        type=str,
        default="data/vibemaxxing.db",
        help="Database path"
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument(
        "--speaker", "-s",
        type=str,
        help="Filter by speaker ID"
    )
    stats_parser.add_argument(
        "--days", "-d",
        type=int,
        default=7,
        help="Number of days to analyze (default: 7)"
    )
    stats_parser.add_argument(
        "--db",
        type=str,
        default="data/vibemaxxing.db",
        help="Database path"
    )

    # Insights command
    insights_parser = subparsers.add_parser("insights", help="Generate insights")
    insights_parser.add_argument(
        "--db",
        type=str,
        default="data/vibemaxxing.db",
        help="Database path"
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export data")
    export_parser.add_argument(
        "--format", "-f",
        choices=["csv", "json"],
        default="csv",
        help="Output format (default: csv)"
    )
    export_parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output file path"
    )
    export_parser.add_argument(
        "--days", "-d",
        type=int,
        default=30,
        help="Days of data to export (default: 30)"
    )
    export_parser.add_argument(
        "--db",
        type=str,
        default="data/vibemaxxing.db",
        help="Database path"
    )

    # Speakers command
    speakers_parser = subparsers.add_parser("speakers", help="List all speakers")
    speakers_parser.add_argument(
        "--db",
        type=str,
        default="data/vibemaxxing.db",
        help="Database path"
    )

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run simulation demo")

    return parser


def cmd_start(args) -> None:
    """Start ambient monitoring."""
    from ..main import Vibemaxxing

    # Choose display
    if args.display == "live":
        display = RealtimeDisplay()
    elif args.display == "minimal":
        display = MinimalDisplay()
    else:
        display = None

    # Create app
    app = Vibemaxxing(
        whisper_model=args.model,
        use_diarization=not args.no_diarization,
        device=args.device,
        db_path=args.db,
        display=display
    )

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\nStopping...")
        app.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start
    try:
        app.start()
    except KeyboardInterrupt:
        app.stop()


def cmd_stats(args) -> None:
    """Show statistics."""
    from ..persistence import SentimentRepository

    repo = SentimentRepository(args.db)

    if args.speaker:
        # Speaker-specific stats
        stats = repo.get_speaker_stats(args.speaker)
        if not stats:
            print(f"Speaker '{args.speaker}' not found.")
            return

        print(f"\nStatistics for {args.speaker}")
        print("=" * 40)
        print(f"Utterances:      {stats['utterance_count']}")
        print(f"Avg Sentiment:   {stats['avg_sentiment']:+.2f}")
        print(f"Std Sentiment:   {stats['std_sentiment']:.2f}")
        print(f"Range:           {stats['min_sentiment']:.2f} to {stats['max_sentiment']:.2f}")
        print(f"Positivity:      {stats['positivity_ratio']*100:.1f}%")
        print(f"Primary Emotion: {stats['primary_emotion']}")
        print(f"First Seen:      {stats['first_seen']}")
        print(f"Last Seen:       {stats['last_seen']}")
    else:
        # Global stats
        stats = repo.get_global_stats(days=args.days)

        print(f"\nGlobal Statistics (last {args.days} days)")
        print("=" * 40)

        if stats['utterance_count'] == 0:
            print("No data available.")
            return

        print(f"Utterances:     {stats['utterance_count']}")
        print(f"Speakers:       {stats['speaker_count']}")
        print(f"Sessions:       {stats['session_count']}")
        print(f"Avg Sentiment:  {stats['avg_sentiment']:+.2f}")
        print(f"Std Sentiment:  {stats['std_sentiment']:.2f}")
        print(f"Range:          {stats['min_sentiment']:.2f} to {stats['max_sentiment']:.2f}")
        print(f"Positivity:     {stats['positivity_ratio']*100:.1f}%")


def cmd_insights(args) -> None:
    """Generate insights."""
    from ..persistence import SentimentRepository
    from ..analysis import PatternDetector, SentimentScore, Emotion
    from datetime import datetime

    repo = SentimentRepository(args.db)

    # Get recent utterances
    utterances = repo.get_recent_utterances(limit=500)

    if len(utterances) < 10:
        print("Not enough data for insights. Keep vibemaxxing!")
        return

    # Convert to SentimentScore objects
    scores = [
        SentimentScore(
            text=u['text'],
            sentiment=u['sentiment'],
            emotion=Emotion(u['emotion']),
            confidence=u['confidence'],
            timestamp=u['created_at'],
            speaker_id=u['speaker_id']
        )
        for u in utterances
    ]

    # Detect patterns
    detector = PatternDetector(min_samples=5)

    # Separate user scores (if we can identify them)
    # For now, use all scores
    patterns = detector.detect_all_patterns(scores, scores)

    print("\nVibemaxxing Insights")
    print("=" * 40)

    if not patterns:
        print("Keep collecting data for personalized insights!")
        return

    for i, pattern in enumerate(patterns[:7], 1):
        if pattern.confidence >= 0.4:
            confidence_bar = "●" * int(pattern.confidence * 5) + "○" * (5 - int(pattern.confidence * 5))
            print(f"\n{i}. {pattern.description}")
            print(f"   Confidence: [{confidence_bar}] {pattern.confidence:.0%}")


def cmd_export(args) -> None:
    """Export data."""
    from ..persistence import SentimentRepository

    repo = SentimentRepository(args.db)

    if args.format == "csv":
        repo.export_to_csv(args.output, days=args.days)
        print(f"Exported to {args.output}")
    else:
        print("JSON export not yet implemented")


def cmd_speakers(args) -> None:
    """List all speakers."""
    from ..persistence import SentimentRepository

    repo = SentimentRepository(args.db)
    speakers = repo.get_all_speakers()

    if not speakers:
        print("No speakers found.")
        return

    print("\nKnown Speakers")
    print("=" * 60)
    print(f"{'ID':<15} {'Name':<15} {'Utterances':<12} {'Last Seen'}")
    print("-" * 60)

    for s in speakers:
        name = s['display_name'] or "-"
        last_seen = s['last_seen'].strftime("%Y-%m-%d %H:%M") if s['last_seen'] else "-"
        print(f"{s['speaker_id']:<15} {name:<15} {s['utterance_count']:<12} {last_seen}")


def cmd_demo(args) -> None:
    """Run simulation demo."""
    import subprocess
    import os

    demo_path = os.path.join(os.path.dirname(__file__), "..", "..", "demo.py")
    subprocess.run([sys.executable, demo_path])


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "start":
        cmd_start(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "insights":
        cmd_insights(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "speakers":
        cmd_speakers(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
