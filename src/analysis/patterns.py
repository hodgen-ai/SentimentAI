"""
Pattern detection for Vibemaxxing.
Identifies emotional patterns, speaker influence, and temporal trends.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import defaultdict
import numpy as np

from .sentiment import SentimentScore, Emotion


@dataclass
class Pattern:
    """Represents a detected pattern."""
    pattern_type: str  # "speaker_influence", "temporal", "trend", "volatility"
    description: str
    confidence: float  # 0.0 to 1.0
    metadata: Dict

    def __str__(self) -> str:
        return f"[{self.pattern_type}] {self.description} (confidence: {self.confidence:.0%})"


class PatternDetector:
    """
    Detects patterns in sentiment data.

    Identifies:
    - Speaker influence on user mood
    - Temporal patterns (time of day, day of week)
    - Emotional trends
    - Volatility patterns
    """

    def __init__(self, min_samples: int = 5):
        """
        Initialize pattern detector.

        Args:
            min_samples: Minimum samples required for pattern detection
        """
        self.min_samples = min_samples

    def detect_all_patterns(
        self,
        user_scores: List[SentimentScore],
        all_scores: List[SentimentScore]
    ) -> List[Pattern]:
        """
        Detect all patterns from the data.

        Args:
            user_scores: Sentiment scores from the user's utterances
            all_scores: All sentiment scores (including other speakers)

        Returns:
            List of detected patterns
        """
        patterns = []

        patterns.extend(self.detect_speaker_influence(user_scores, all_scores))
        patterns.extend(self.detect_temporal_patterns(user_scores))
        patterns.extend(self.detect_trend_patterns(user_scores))
        patterns.extend(self.detect_volatility_patterns(user_scores))

        # Sort by confidence
        patterns.sort(key=lambda p: p.confidence, reverse=True)

        return patterns

    def detect_speaker_influence(
        self,
        user_scores: List[SentimentScore],
        all_scores: List[SentimentScore]
    ) -> List[Pattern]:
        """
        Detect how other speakers influence user's sentiment.

        Looks at user's sentiment after interacting with each speaker.
        """
        patterns = []

        if len(user_scores) < self.min_samples:
            return patterns

        # Group user sentiment by preceding speaker
        speaker_influence: Dict[str, List[float]] = defaultdict(list)

        for score in user_scores:
            # Find who spoke before the user (within 5 minutes)
            preceding = [
                s for s in all_scores
                if s.timestamp < score.timestamp
                and s.speaker_id
                and s.speaker_id != score.speaker_id
                and (score.timestamp - s.timestamp).total_seconds() < 300
            ]

            if preceding:
                last_speaker = max(preceding, key=lambda s: s.timestamp)
                speaker_influence[last_speaker.speaker_id].append(score.sentiment)

        # Analyze influence
        user_baseline = np.mean([s.sentiment for s in user_scores])

        for speaker_id, sentiments in speaker_influence.items():
            if len(sentiments) < self.min_samples:
                continue

            avg_after = np.mean(sentiments)
            diff = avg_after - user_baseline

            # Calculate confidence based on sample size and effect size
            confidence = min(len(sentiments) / 20, 1.0) * min(abs(diff) * 2, 1.0)

            if abs(diff) > 0.15:  # Significant difference
                direction = "more positive" if diff > 0 else "more negative"
                influence_type = "POSITIVE" if diff > 0 else "NEGATIVE"

                patterns.append(Pattern(
                    pattern_type="speaker_influence",
                    description=f"You tend to be {direction} after talking to {speaker_id}",
                    confidence=confidence,
                    metadata={
                        "speaker_id": speaker_id,
                        "average_sentiment_after": avg_after,
                        "baseline": user_baseline,
                        "difference": diff,
                        "sample_count": len(sentiments),
                        "influence_type": influence_type
                    }
                ))

        return patterns

    def detect_temporal_patterns(
        self,
        scores: List[SentimentScore]
    ) -> List[Pattern]:
        """
        Detect time-based patterns (time of day, day of week).
        """
        patterns = []

        if len(scores) < self.min_samples:
            return patterns

        # Group by hour of day
        hourly: Dict[int, List[float]] = defaultdict(list)
        for score in scores:
            hourly[score.timestamp.hour].append(score.sentiment)

        # Find peak and trough hours
        hourly_avgs = {
            h: np.mean(s) for h, s in hourly.items()
            if len(s) >= max(2, self.min_samples // 2)
        }

        if len(hourly_avgs) >= 2:
            peak_hour = max(hourly_avgs, key=hourly_avgs.get)
            trough_hour = min(hourly_avgs, key=hourly_avgs.get)

            spread = hourly_avgs[peak_hour] - hourly_avgs[trough_hour]
            if spread > 0.2:
                confidence = min(spread * 2, 1.0)

                patterns.append(Pattern(
                    pattern_type="temporal",
                    description=f"Your positivity peaks around {peak_hour}:00",
                    confidence=confidence,
                    metadata={
                        "peak_hour": peak_hour,
                        "peak_sentiment": hourly_avgs[peak_hour],
                        "trough_hour": trough_hour,
                        "trough_sentiment": hourly_avgs[trough_hour],
                        "spread": spread
                    }
                ))

        # Group by day of week
        daily: Dict[int, List[float]] = defaultdict(list)
        for score in scores:
            daily[score.timestamp.weekday()].append(score.sentiment)

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        daily_avgs = {
            d: np.mean(s) for d, s in daily.items()
            if len(s) >= max(2, self.min_samples // 2)
        }

        if len(daily_avgs) >= 2:
            best_day = max(daily_avgs, key=daily_avgs.get)
            worst_day = min(daily_avgs, key=daily_avgs.get)

            spread = daily_avgs[best_day] - daily_avgs[worst_day]
            if spread > 0.15:
                confidence = min(spread * 2.5, 1.0)

                patterns.append(Pattern(
                    pattern_type="temporal",
                    description=f"Your best days are {day_names[best_day]}s",
                    confidence=confidence,
                    metadata={
                        "best_day": day_names[best_day],
                        "best_sentiment": daily_avgs[best_day],
                        "worst_day": day_names[worst_day],
                        "worst_sentiment": daily_avgs[worst_day]
                    }
                ))

        return patterns

    def detect_trend_patterns(
        self,
        scores: List[SentimentScore]
    ) -> List[Pattern]:
        """
        Detect overall sentiment trends over time.
        """
        patterns = []

        if len(scores) < self.min_samples * 2:
            return patterns

        # Sort by timestamp
        sorted_scores = sorted(scores, key=lambda s: s.timestamp)

        # Compare first half to second half
        midpoint = len(sorted_scores) // 2
        first_half = sorted_scores[:midpoint]
        second_half = sorted_scores[midpoint:]

        first_avg = np.mean([s.sentiment for s in first_half])
        second_avg = np.mean([s.sentiment for s in second_half])

        diff = second_avg - first_avg
        if abs(diff) > 0.1:
            direction = "improving" if diff > 0 else "declining"
            confidence = min(abs(diff) * 3, 1.0)

            # Calculate time span
            time_span = sorted_scores[-1].timestamp - sorted_scores[0].timestamp

            patterns.append(Pattern(
                pattern_type="trend",
                description=f"Your overall sentiment is {direction}",
                confidence=confidence,
                metadata={
                    "first_half_avg": first_avg,
                    "second_half_avg": second_avg,
                    "change": diff,
                    "time_span_hours": time_span.total_seconds() / 3600
                }
            ))

        # Weekly trend (if enough data)
        if len(sorted_scores) >= 14:
            last_week = [
                s for s in sorted_scores
                if (sorted_scores[-1].timestamp - s.timestamp).days < 7
            ]
            prev_week = [
                s for s in sorted_scores
                if 7 <= (sorted_scores[-1].timestamp - s.timestamp).days < 14
            ]

            if len(last_week) >= 5 and len(prev_week) >= 5:
                last_avg = np.mean([s.sentiment for s in last_week])
                prev_avg = np.mean([s.sentiment for s in prev_week])
                diff = last_avg - prev_avg

                if abs(diff) > 0.1:
                    direction = "up" if diff > 0 else "down"
                    patterns.append(Pattern(
                        pattern_type="trend",
                        description=f"Your sentiment is {direction} compared to last week",
                        confidence=min(abs(diff) * 2, 0.9),
                        metadata={
                            "this_week_avg": last_avg,
                            "last_week_avg": prev_avg,
                            "change": diff
                        }
                    ))

        return patterns

    def detect_volatility_patterns(
        self,
        scores: List[SentimentScore]
    ) -> List[Pattern]:
        """
        Detect emotional volatility patterns.
        """
        patterns = []

        if len(scores) < self.min_samples:
            return patterns

        sentiments = [s.sentiment for s in scores]
        std = np.std(sentiments)

        # High volatility
        if std > 0.4:
            patterns.append(Pattern(
                pattern_type="volatility",
                description="High emotional variability detected",
                confidence=min(std, 1.0),
                metadata={
                    "std": std,
                    "range": max(sentiments) - min(sentiments),
                    "interpretation": "Your mood swings significantly throughout conversations"
                }
            ))

        # Low volatility (very stable)
        elif std < 0.15 and len(scores) >= 10:
            patterns.append(Pattern(
                pattern_type="volatility",
                description="Very stable emotional state",
                confidence=0.7,
                metadata={
                    "std": std,
                    "interpretation": "Your mood remains consistent across conversations"
                }
            ))

        # Detect rapid swings
        if len(scores) >= 3:
            sorted_scores = sorted(scores, key=lambda s: s.timestamp)
            swings = []

            for i in range(1, len(sorted_scores)):
                diff = abs(sorted_scores[i].sentiment - sorted_scores[i-1].sentiment)
                time_diff = (sorted_scores[i].timestamp - sorted_scores[i-1].timestamp).total_seconds()

                if time_diff < 600 and diff > 0.5:  # Within 10 min, big swing
                    swings.append({
                        "magnitude": diff,
                        "time": sorted_scores[i].timestamp
                    })

            if len(swings) >= 3:
                patterns.append(Pattern(
                    pattern_type="volatility",
                    description="Frequent rapid mood changes detected",
                    confidence=min(len(swings) / 5, 1.0),
                    metadata={
                        "swing_count": len(swings),
                        "avg_magnitude": np.mean([s["magnitude"] for s in swings])
                    }
                ))

        return patterns


def generate_insights_from_patterns(patterns: List[Pattern]) -> List[str]:
    """
    Generate human-readable insights from detected patterns.

    Args:
        patterns: List of detected patterns

    Returns:
        List of insight strings
    """
    insights = []

    for pattern in patterns[:5]:  # Top 5 patterns
        if pattern.confidence >= 0.5:
            insights.append(pattern.description)

    if not insights:
        insights.append("Keep collecting data for more personalized insights")

    return insights
