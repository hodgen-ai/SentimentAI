"""
Core sentiment analysis classes for SentimentAI.
Handles sentiment scoring, emotion classification, and speaker-level analytics.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np


class Emotion(Enum):
    """Primary emotion categories."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"


@dataclass
class SentimentScore:
    """Represents a sentiment analysis result for a text segment."""
    text: str
    sentiment: float  # -1.0 (negative) to +1.0 (positive)
    emotion: Emotion
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    speaker_id: Optional[str] = None

    @property
    def polarity_label(self) -> str:
        """Human-readable sentiment label."""
        if self.sentiment < -0.3:
            return "Negative"
        elif self.sentiment > 0.3:
            return "Positive"
        else:
            return "Neutral"


class SentimentAnalyzer:
    """
    Core sentiment analysis engine.
    In production, this would use transformer models like DistilBERT or RoBERTa.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize sentiment analyzer.

        Args:
            model_name: HuggingFace model identifier for sentiment analysis
        """
        self.model_name = model_name
        self._emotion_keywords = self._build_emotion_lexicon()

    def _build_emotion_lexicon(self) -> Dict[Emotion, List[str]]:
        """Build keyword-based emotion detection (simplified for demo)."""
        return {
            Emotion.JOY: ["happy", "great", "wonderful", "excellent", "love", "awesome",
                         "fantastic", "perfect", "brilliant", "excited"],
            Emotion.SADNESS: ["sad", "unhappy", "disappointed", "depressed", "terrible",
                            "awful", "miserable", "unfortunate", "sorry"],
            Emotion.ANGER: ["angry", "furious", "annoyed", "frustrated", "mad", "hate",
                          "irritated", "outraged", "pissed"],
            Emotion.FEAR: ["afraid", "scared", "worried", "anxious", "nervous", "terrified",
                         "concerned", "frightened"],
            Emotion.SURPRISE: ["surprised", "shocked", "amazed", "astonished", "unexpected",
                             "wow", "incredible", "unbelievable"],
            Emotion.NEUTRAL: ["okay", "fine", "alright", "normal", "regular"]
        }

    def analyze(self, text: str, speaker_id: Optional[str] = None) -> SentimentScore:
        """
        Analyze sentiment of a text segment.

        In production, this would use a trained transformer model.
        For demo purposes, uses keyword-based heuristics.

        Args:
            text: Text to analyze
            speaker_id: Optional speaker identifier

        Returns:
            SentimentScore object with analysis results
        """
        text_lower = text.lower()

        # Simple keyword-based sentiment (demo only)
        positive_words = ["good", "great", "happy", "love", "excellent", "wonderful",
                         "perfect", "amazing", "fantastic", "brilliant", "awesome"]
        negative_words = ["bad", "terrible", "hate", "awful", "horrible", "worst",
                         "disappointing", "frustrated", "annoyed", "angry", "sad"]

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        # Calculate sentiment score
        total_words = len(text.split())
        if total_words == 0:
            sentiment = 0.0
        else:
            sentiment = (pos_count - neg_count) / max(total_words * 0.3, 1.0)
            sentiment = np.clip(sentiment, -1.0, 1.0)

        # Detect primary emotion
        emotion_scores = {}
        for emotion, keywords in self._emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = score

        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        if emotion_scores[primary_emotion] == 0:
            primary_emotion = Emotion.NEUTRAL

        # Confidence based on keyword matches
        confidence = min(0.5 + (pos_count + neg_count) * 0.1, 0.95)

        return SentimentScore(
            text=text,
            sentiment=sentiment,
            emotion=primary_emotion,
            confidence=confidence,
            timestamp=datetime.now(),
            speaker_id=speaker_id
        )


class SpeakerProfile:
    """Tracks sentiment patterns for an individual speaker."""

    def __init__(self, speaker_id: str):
        self.speaker_id = speaker_id
        self.sentiment_history: List[SentimentScore] = []
        self.conversation_count = 0

    def add_sentiment(self, score: SentimentScore):
        """Add a new sentiment score to this speaker's history."""
        self.sentiment_history.append(score)

    @property
    def average_sentiment(self) -> float:
        """Calculate average sentiment across all utterances."""
        if not self.sentiment_history:
            return 0.0
        return np.mean([s.sentiment for s in self.sentiment_history])

    @property
    def sentiment_std(self) -> float:
        """Calculate sentiment standard deviation (emotional volatility)."""
        if len(self.sentiment_history) < 2:
            return 0.0
        return np.std([s.sentiment for s in self.sentiment_history])

    @property
    def positivity_ratio(self) -> float:
        """Ratio of positive utterances to total utterances."""
        if not self.sentiment_history:
            return 0.0
        positive_count = sum(1 for s in self.sentiment_history if s.sentiment > 0.3)
        return positive_count / len(self.sentiment_history)

    @property
    def primary_emotion(self) -> Emotion:
        """Most common emotion for this speaker."""
        if not self.sentiment_history:
            return Emotion.NEUTRAL
        emotions = [s.emotion for s in self.sentiment_history]
        return max(set(emotions), key=emotions.count)

    def get_recent_trend(self, n: int = 10) -> str:
        """Analyze recent sentiment trend."""
        if len(self.sentiment_history) < n:
            return "Insufficient data"

        recent = self.sentiment_history[-n:]
        first_half_avg = np.mean([s.sentiment for s in recent[:n//2]])
        second_half_avg = np.mean([s.sentiment for s in recent[n//2:]])

        diff = second_half_avg - first_half_avg
        if diff > 0.15:
            return "Improving ↑"
        elif diff < -0.15:
            return "Declining ↓"
        else:
            return "Stable →"


class ConversationAnalytics:
    """Analyzes patterns across multiple speakers and conversations."""

    def __init__(self):
        self.speakers: Dict[str, SpeakerProfile] = {}
        self.all_scores: List[SentimentScore] = []

    def add_utterance(self, score: SentimentScore):
        """Add a new utterance to the analytics database."""
        self.all_scores.append(score)

        if score.speaker_id:
            if score.speaker_id not in self.speakers:
                self.speakers[score.speaker_id] = SpeakerProfile(score.speaker_id)
            self.speakers[score.speaker_id].add_sentiment(score)

    def get_speaker_profile(self, speaker_id: str) -> Optional[SpeakerProfile]:
        """Retrieve profile for a specific speaker."""
        return self.speakers.get(speaker_id)

    def compare_speakers(self) -> List[Tuple[str, float]]:
        """
        Compare average sentiment across all speakers.

        Returns:
            List of (speaker_id, avg_sentiment) tuples sorted by sentiment
        """
        comparisons = [
            (speaker_id, profile.average_sentiment)
            for speaker_id, profile in self.speakers.items()
        ]
        return sorted(comparisons, key=lambda x: x[1], reverse=True)

    def detect_correlations(self, target_speaker: str) -> Dict[str, float]:
        """
        Detect how other speakers correlate with target speaker's sentiment.

        Returns:
            Dictionary mapping speaker_id to correlation coefficient
        """
        if target_speaker not in self.speakers:
            return {}

        target_profile = self.speakers[target_speaker]
        correlations = {}

        # Simplified correlation detection
        # In production, this would use time-series analysis
        for speaker_id, profile in self.speakers.items():
            if speaker_id == target_speaker:
                continue

            # Compare average sentiment when interacting
            target_avg = target_profile.average_sentiment
            other_avg = profile.average_sentiment

            # Simple correlation proxy
            diff = target_avg - other_avg
            correlations[speaker_id] = -diff  # Negative diff = positive correlation

        return correlations

    @property
    def global_sentiment(self) -> float:
        """Overall sentiment across all conversations."""
        if not self.all_scores:
            return 0.0
        return np.mean([s.sentiment for s in self.all_scores])

    def generate_insights(self) -> List[str]:
        """Generate human-readable insights from the data."""
        insights = []

        if len(self.speakers) < 2:
            return ["Need more conversation data to generate insights."]

        # Most positive/negative speakers
        comparisons = self.compare_speakers()
        if comparisons:
            most_positive = comparisons[0]
            most_negative = comparisons[-1]
            insights.append(
                f"Most positive speaker: {most_positive[0]} "
                f"(avg sentiment: {most_positive[1]:.2f})"
            )
            insights.append(
                f"Most negative speaker: {most_negative[0]} "
                f"(avg sentiment: {most_negative[1]:.2f})"
            )

        # Emotional volatility
        volatility_scores = [
            (sid, profile.sentiment_std)
            for sid, profile in self.speakers.items()
        ]
        most_volatile = max(volatility_scores, key=lambda x: x[1])
        insights.append(
            f"Most emotionally variable: {most_volatile[0]} "
            f"(std: {most_volatile[1]:.2f})"
        )

        # Global stats
        insights.append(
            f"Overall environment sentiment: {self.global_sentiment:.2f} "
            f"({len(self.all_scores)} utterances analyzed)"
        )

        return insights
