"""
Core sentiment analysis classes for Vibemaxxing.
Handles sentiment scoring, emotion classification, and speaker-level analytics.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from enum import Enum
import logging
import gc

import numpy as np

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


logger = logging.getLogger(__name__)


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


class TransformerSentimentAnalyzer:
    """
    Production sentiment analyzer using RoBERTa.

    Uses the cardiffnlp/twitter-roberta-base-sentiment-latest model
    for high-accuracy sentiment classification.
    """

    MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    def __init__(self, device: str = "auto"):
        """
        Initialize transformer-based sentiment analyzer.

        Args:
            device: Device to use (auto, cpu, cuda)
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers and torch are required. Install with: "
                "pip install transformers torch"
            )

        self.device = self._detect_device(device)
        self._model = None
        self._tokenizer = None
        self._emotion_keywords = self._build_emotion_lexicon()

        logger.info(f"TransformerSentimentAnalyzer initialized: device={self.device}")

    def _detect_device(self, device: str) -> str:
        """Detect the best available device."""
        if device != "auto":
            return device

        if HAS_TRANSFORMERS and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _build_emotion_lexicon(self) -> Dict[Emotion, List[str]]:
        """Build keyword-based emotion detection for sub-classification."""
        return {
            Emotion.JOY: ["happy", "great", "wonderful", "excellent", "love", "awesome",
                         "fantastic", "perfect", "brilliant", "excited", "glad", "joy"],
            Emotion.SADNESS: ["sad", "unhappy", "disappointed", "depressed", "terrible",
                             "awful", "miserable", "unfortunate", "sorry", "grief"],
            Emotion.ANGER: ["angry", "furious", "annoyed", "frustrated", "mad", "hate",
                           "irritated", "outraged", "pissed", "rage"],
            Emotion.FEAR: ["afraid", "scared", "worried", "anxious", "nervous", "terrified",
                          "concerned", "frightened", "panic"],
            Emotion.SURPRISE: ["surprised", "shocked", "amazed", "astonished", "unexpected",
                              "wow", "incredible", "unbelievable"],
            Emotion.NEUTRAL: ["okay", "fine", "alright", "normal", "regular"]
        }

    def _load_model(self) -> None:
        """Lazy-load the model."""
        if self._model is None:
            logger.info(f"Loading sentiment model: {self.MODEL_ID}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_ID)
            self._model.to(self.device)
            self._model.eval()
            logger.info("Sentiment model loaded successfully")

    def _preprocess(self, text: str) -> str:
        """Preprocess text for the model."""
        tokens = []
        for token in text.split():
            if token.startswith("@") and len(token) > 1:
                tokens.append("@user")
            elif token.startswith("http"):
                tokens.append("http")
            else:
                tokens.append(token)
        return " ".join(tokens)

    def analyze(self, text: str, speaker_id: Optional[str] = None) -> SentimentScore:
        """
        Analyze sentiment of a text segment.

        Args:
            text: Text to analyze
            speaker_id: Optional speaker identifier

        Returns:
            SentimentScore object with analysis results
        """
        self._load_model()

        # Preprocess and tokenize
        processed_text = self._preprocess(text)
        encoded = self._tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Get model predictions
        with torch.no_grad():
            output = self._model(**encoded)

        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(output.logits[0], dim=0).cpu().numpy()

        # Model output: [negative, neutral, positive]
        sentiment_value = float(probs[2] - probs[0])  # positive - negative
        confidence = float(max(probs))

        # Determine emotion
        if sentiment_value > 0.3:
            emotion = Emotion.JOY
        elif sentiment_value < -0.3:
            emotion = self._detect_negative_emotion(text)
        else:
            emotion = Emotion.NEUTRAL

        return SentimentScore(
            text=text,
            sentiment=sentiment_value,
            emotion=emotion,
            confidence=confidence,
            timestamp=datetime.now(),
            speaker_id=speaker_id
        )

    def _detect_negative_emotion(self, text: str) -> Emotion:
        """Detect specific negative emotion using keywords."""
        text_lower = text.lower()

        scores = {
            Emotion.ANGER: 0,
            Emotion.FEAR: 0,
            Emotion.SADNESS: 0
        }

        for emotion, keywords in self._emotion_keywords.items():
            if emotion in scores:
                scores[emotion] = sum(1 for w in keywords if w in text_lower)

        best_emotion = max(scores, key=scores.get)
        if scores[best_emotion] == 0:
            return Emotion.SADNESS  # Default negative emotion

        return best_emotion

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            logger.info("Unloading sentiment model")
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            gc.collect()

            if HAS_TRANSFORMERS and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._model is not None


class SentimentAnalyzer:
    """
    Keyword-based sentiment analyzer (fallback/demo mode).

    Use TransformerSentimentAnalyzer for production accuracy.
    """

    def __init__(self, model_name: str = "keyword-based"):
        """
        Initialize sentiment analyzer.

        Args:
            model_name: Identifier (unused, kept for compatibility)
        """
        self.model_name = model_name
        self._emotion_keywords = self._build_emotion_lexicon()

    def _build_emotion_lexicon(self) -> Dict[Emotion, List[str]]:
        """Build keyword-based emotion detection."""
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
        Analyze sentiment using keyword heuristics.

        Args:
            text: Text to analyze
            speaker_id: Optional speaker identifier

        Returns:
            SentimentScore object with analysis results
        """
        text_lower = text.lower()

        positive_words = ["good", "great", "happy", "love", "excellent", "wonderful",
                         "perfect", "amazing", "fantastic", "brilliant", "awesome"]
        negative_words = ["bad", "terrible", "hate", "awful", "horrible", "worst",
                         "disappointing", "frustrated", "annoyed", "angry", "sad"]

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        total_words = len(text.split())
        if total_words == 0:
            sentiment = 0.0
        else:
            sentiment = (pos_count - neg_count) / max(total_words * 0.3, 1.0)
            sentiment = np.clip(sentiment, -1.0, 1.0)

        emotion_scores = {}
        for emotion, keywords in self._emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = score

        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        if emotion_scores[primary_emotion] == 0:
            primary_emotion = Emotion.NEUTRAL

        confidence = min(0.5 + (pos_count + neg_count) * 0.1, 0.95)

        return SentimentScore(
            text=text,
            sentiment=sentiment,
            emotion=primary_emotion,
            confidence=confidence,
            timestamp=datetime.now(),
            speaker_id=speaker_id
        )


def create_analyzer(use_transformer: bool = True, device: str = "auto"):
    """
    Factory function to create the appropriate analyzer.

    Args:
        use_transformer: Use transformer model (True) or keyword fallback (False)
        device: Device for transformer model

    Returns:
        SentimentAnalyzer or TransformerSentimentAnalyzer
    """
    if use_transformer and HAS_TRANSFORMERS:
        return TransformerSentimentAnalyzer(device=device)
    else:
        if use_transformer:
            logger.warning(
                "Transformers not available, falling back to keyword-based analyzer. "
                "Install with: pip install transformers torch"
            )
        return SentimentAnalyzer()


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
        return float(np.mean([s.sentiment for s in self.sentiment_history]))

    @property
    def sentiment_std(self) -> float:
        """Calculate sentiment standard deviation (emotional volatility)."""
        if len(self.sentiment_history) < 2:
            return 0.0
        return float(np.std([s.sentiment for s in self.sentiment_history]))

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

        for speaker_id, profile in self.speakers.items():
            if speaker_id == target_speaker:
                continue

            target_avg = target_profile.average_sentiment
            other_avg = profile.average_sentiment

            diff = target_avg - other_avg
            correlations[speaker_id] = -diff

        return correlations

    @property
    def global_sentiment(self) -> float:
        """Overall sentiment across all conversations."""
        if not self.all_scores:
            return 0.0
        return float(np.mean([s.sentiment for s in self.all_scores]))

    def generate_insights(self) -> List[str]:
        """Generate human-readable insights from the data."""
        insights = []

        if len(self.speakers) < 2:
            return ["Need more conversation data to generate insights."]

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

        volatility_scores = [
            (sid, profile.sentiment_std)
            for sid, profile in self.speakers.items()
        ]
        most_volatile = max(volatility_scores, key=lambda x: x[1])
        insights.append(
            f"Most emotionally variable: {most_volatile[0]} "
            f"(std: {most_volatile[1]:.2f})"
        )

        insights.append(
            f"Overall environment sentiment: {self.global_sentiment:.2f} "
            f"({len(self.all_scores)} utterances analyzed)"
        )

        return insights
