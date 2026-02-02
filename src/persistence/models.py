"""SQLAlchemy ORM models for Vibemaxxing."""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, DateTime, ForeignKey, Text, Enum as SQLEnum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum


Base = declarative_base()


class EmotionType(enum.Enum):
    """Emotion types matching the analysis module."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"


class Speaker(Base):
    """Represents a unique speaker identified by diarization."""
    __tablename__ = "speakers"

    id = Column(Integer, primary_key=True)
    speaker_id = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(200), nullable=True)
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_user = Column(Integer, default=0)  # 1 if this is the user themselves

    utterances = relationship("Utterance", back_populates="speaker")

    def __repr__(self):
        return f"<Speaker(id={self.id}, speaker_id='{self.speaker_id}')>"


class Session(Base):
    """Represents a monitoring session."""
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    total_utterances = Column(Integer, default=0)
    avg_sentiment = Column(Float, nullable=True)

    utterances = relationship("Utterance", back_populates="session")

    def __repr__(self):
        return f"<Session(id={self.id}, started_at={self.started_at})>"


class Utterance(Base):
    """A single analyzed utterance."""
    __tablename__ = "utterances"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False, index=True)
    speaker_id = Column(Integer, ForeignKey("speakers.id"), nullable=True, index=True)

    text = Column(Text, nullable=False)
    start_time = Column(Float, nullable=False)  # Seconds from session start
    end_time = Column(Float, nullable=False)
    duration = Column(Float, nullable=True)

    sentiment = Column(Float, nullable=False)  # -1.0 to +1.0
    emotion = Column(SQLEnum(EmotionType), nullable=False, default=EmotionType.NEUTRAL)
    confidence = Column(Float, nullable=False, default=0.5)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    speaker = relationship("Speaker", back_populates="utterances")
    session = relationship("Session", back_populates="utterances")

    def __repr__(self):
        return f"<Utterance(id={self.id}, sentiment={self.sentiment:.2f})>"


class DailySummary(Base):
    """Daily aggregated statistics."""
    __tablename__ = "daily_summaries"

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False, unique=True, index=True)

    avg_sentiment = Column(Float, nullable=True)
    min_sentiment = Column(Float, nullable=True)
    max_sentiment = Column(Float, nullable=True)
    std_sentiment = Column(Float, nullable=True)

    utterance_count = Column(Integer, default=0)
    speaker_count = Column(Integer, default=0)
    session_count = Column(Integer, default=0)

    primary_emotion = Column(SQLEnum(EmotionType), nullable=True)
    positive_ratio = Column(Float, nullable=True)  # % of positive utterances

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<DailySummary(date={self.date}, avg_sentiment={self.avg_sentiment})>"
