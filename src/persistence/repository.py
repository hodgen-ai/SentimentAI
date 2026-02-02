"""Data access layer for Vibemaxxing."""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
import logging
from pathlib import Path

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session as DBSession
import numpy as np

from .models import Base, Speaker, Session, Utterance, DailySummary, EmotionType


logger = logging.getLogger(__name__)


class SentimentRepository:
    """
    Data access layer for sentiment data.

    Handles all database operations including CRUD, queries, and aggregations.
    """

    def __init__(self, db_path: str = "data/vibemaxxing.db"):
        """
        Initialize repository.

        Args:
            db_path: Path to SQLite database file
        """
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

        logger.info(f"Repository initialized: {db_path}")

    @contextmanager
    def get_session(self):
        """Get a database session with automatic commit/rollback."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # Session operations

    def create_session(self) -> int:
        """Create a new monitoring session."""
        with self.get_session() as db:
            session = Session()
            db.add(session)
            db.flush()
            session_id = session.id
            logger.info(f"Created session {session_id}")
            return session_id

    def end_session(self, session_id: int) -> None:
        """Mark a session as ended."""
        with self.get_session() as db:
            session = db.query(Session).filter_by(id=session_id).first()
            if session:
                session.ended_at = datetime.utcnow()
                if session.started_at:
                    session.duration_seconds = (
                        session.ended_at - session.started_at
                    ).total_seconds()

                # Calculate session stats
                utterances = db.query(Utterance).filter_by(session_id=session_id).all()
                session.total_utterances = len(utterances)
                if utterances:
                    session.avg_sentiment = np.mean([u.sentiment for u in utterances])

                logger.info(f"Ended session {session_id}")

    def get_session(self, session_id: int) -> Optional[Session]:
        """Get a session by ID."""
        with self.get_session() as db:
            return db.query(Session).filter_by(id=session_id).first()

    # Speaker operations

    def get_or_create_speaker(self, speaker_id: str) -> int:
        """Get or create a speaker by their ID."""
        with self.get_session() as db:
            speaker = db.query(Speaker).filter_by(speaker_id=speaker_id).first()
            if not speaker:
                speaker = Speaker(speaker_id=speaker_id)
                db.add(speaker)
                db.flush()
                logger.debug(f"Created speaker: {speaker_id}")
            else:
                speaker.last_seen = datetime.utcnow()
            return speaker.id

    def get_all_speakers(self) -> List[Dict[str, Any]]:
        """Get all speakers with stats."""
        with self.get_session() as db:
            speakers = db.query(Speaker).all()
            result = []
            for s in speakers:
                utterance_count = db.query(Utterance).filter_by(speaker_id=s.id).count()
                result.append({
                    "id": s.id,
                    "speaker_id": s.speaker_id,
                    "display_name": s.display_name,
                    "first_seen": s.first_seen,
                    "last_seen": s.last_seen,
                    "utterance_count": utterance_count
                })
            return result

    def set_speaker_name(self, speaker_id: str, display_name: str) -> None:
        """Set a display name for a speaker."""
        with self.get_session() as db:
            speaker = db.query(Speaker).filter_by(speaker_id=speaker_id).first()
            if speaker:
                speaker.display_name = display_name

    # Utterance operations

    def add_utterance(
        self,
        session_id: int,
        text: str,
        start_time: float,
        end_time: float,
        sentiment: float,
        emotion: str,
        confidence: float,
        speaker_id: Optional[str] = None
    ) -> int:
        """Add a new utterance."""
        with self.get_session() as db:
            # Get or create speaker
            db_speaker_id = None
            if speaker_id:
                db_speaker_id = self.get_or_create_speaker(speaker_id)

            # Map emotion string to enum
            try:
                emotion_enum = EmotionType(emotion.lower())
            except ValueError:
                emotion_enum = EmotionType.NEUTRAL

            utterance = Utterance(
                session_id=session_id,
                speaker_id=db_speaker_id,
                text=text,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                sentiment=sentiment,
                emotion=emotion_enum,
                confidence=confidence
            )
            db.add(utterance)
            db.flush()
            return utterance.id

    def get_recent_utterances(
        self,
        limit: int = 100,
        speaker_id: Optional[str] = None,
        session_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get recent utterances."""
        with self.get_session() as db:
            query = db.query(Utterance).order_by(Utterance.created_at.desc())

            if speaker_id:
                speaker = db.query(Speaker).filter_by(speaker_id=speaker_id).first()
                if speaker:
                    query = query.filter(Utterance.speaker_id == speaker.id)

            if session_id:
                query = query.filter(Utterance.session_id == session_id)

            utterances = query.limit(limit).all()

            return [
                {
                    "id": u.id,
                    "text": u.text,
                    "sentiment": u.sentiment,
                    "emotion": u.emotion.value,
                    "confidence": u.confidence,
                    "speaker_id": u.speaker.speaker_id if u.speaker else None,
                    "created_at": u.created_at
                }
                for u in utterances
            ]

    # Statistics

    def get_speaker_stats(self, speaker_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific speaker."""
        with self.get_session() as db:
            speaker = db.query(Speaker).filter_by(speaker_id=speaker_id).first()
            if not speaker:
                return None

            utterances = db.query(Utterance).filter_by(speaker_id=speaker.id).all()
            if not utterances:
                return {
                    "speaker_id": speaker_id,
                    "utterance_count": 0
                }

            sentiments = [u.sentiment for u in utterances]
            emotions = [u.emotion for u in utterances]

            return {
                "speaker_id": speaker_id,
                "display_name": speaker.display_name,
                "utterance_count": len(utterances),
                "avg_sentiment": float(np.mean(sentiments)),
                "std_sentiment": float(np.std(sentiments)) if len(sentiments) > 1 else 0,
                "min_sentiment": float(min(sentiments)),
                "max_sentiment": float(max(sentiments)),
                "primary_emotion": max(set(emotions), key=emotions.count).value,
                "positivity_ratio": sum(1 for s in sentiments if s > 0.3) / len(sentiments),
                "first_seen": speaker.first_seen,
                "last_seen": speaker.last_seen
            }

    def get_global_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get global statistics for the past N days."""
        with self.get_session() as db:
            cutoff = datetime.utcnow() - timedelta(days=days)

            utterances = (
                db.query(Utterance)
                .filter(Utterance.created_at >= cutoff)
                .all()
            )

            if not utterances:
                return {
                    "period_days": days,
                    "utterance_count": 0
                }

            sentiments = [u.sentiment for u in utterances]
            speakers = set(u.speaker_id for u in utterances if u.speaker_id)
            sessions = set(u.session_id for u in utterances)

            return {
                "period_days": days,
                "utterance_count": len(utterances),
                "speaker_count": len(speakers),
                "session_count": len(sessions),
                "avg_sentiment": float(np.mean(sentiments)),
                "std_sentiment": float(np.std(sentiments)) if len(sentiments) > 1 else 0,
                "min_sentiment": float(min(sentiments)),
                "max_sentiment": float(max(sentiments)),
                "positivity_ratio": sum(1 for s in sentiments if s > 0.3) / len(sentiments)
            }

    # Daily summaries

    def update_daily_summary(self, date: datetime = None) -> None:
        """Update or create daily summary for a given date."""
        if date is None:
            date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            date = date.replace(hour=0, minute=0, second=0, microsecond=0)

        with self.get_session() as db:
            # Get all utterances for the day
            next_day = date + timedelta(days=1)
            utterances = (
                db.query(Utterance)
                .filter(Utterance.created_at >= date)
                .filter(Utterance.created_at < next_day)
                .all()
            )

            # Get or create summary
            summary = db.query(DailySummary).filter_by(date=date).first()
            if not summary:
                summary = DailySummary(date=date)
                db.add(summary)

            if utterances:
                sentiments = [u.sentiment for u in utterances]
                emotions = [u.emotion for u in utterances]
                speakers = set(u.speaker_id for u in utterances if u.speaker_id)
                sessions = set(u.session_id for u in utterances)

                summary.utterance_count = len(utterances)
                summary.speaker_count = len(speakers)
                summary.session_count = len(sessions)
                summary.avg_sentiment = float(np.mean(sentiments))
                summary.std_sentiment = float(np.std(sentiments)) if len(sentiments) > 1 else 0
                summary.min_sentiment = float(min(sentiments))
                summary.max_sentiment = float(max(sentiments))
                summary.primary_emotion = max(set(emotions), key=emotions.count)
                summary.positive_ratio = sum(1 for s in sentiments if s > 0.3) / len(sentiments)

    def get_daily_summaries(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get daily summaries for the past N days."""
        with self.get_session() as db:
            cutoff = datetime.utcnow() - timedelta(days=days)
            summaries = (
                db.query(DailySummary)
                .filter(DailySummary.date >= cutoff)
                .order_by(DailySummary.date.desc())
                .all()
            )

            return [
                {
                    "date": s.date,
                    "avg_sentiment": s.avg_sentiment,
                    "utterance_count": s.utterance_count,
                    "speaker_count": s.speaker_count,
                    "primary_emotion": s.primary_emotion.value if s.primary_emotion else None,
                    "positive_ratio": s.positive_ratio
                }
                for s in summaries
            ]

    # Cleanup

    def cleanup_old_data(self, days: int = 30) -> int:
        """Remove data older than specified days."""
        with self.get_session() as db:
            cutoff = datetime.utcnow() - timedelta(days=days)

            # Delete old utterances
            deleted = db.query(Utterance).filter(Utterance.created_at < cutoff).delete()

            # Delete old sessions
            db.query(Session).filter(Session.started_at < cutoff).delete()

            # Delete old daily summaries
            db.query(DailySummary).filter(DailySummary.date < cutoff).delete()

            logger.info(f"Cleaned up {deleted} utterances older than {days} days")
            return deleted

    # Export

    def export_to_csv(self, output_path: str, days: int = 30) -> None:
        """Export utterances to CSV."""
        import csv

        with self.get_session() as db:
            cutoff = datetime.utcnow() - timedelta(days=days)
            utterances = (
                db.query(Utterance)
                .filter(Utterance.created_at >= cutoff)
                .order_by(Utterance.created_at)
                .all()
            )

            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "speaker_id", "text", "sentiment",
                    "emotion", "confidence"
                ])

                for u in utterances:
                    writer.writerow([
                        u.created_at.isoformat(),
                        u.speaker.speaker_id if u.speaker else "",
                        u.text,
                        u.sentiment,
                        u.emotion.value,
                        u.confidence
                    ])

        logger.info(f"Exported {len(utterances)} utterances to {output_path}")
