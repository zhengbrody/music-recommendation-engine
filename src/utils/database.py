"""
Database utilities using SQLAlchemy.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional
from datetime import datetime
from config.config import Config

Base = declarative_base()


class User(Base):
    """User table model."""
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    user_id = Column(String, unique=True, nullable=False)
    user_idx = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_recommendation_at = Column(DateTime)


class RecommendationLog(Base):
    """Recommendation log table model."""
    __tablename__ = 'recommendation_logs'

    id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=False)
    artist_idx = Column(Integer, nullable=False)
    score = Column(Float, nullable=False)
    model = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class SearchLog(Base):
    """Search log table model."""
    __tablename__ = 'search_logs'

    id = Column(Integer, primary_key=True)
    query = Column(String, nullable=False)
    search_type = Column(String, nullable=False)  # 'artist', 'user', 'popularity'
    n_results = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """Manage database connections and operations."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize database manager.

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or Config()

        # Create database URL
        db_url = (
            f"postgresql://{self.config.POSTGRES_USER}:{self.config.POSTGRES_PASSWORD}"
            f"@{self.config.POSTGRES_HOST}:{self.config.POSTGRES_PORT}/{self.config.POSTGRES_DB}"
        )

        try:
            self.engine = create_engine(db_url, echo=False)
            self.SessionLocal = sessionmaker(bind=self.engine)
            self.enabled = True
            print("Database connection configured successfully")
        except Exception as e:
            print(f"Database connection failed: {e}. Database features disabled.")
            self.enabled = False
            self.engine = None
            self.SessionLocal = None

    def create_tables(self) -> bool:
        """
        Create all tables if they don't exist.

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            Base.metadata.create_all(self.engine)
            print("Database tables created successfully")
            return True
        except Exception as e:
            print(f"Table creation error: {e}")
            return False

    def get_session(self) -> Optional[Session]:
        """
        Get a database session.

        Returns:
            SQLAlchemy session or None if disabled
        """
        if not self.enabled:
            return None

        return self.SessionLocal()

    def log_recommendation(
        self,
        user_id: str,
        artist_idx: int,
        score: float,
        model: str
    ) -> bool:
        """
        Log a recommendation to database.

        Args:
            user_id: User ID
            artist_idx: Artist index
            score: Recommendation score
            model: Model name

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            session = self.get_session()
            log = RecommendationLog(
                user_id=user_id,
                artist_idx=artist_idx,
                score=score,
                model=model
            )
            session.add(log)
            session.commit()
            session.close()
            return True
        except Exception as e:
            print(f"Recommendation logging error: {e}")
            return False

    def log_search(
        self,
        query: str,
        search_type: str,
        n_results: int
    ) -> bool:
        """
        Log a search query to database.

        Args:
            query: Search query
            search_type: Type of search
            n_results: Number of results

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            session = self.get_session()
            log = SearchLog(
                query=query,
                search_type=search_type,
                n_results=n_results
            )
            session.add(log)
            session.commit()
            session.close()
            return True
        except Exception as e:
            print(f"Search logging error: {e}")
            return False
