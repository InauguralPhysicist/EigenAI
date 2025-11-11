"""
Database models and utilities for EigenAI chatbot persistence.

Stores:
- Learned tokens (discrete token vocabulary)
- Conversation sessions
- Messages and metrics
"""

import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from contextlib import contextmanager

Base = declarative_base()

class Session(Base):
    """Conversation session tracking"""
    __tablename__ = 'sessions'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    tokens = relationship("LearnedToken", back_populates="session", cascade="all, delete-orphan")
    ai_state = relationship("AIState", back_populates="session", uselist=False, cascade="all, delete-orphan")

class Message(Base):
    """Chat messages with metrics"""
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('sessions.id'), nullable=False)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Metrics (JSON stored as text)
    eigenstate = Column(Boolean, default=False)
    iteration = Column(Integer, default=0)
    m_context_norm = Column(Float, default=0.0)
    vocab_size = Column(Integer, default=0)
    
    session = relationship("Session", back_populates="messages")

class LearnedToken(Base):
    """Discrete tokens learned during conversation with transition statistics"""
    __tablename__ = 'learned_tokens'

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('sessions.id'), nullable=False)
    word = Column(String(255), nullable=False, index=True)
    l_value = Column(Integer, nullable=False)  # Lexical (0-255)
    r_value = Column(Integer, nullable=False)  # Relational (0-255)
    v_value = Column(Integer, nullable=False)  # Value (0-255)
    m_value = Column(Integer, nullable=False)  # Meta (0-255)

    # Geometric classification statistics (from ds² = S² - C²)
    time_like_count = Column(Integer, default=0)  # S > C: structural/sequential
    space_like_count = Column(Integer, default=0)  # C > S: semantic/content
    light_like_count = Column(Integer, default=0)  # C = S: relational/transformational
    usage_count = Column(Integer, default=0)  # Total usage

    learned_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    session = relationship("Session", back_populates="tokens")

    # Indices for efficient classification queries (CRITICAL for global mode at scale)
    __table_args__ = (
        # Composite index for time-like token queries
        Index('idx_time_like', 'session_id', 'time_like_count', 'space_like_count', 'light_like_count', 'usage_count'),
        # Composite index for space-like token queries
        Index('idx_space_like', 'session_id', 'space_like_count', 'time_like_count', 'light_like_count', 'usage_count'),
        # Composite index for light-like token queries
        Index('idx_light_like', 'session_id', 'light_like_count', 'time_like_count', 'space_like_count', 'usage_count'),
        # Index for session+word lookups (checking if token exists)
        Index('idx_session_word', 'session_id', 'word'),
    )

class AIState(Base):
    """Recursive AI state (framework parameters)"""
    __tablename__ = 'ai_states'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('sessions.id'), nullable=False)
    
    # Extraction rules
    l_weight = Column(Float, default=1.0)
    r_weight = Column(Float, default=1.0)
    v_weight = Column(Float, default=1.0)
    context_influence = Column(Float, default=0.1)
    
    # State tracking
    iteration_count = Column(Integer, default=0)
    eigenstate_reached = Column(Boolean, default=False)
    m_context_norm = Column(Float, default=0.0)
    
    # Context embeddings (stored as JSON)
    l_context_json = Column(Text, nullable=True)
    r_context_json = Column(Text, nullable=True)
    v_context_json = Column(Text, nullable=True)
    m_context_json = Column(Text, nullable=True)
    
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    session = relationship("Session", back_populates="ai_state")

# Database connection
def get_database_url():
    """Get database URL from environment"""
    return os.environ.get('DATABASE_URL')

def create_db_engine():
    """
    Create SQLAlchemy engine with SSL and optimized connection pooling

    Pool settings optimized for community-scale global mode:
    - Handles concurrent users efficiently
    - Prevents connection exhaustion
    - Recycles stale connections
    """
    url = get_database_url()
    if not url:
        raise ValueError("DATABASE_URL not found in environment variables")

    # Configure connection args for PostgreSQL with SSL
    connect_args = {}
    if 'postgresql' in url or 'postgres' in url:
        connect_args = {
            'sslmode': 'require',
            'connect_timeout': 10
        }

    return create_engine(
        url,
        echo=False,
        connect_args=connect_args,
        # Connection pool settings for multi-user deployment
        pool_size=20,           # Base connections (up from default 5)
        max_overflow=40,        # Additional connections under load (up from 10)
        pool_timeout=30,        # Wait time for connection (default 30)
        pool_pre_ping=True,     # Verify connections before using
        pool_recycle=3600,      # Recycle connections after 1 hour
        pool_use_lifo=True      # LIFO reuses recent connections (better for pooling)
    )

# Create session maker
engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_database():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

@contextmanager
def get_db():
    """Get database session with automatic cleanup"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
