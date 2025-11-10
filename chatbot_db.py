#!/usr/bin/env python3
"""
EigenAI Chatbot Database Manager

Handles PostgreSQL persistence for chatbot state, learned tokens,
conversation history, and metrics across sessions.
"""

import os
import json
import pickle
import base64
from typing import Dict, List, Optional, Tuple
from datetime import datetime

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    print("Warning: psycopg2 not available. Install with: pip install psycopg2-binary")


class ChatbotDatabase:
    """Manages persistent storage for EigenAI chatbot state"""

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database connection

        Args:
            database_url: PostgreSQL connection URL. If None, uses REPLIT_DB_URL env var
        """
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 is required for database persistence")

        self.database_url = database_url or os.getenv('DATABASE_URL') or os.getenv('REPLIT_DB_URL')
        if not self.database_url:
            raise ValueError("Database URL not provided. Set DATABASE_URL environment variable.")

        self.conn = None
        self._connect()
        self._initialize_schema()

    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(self.database_url)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {e}")

    def _initialize_schema(self):
        """Create tables if they don't exist"""
        schema_file = os.path.join(os.path.dirname(__file__), 'db_schema.sql')

        if os.path.exists(schema_file):
            with open(schema_file, 'r') as f:
                schema_sql = f.read()

            with self.conn.cursor() as cur:
                cur.execute(schema_sql)
            self.conn.commit()

    def create_session(self) -> int:
        """
        Create a new chatbot session

        Returns:
            session_id: ID of the newly created session
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chatbot_sessions (created_at, last_updated)
                VALUES (NOW(), NOW())
                RETURNING session_id
            """)
            session_id = cur.fetchone()[0]
        self.conn.commit()
        return session_id

    def get_latest_session(self) -> Optional[Dict]:
        """
        Get the most recent session

        Returns:
            Session data dict or None if no sessions exist
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM latest_session")
            result = cur.fetchone()
            return dict(result) if result else None

    def update_session(self, session_id: int, ai_state: object,
                      extraction_rules: Dict, total_iterations: int,
                      eigenstate_reached: bool = False):
        """
        Update session state

        Args:
            session_id: Session to update
            ai_state: RecursiveEigenAI object (will be pickled)
            extraction_rules: Dict of L, R, V, Context weights
            total_iterations: Current iteration count
            eigenstate_reached: Whether eigenstate has been reached
        """
        # Serialize AI state using pickle + base64
        ai_state_bytes = pickle.dumps(ai_state)
        ai_state_b64 = base64.b64encode(ai_state_bytes).decode('utf-8')

        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE chatbot_sessions
                SET last_updated = NOW(),
                    total_iterations = %s,
                    eigenstate_reached = %s,
                    ai_state = %s,
                    extraction_rules = %s
                WHERE session_id = %s
            """, (total_iterations, eigenstate_reached,
                  Json({'pickled': ai_state_b64}),
                  Json(extraction_rules), session_id))
        self.conn.commit()

    def load_ai_state(self, session_id: int) -> Optional[object]:
        """
        Load RecursiveEigenAI object from database

        Args:
            session_id: Session to load from

        Returns:
            RecursiveEigenAI object or None
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT ai_state FROM chatbot_sessions
                WHERE session_id = %s
            """, (session_id,))
            result = cur.fetchone()

            if result and result['ai_state']:
                ai_state_b64 = result['ai_state'].get('pickled')
                if ai_state_b64:
                    ai_state_bytes = base64.b64decode(ai_state_b64)
                    return pickle.loads(ai_state_bytes)
        return None

    def save_learned_token(self, session_id: int, token: object, iteration: int):
        """
        Save a learned token

        Args:
            session_id: Current session
            token: DiscreteToken object with word, L, R, V, M attributes
            iteration: Iteration when token was learned
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO learned_tokens
                (session_id, word, l_bits, r_bits, v_bits, m_bits, learned_at_iteration)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (session_id, word) DO NOTHING
            """, (session_id, token.word, token.L, token.R, token.V, token.M, iteration))
        self.conn.commit()

    def load_learned_tokens(self, session_id: int) -> Dict:
        """
        Load all learned tokens for a session

        Args:
            session_id: Session to load from

        Returns:
            Dict mapping word -> DiscreteToken object
        """
        from src.eigen_discrete_tokenizer import DiscreteToken

        tokens = {}
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT word, l_bits, r_bits, v_bits, m_bits
                FROM learned_tokens
                WHERE session_id = %s
            """, (session_id,))

            for row in cur.fetchall():
                token = DiscreteToken(
                    word=row['word'],
                    L=row['l_bits'],
                    R=row['r_bits'],
                    V=row['v_bits'],
                    M=row['m_bits']
                )
                tokens[row['word']] = token

        return tokens

    def save_message(self, session_id: int, role: str, content: str,
                    iteration: int, metrics: Optional[Dict] = None):
        """
        Save a conversation message

        Args:
            session_id: Current session
            role: 'user' or 'assistant'
            content: Message text
            iteration: Current iteration
            metrics: Optional metrics dict
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO messages (session_id, role, content, iteration, metrics)
                VALUES (%s, %s, %s, %s, %s)
            """, (session_id, role, content, iteration, Json(metrics) if metrics else None))
        self.conn.commit()

    def load_messages(self, session_id: int) -> List[Dict]:
        """
        Load conversation history

        Args:
            session_id: Session to load from

        Returns:
            List of message dicts
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT role, content, metrics, created_at
                FROM messages
                WHERE session_id = %s
                ORDER BY created_at ASC
            """, (session_id,))

            return [dict(row) for row in cur.fetchall()]

    def save_metrics(self, session_id: int, iteration: int, eigenstate: bool,
                    m_context_norm: float, vocab_size: int,
                    entropy_weighted: bool = False):
        """
        Save metrics for an iteration

        Args:
            session_id: Current session
            iteration: Current iteration
            eigenstate: Whether eigenstate was reached
            m_context_norm: Framework strength
            vocab_size: Number of learned tokens
            entropy_weighted: Whether entropy weighting was used
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO metrics_history
                (session_id, iteration, eigenstate, m_context_norm, vocab_size, entropy_weighted)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (session_id, iteration, eigenstate, m_context_norm, vocab_size, entropy_weighted))
        self.conn.commit()

    def load_metrics_history(self, session_id: int) -> List[Dict]:
        """
        Load metrics history

        Args:
            session_id: Session to load from

        Returns:
            List of metrics dicts
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT iteration, eigenstate, m_context_norm, vocab_size, entropy_weighted
                FROM metrics_history
                WHERE session_id = %s
                ORDER BY iteration ASC
            """, (session_id,))

            return [dict(row) for row in cur.fetchall()]

    def delete_session(self, session_id: int):
        """
        Delete a session and all associated data

        Args:
            session_id: Session to delete
        """
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM chatbot_sessions WHERE session_id = %s", (session_id,))
        self.conn.commit()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
