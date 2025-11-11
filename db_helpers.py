"""
Helper functions for loading and saving EigenAI chatbot state to PostgreSQL.
"""

import json
import numpy as np
from database import get_db, Session, Message, LearnedToken, AIState, init_database
from src.eigen_discrete_tokenizer import DiscreteToken

def ensure_session(session_id: str) -> int:
    """Ensure session exists and return session DB ID"""
    with get_db() as db:
        session = db.query(Session).filter_by(session_id=session_id).first()
        if not session:
            session = Session(session_id=session_id)
            db.add(session)
            db.flush()
        return session.id

def save_message(session_id: str, role: str, content: str, metrics: dict = None):
    """Save a chat message with optional metrics"""
    with get_db() as db:
        session_db_id = ensure_session(session_id)
        
        message = Message(
            session_id=session_db_id,
            role=role,
            content=content
        )
        
        if metrics:
            message.eigenstate = metrics.get('eigenstate', False)
            message.iteration = metrics.get('iteration', 0)
            message.m_context_norm = metrics.get('M_context_norm', 0.0)
            message.vocab_size = metrics.get('vocab_size', 0)
        
        db.add(message)

def load_messages(session_id: str) -> list:
    """Load all messages for a session"""
    with get_db() as db:
        session = db.query(Session).filter_by(session_id=session_id).first()
        if not session:
            return []
        
        messages = []
        for msg in session.messages:
            message_dict = {
                "role": msg.role,
                "content": msg.content
            }
            
            # Add metrics if this is an assistant message
            if msg.role == "assistant":
                message_dict["metrics"] = {
                    "eigenstate": msg.eigenstate,
                    "iteration": msg.iteration,
                    "M_context_norm": msg.m_context_norm,
                    "vocab_size": msg.vocab_size
                }
            
            messages.append(message_dict)
        
        return messages

def save_learned_token(session_id: str, token: DiscreteToken):
    """
    Save a learned discrete token with atomic increment (legacy single-token API)

    Note: Prefer save_learned_tokens_batch for better performance
    """
    with get_db() as db:
        session_db_id = ensure_session(session_id)

        # Check if token already exists for this session
        existing = db.query(LearnedToken).filter_by(
            session_id=session_db_id,
            word=token.word.lower()
        ).first()

        if existing:
            # Atomic increment using delta (prevents race conditions)
            existing.time_like_count += int(token.time_like_delta)
            existing.space_like_count += int(token.space_like_delta)
            existing.light_like_count += int(token.light_like_delta)
            existing.usage_count += int(token.usage_delta)
        else:
            # Create new token with initial counts (delta = total for new tokens)
            learned_token = LearnedToken(
                session_id=session_db_id,
                word=token.word.lower(),
                l_value=int(token.L),
                r_value=int(token.R),
                v_value=int(token.V),
                m_value=int(token.M),
                time_like_count=int(token.time_like_delta),
                space_like_count=int(token.space_like_delta),
                light_like_count=int(token.light_like_delta),
                usage_count=int(token.usage_delta)
            )
            db.add(learned_token)

def save_learned_tokens_batch(session_id: str, tokens: list):
    """
    Save multiple tokens in a single transaction with atomic increments

    CRITICAL for global mode: Uses atomic DB increments to prevent race conditions
    where concurrent users would lose each other's updates.

    Example race condition prevented:
    - User A loads token "quantum" (time_like_count=100)
    - User B loads token "quantum" (time_like_count=100)
    - User A increments to 101, saves → DB: 101
    - User B increments to 101, saves → DB: 101 (LOST UPDATE!)

    With atomic increments:
    - User A: DB += 1 → 101
    - User B: DB += 1 → 102 ✓

    Parameters
    ----------
    session_id : str
        Session identifier
    tokens : list of DiscreteToken
        Tokens to save (with geometric transition deltas)
    """
    if not tokens:
        return

    with get_db() as db:
        session_db_id = ensure_session(session_id)

        # Get all existing tokens for this session in one query
        word_list = [token.word.lower() for token in tokens]
        existing_tokens = db.query(LearnedToken).filter(
            LearnedToken.session_id == session_db_id,
            LearnedToken.word.in_(word_list)
        ).all()

        # Create lookup map for existing tokens
        existing_map = {token.word: token for token in existing_tokens}

        # Process all tokens
        for token in tokens:
            word_key = token.word.lower()

            if word_key in existing_map:
                # Atomic increment using delta (prevents race conditions)
                existing = existing_map[word_key]
                existing.time_like_count += int(token.time_like_delta)
                existing.space_like_count += int(token.space_like_delta)
                existing.light_like_count += int(token.light_like_delta)
                existing.usage_count += int(token.usage_delta)
            else:
                # Create new token with initial counts (delta = total for new tokens)
                learned_token = LearnedToken(
                    session_id=session_db_id,
                    word=word_key,
                    l_value=int(token.L),
                    r_value=int(token.R),
                    v_value=int(token.V),
                    m_value=int(token.M),
                    time_like_count=int(token.time_like_delta),
                    space_like_count=int(token.space_like_delta),
                    light_like_count=int(token.light_like_delta),
                    usage_count=int(token.usage_delta)
                )
                db.add(learned_token)

        # Single commit for all tokens (atomic at transaction level)

def load_learned_tokens(session_id: str) -> dict:
    """Load all learned tokens for a session with transition statistics"""
    with get_db() as db:
        session = db.query(Session).filter_by(session_id=session_id).first()
        if not session:
            return {}

        tokens = {}
        for token in session.tokens:
            # Reconstruct DiscreteToken object with transition statistics
            tokens[token.word] = DiscreteToken(
                word=token.word,
                L=token.l_value,
                R=token.r_value,
                V=token.v_value,
                M=token.m_value,
                time_like_count=token.time_like_count,
                space_like_count=token.space_like_count,
                light_like_count=token.light_like_count,
                usage_count=token.usage_count
            )

        return tokens

def load_tokens_by_classification(session_id: str, classification: str, limit: int = None) -> dict:
    """
    Load tokens filtered by geometric classification (database-side filtering)

    Critical for global mode: Don't load millions of tokens into memory

    Parameters
    ----------
    session_id : str
        Session identifier
    classification : str
        'time-like', 'space-like', 'light-like', or 'unknown'
    limit : int, optional
        Maximum number of tokens to return

    Returns
    -------
    tokens : dict
        {word: DiscreteToken} with only classified tokens
    """
    with get_db() as db:
        session = db.query(Session).filter_by(session_id=session_id).first()
        if not session:
            return {}

        # Build query based on classification using geometric criteria
        query = db.query(LearnedToken).filter(LearnedToken.session_id == session.id)

        if classification == 'time-like':
            # S > C: time_like_count is maximum
            query = query.filter(
                LearnedToken.time_like_count > LearnedToken.space_like_count,
                LearnedToken.time_like_count > LearnedToken.light_like_count,
                LearnedToken.usage_count > 0
            )
        elif classification == 'space-like':
            # C > S: space_like_count is maximum
            query = query.filter(
                LearnedToken.space_like_count > LearnedToken.time_like_count,
                LearnedToken.space_like_count > LearnedToken.light_like_count,
                LearnedToken.usage_count > 0
            )
        elif classification == 'light-like':
            # C = S: light_like_count is maximum
            query = query.filter(
                LearnedToken.light_like_count >= LearnedToken.time_like_count,
                LearnedToken.light_like_count >= LearnedToken.space_like_count,
                LearnedToken.usage_count > 0
            )
        elif classification == 'unknown':
            # No usage data yet
            query = query.filter(LearnedToken.usage_count == 0)
        else:
            return {}

        if limit:
            query = query.limit(limit)

        # Convert to DiscreteToken objects
        tokens = {}
        for token in query.all():
            tokens[token.word] = DiscreteToken(
                word=token.word,
                L=token.l_value,
                R=token.r_value,
                V=token.v_value,
                M=token.m_value,
                time_like_count=token.time_like_count,
                space_like_count=token.space_like_count,
                light_like_count=token.light_like_count,
                usage_count=token.usage_count
            )

        return tokens

def save_ai_state(session_id: str, ai):
    """Save the RecursiveEigenAI state"""
    with get_db() as db:
        session_db_id = ensure_session(session_id)
        
        # Get or create AI state
        ai_state = db.query(AIState).filter_by(session_id=session_db_id).first()
        if not ai_state:
            ai_state = AIState(session_id=session_db_id)
            db.add(ai_state)
        
        # Update extraction rules
        ai_state.l_weight = float(ai.extraction_rules['L_weight'])
        ai_state.r_weight = float(ai.extraction_rules['R_weight'])
        ai_state.v_weight = float(ai.extraction_rules['V_weight'])
        ai_state.context_influence = float(ai.extraction_rules['context_influence'])
        
        # Update state tracking
        state = ai.get_state_summary()
        ai_state.iteration_count = state['iteration']
        ai_state.eigenstate_reached = state['eigenstate_reached']
        ai_state.m_context_norm = float(state['M_context_norm'])
        
        # Save context embeddings as JSON (convert numpy arrays)
        ai_state.l_context_json = json.dumps(ai.L_context.tolist())
        ai_state.r_context_json = json.dumps(ai.R_context.tolist())
        ai_state.v_context_json = json.dumps(ai.V_context.tolist())
        ai_state.m_context_json = json.dumps(ai.M_context.tolist())

def load_ai_state(session_id: str, ai):
    """Load the RecursiveEigenAI state from database"""
    with get_db() as db:
        session = db.query(Session).filter_by(session_id=session_id).first()
        if not session or not session.ai_state:
            return
        
        ai_state = session.ai_state
        
        # Restore extraction rules
        ai.extraction_rules['L_weight'] = ai_state.l_weight
        ai.extraction_rules['R_weight'] = ai_state.r_weight
        ai.extraction_rules['V_weight'] = ai_state.v_weight
        ai.extraction_rules['context_influence'] = ai_state.context_influence
        
        # Restore iteration count
        ai.iteration = ai_state.iteration_count
        
        # Restore context embeddings
        if ai_state.l_context_json:
            ai.L_context = np.array(json.loads(ai_state.l_context_json))
        if ai_state.r_context_json:
            ai.R_context = np.array(json.loads(ai_state.r_context_json))
        if ai_state.v_context_json:
            ai.V_context = np.array(json.loads(ai_state.v_context_json))
        if ai_state.m_context_json:
            ai.M_context = np.array(json.loads(ai_state.m_context_json))

def get_session_stats(session_id: str) -> dict:
    """Get statistics for a session"""
    with get_db() as db:
        session = db.query(Session).filter_by(session_id=session_id).first()
        if not session:
            return {
                "message_count": 0,
                "token_count": 0,
                "created_at": None
            }
        
        return {
            "message_count": len(session.messages),
            "token_count": len(session.tokens),
            "created_at": session.created_at
        }
