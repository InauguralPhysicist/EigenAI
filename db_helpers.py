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
    """Save a learned discrete token with transition statistics"""
    with get_db() as db:
        session_db_id = ensure_session(session_id)

        # Check if token already exists for this session
        existing = db.query(LearnedToken).filter_by(
            session_id=session_db_id,
            word=token.word.lower()
        ).first()

        if existing:
            # Update existing token's transition statistics
            existing.time_like_count = int(token.time_like_count)
            existing.space_like_count = int(token.space_like_count)
            existing.light_like_count = int(token.light_like_count)
            existing.usage_count = int(token.usage_count)
        else:
            # Create new token with all attributes
            learned_token = LearnedToken(
                session_id=session_db_id,
                word=token.word.lower(),
                l_value=int(token.L),
                r_value=int(token.R),
                v_value=int(token.V),
                m_value=int(token.M),
                time_like_count=int(token.time_like_count),
                space_like_count=int(token.space_like_count),
                light_like_count=int(token.light_like_count),
                usage_count=int(token.usage_count)
            )
            db.add(learned_token)

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
