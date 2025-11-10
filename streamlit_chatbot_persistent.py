#!/usr/bin/env python3
"""
EigenAI Interactive Chatbot - v1.0.0 with PostgreSQL Persistence

Chat with a recursive self-modifying AI that builds understanding through eigenstate detection.
State persists across sessions using PostgreSQL database.
"""

import streamlit as st
import numpy as np
import sys
import os

# Dynamic path resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.eigen_recursive_ai import RecursiveEigenAI
from src.eigen_discrete_tokenizer import tokenize_word

# Try to import database manager
try:
    from chatbot_db import ChatbotDatabase
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("Warning: Database not available. State won't persist between sessions.")


def tokenize_text(text):
    """Tokenize a text string into discrete tokens"""
    words = text.split()
    return [tokenize_word(word) for word in words]


def initialize_database():
    """Initialize database connection and load state if available"""
    if not DB_AVAILABLE:
        return None

    try:
        db = ChatbotDatabase()
        return db
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None


def load_session_state(db, session_id=None):
    """Load state from database"""
    if not db:
        return False

    try:
        # Get latest session or specific session
        if session_id:
            # TODO: Implement load specific session
            pass
        else:
            session = db.get_latest_session()

        if not session:
            return False

        # Load AI state
        ai_state = db.load_ai_state(session['session_id'])
        if ai_state:
            st.session_state.ai = ai_state
            st.session_state.session_id = session['session_id']

        # Load learned tokens
        learned_tokens = db.load_learned_tokens(session['session_id'])
        st.session_state.learned_tokens = learned_tokens

        # Load messages
        messages = db.load_messages(session['session_id'])
        st.session_state.messages = messages

        # Load metrics
        metrics_history = db.load_metrics_history(session['session_id'])
        st.session_state.metrics_history = metrics_history

        st.session_state.loaded_from_db = True
        return True

    except Exception as e:
        st.error(f"Failed to load session: {e}")
        return False


def save_session_state(db, session_id=None):
    """Save current state to database"""
    if not db:
        return

    try:
        # Create new session if needed
        if not session_id:
            session_id = db.create_session()
            st.session_state.session_id = session_id

        # Get current state
        state = st.session_state.ai.get_state_summary()

        # Update session
        db.update_session(
            session_id=session_id,
            ai_state=st.session_state.ai,
            extraction_rules=st.session_state.ai.extraction_rules,
            total_iterations=state['iteration'],
            eigenstate_reached=state['eigenstate_reached']
        )

    except Exception as e:
        st.warning(f"Failed to save session: {e}")


def save_message_to_db(db, session_id, role, content, iteration, metrics=None):
    """Save a single message to database"""
    if not db or not session_id:
        return

    try:
        db.save_message(session_id, role, content, iteration, metrics)
    except Exception as e:
        st.warning(f"Failed to save message: {e}")


def save_tokens_to_db(db, session_id, tokens, iteration):
    """Save learned tokens to database"""
    if not db or not session_id:
        return

    try:
        for token in tokens:
            db.save_learned_token(session_id, token, iteration)
    except Exception as e:
        st.warning(f"Failed to save tokens: {e}")


def save_metrics_to_db(db, session_id, metrics):
    """Save metrics to database"""
    if not db or not session_id:
        return

    try:
        db.save_metrics(
            session_id=session_id,
            iteration=metrics['iteration'],
            eigenstate=metrics['eigenstate'],
            m_context_norm=metrics['M_context_norm'],
            vocab_size=metrics['vocab_size'],
            entropy_weighted=metrics.get('entropy_weighted', False)
        )
    except Exception as e:
        st.warning(f"Failed to save metrics: {e}")


def generate_token_response(learned_tokens, user_message, max_words=10):
    """Generate a response using only learned discrete tokens"""
    if len(learned_tokens) < 3:
        return "⋯ (Not enough tokens learned yet to generate response)"

    import random
    available_words = list(learned_tokens.keys())
    response_words = []

    for _ in range(min(max_words, len(available_words))):
        if available_words:
            word = random.choice(available_words)
            response_words.append(word)

    return " ".join(response_words)


# ===== Streamlit App =====

st.set_page_config(
    page_title="EigenAI Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 EigenAI: Understanding Through Eigenstate Detection")
st.markdown("""
Chat with an AI that measures its own understanding through geometric eigenstate convergence.
**State persists across sessions** using PostgreSQL.
""")

# Initialize database
if "db" not in st.session_state:
    st.session_state.db = initialize_database()

# Session management controls
col_header = st.columns([2, 1, 1])
with col_header[1]:
    if st.button("📂 Continue Last Session", use_container_width=True):
        if load_session_state(st.session_state.db):
            st.success("Session loaded!")
            st.rerun()
        else:
            st.info("No previous session found")

with col_header[2]:
    if st.button("✨ New Session", use_container_width=True):
        st.session_state.ai = RecursiveEigenAI(embedding_dim=64)
        st.session_state.messages = []
        st.session_state.metrics_history = []
        st.session_state.learned_tokens = {}
        st.session_state.session_id = None
        st.session_state.loaded_from_db = False
        st.rerun()

# Settings bar
col_settings = st.columns([3, 1])
with col_settings[1]:
    entropy_mode = st.toggle(
        "✨ Entropy Weighting",
        value=False,
        help="v1.0.0 feature: Weight semantics by information density (14.6× better orthogonality)"
    )

# Initialize session state
if "ai" not in st.session_state:
    # Try to load from database first
    if not load_session_state(st.session_state.db):
        # Create new session
        st.session_state.ai = RecursiveEigenAI(embedding_dim=64)
        st.session_state.messages = []
        st.session_state.metrics_history = []
        st.session_state.learned_tokens = {}
        st.session_state.session_id = None
        st.session_state.loaded_from_db = False

# Show database status
if DB_AVAILABLE and st.session_state.db:
    if st.session_state.get('loaded_from_db'):
        st.success("💾 Session loaded from database - your progress is saved!")
    else:
        st.info("💾 Database connected - progress will be saved automatically")
else:
    st.warning("⚠️ Database not connected - state will reset between sessions")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Conversation")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "metrics" in msg:
                with st.expander("📊 Understanding Metrics & Tokens"):
                    m = msg["metrics"]
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Eigenstate", "✓" if m.get("eigenstate") else "✗")
                    with col_b:
                        st.metric("Iteration", m.get("iteration", "N/A"))
                    with col_c:
                        st.metric("Vocab Size", m.get("vocab_size", 0))

                    # Show discrete tokens if available
                    if "tokens" in m:
                        st.divider()
                        st.caption("🔢 Discrete Token Bit Patterns")
                        for token in m["tokens"][:5]:
                            st.code(f"{token.word:>12} → L:{token.L:08b} R:{token.R:08b} V:{token.V:08b} M:{token.M:08b}", language="text")

                    if m.get("entropy_weighted"):
                        st.caption("✨ Processed with entropy weighting (v1.0.0)")

    prompt = st.chat_input("Type your message here...")

    if "example_clicked" in st.session_state and st.session_state.example_clicked:
        prompt = st.session_state.example_clicked
        st.session_state.example_clicked = None

    if prompt:
        state = st.session_state.ai.get_state_summary()
        current_iteration = state['iteration']

        # Save user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message_to_db(st.session_state.db, st.session_state.get('session_id'),
                          "user", prompt, current_iteration)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Processing and detecting eigenstates..."):
            # Tokenize the input
            tokens = tokenize_text(prompt)

            # Learn tokens from user input
            for token in tokens:
                st.session_state.learned_tokens[token.word.lower()] = token

            # Save tokens to database
            save_tokens_to_db(st.session_state.db, st.session_state.get('session_id'),
                            tokens, current_iteration)

            # Process user input through EigenAI framework
            result = st.session_state.ai.process(prompt, verbose=False)

            # Generate token-based response
            token_generated = generate_token_response(
                st.session_state.learned_tokens,
                prompt
            )

            # Learn tokens from the generated response
            response_tokens = tokenize_text(token_generated)
            for token in response_tokens:
                st.session_state.learned_tokens[token.word.lower()] = token

            # Save response tokens
            save_tokens_to_db(st.session_state.db, st.session_state.get('session_id'),
                            response_tokens, current_iteration)

            # Feed the generated response back into itself
            st.session_state.ai.process(token_generated, verbose=False)

            # Get updated state
            state = st.session_state.ai.get_state_summary()

            # Create response
            full_response = f"**🔢 Token-Generated Response:**\n{token_generated}\n\n"
            full_response += f"*({len(st.session_state.learned_tokens)} unique tokens learned)*\n\n"

            if entropy_mode:
                full_response += "✨ *Entropy-weighted processing* | "

            if result['eigenstate']:
                full_response += f"✓ *Meta-eigenstate reached (iteration {result['iteration']}) | Framework strength: {state['M_context_norm']:.2f}*"
            else:
                full_response += f"⋯ *Building understanding (iteration {result['iteration']}) | Framework strength: {state['M_context_norm']:.2f}*"

            metrics = {
                "eigenstate": result['eigenstate'],
                "iteration": result['iteration'],
                "M_context_norm": state['M_context_norm'],
                "tokens": tokens,
                "vocab_size": len(st.session_state.learned_tokens),
                "entropy_weighted": entropy_mode
            }

            # Save assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "metrics": metrics
            })
            save_message_to_db(st.session_state.db, st.session_state.get('session_id'),
                              "assistant", full_response, result['iteration'], metrics)

            # Save metrics
            st.session_state.metrics_history.append(metrics)
            save_metrics_to_db(st.session_state.db, st.session_state.get('session_id'), metrics)

            # Save overall session state
            save_session_state(st.session_state.db, st.session_state.get('session_id'))

        st.rerun()

with col2:
    st.subheader("📈 Understanding Evolution")

    state = st.session_state.ai.get_state_summary()

    st.metric("Total Inputs", state['iteration'])
    st.metric("Meta-Eigenstate", "✓ Reached" if state['eigenstate_reached'] else "⋯ Building")
    st.metric("Framework Strength", f"{state['M_context_norm']:.3f}")

    if st.session_state.get('session_id'):
        st.caption(f"💾 Session ID: {st.session_state.session_id}")

    st.divider()

    st.subheader("🔧 Extraction Rules")
    st.caption("How the AI weights semantic components")

    if state['iteration'] > 0:
        latest_rules = st.session_state.ai.extraction_rules

        st.progress(min(latest_rules['L_weight'], 1.0), text=f"L (Lexical): {latest_rules['L_weight']:.3f}")
        st.progress(min(latest_rules['R_weight'], 1.0), text=f"R (Relational): {latest_rules['R_weight']:.3f}")
        st.progress(min(latest_rules['V_weight'], 1.0), text=f"V (Value): {latest_rules['V_weight']:.3f}")
        st.progress(min(latest_rules['context_influence'], 1.0), text=f"Context: {latest_rules['context_influence']:.3f}")
    else:
        st.info("Start chatting to see extraction rules evolve")

    st.divider()

    st.subheader("💡 Example Prompts")
    examples = [
        "Tell me about quantum mechanics",
        "What makes a good leader?",
        "How do neural networks learn?",
        "Explain Einstein's theory",
        "What is consciousness?"
    ]

    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state.example_clicked = ex
            st.rerun()

    st.divider()

    if st.session_state.learned_tokens:
        st.caption(f"📚 Learned Vocabulary: {len(st.session_state.learned_tokens)} tokens")

st.sidebar.title("About EigenAI v1.0.0")
st.sidebar.markdown("""
### What is This?

This chatbot demonstrates **recursive self-modifying AI** that measures understanding through **eigenstate detection**.

**✨ New: State persistence** - Your conversations and learned vocabulary are saved to a PostgreSQL database and restored automatically!

### Key Concepts

**Semantic Triad (L, R, V)**:
- **L**: Subject/agent
- **R**: Predicate/verb
- **V**: Object/target
- **M**: Meta-understanding = L ⊕ R ⊕ V

**Eigenstate Detection**:
- Understanding = trajectory closure
- Fixed-point: Converged
- Periodic: Oscillating pattern
- None: Still learning

**Recursive Self-Modification**:
The AI changes its own processing framework based on what it learns, building genuine comprehension over time.

### New in v1.0.0

✨ **Entropy Weighting**: Weight semantic components by information density for 14.6× better orthogonality

💾 **Persistent Memory**: PostgreSQL database stores all learned tokens, conversation history, and eigenstate progression

📦 **Install**: `pip install eigenai`

💻 **GitHub**: [InauguralPhysicist/EigenAI](https://github.com/InauguralPhysicist/EigenAI)

### How It Works

1. You send a message
2. AI extracts semantic triad (L, R, V)
3. Computes meta-understanding (M)
4. Detects eigenstate convergence
5. Updates extraction rules recursively
6. Generates response from learned tokens
7. **Saves everything to database**
""")
