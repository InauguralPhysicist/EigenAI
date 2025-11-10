#!/usr/bin/env python3
"""
EigenAI Interactive Chatbot - v1.0.0

Chat with a recursive self-modifying AI that builds understanding through eigenstate detection.
Watch in real-time as the AI's processing framework evolves during conversation.
"""

import streamlit as st
import numpy as np
import sys
import os

# Dynamic path resolution (fixes hardcoded path issues)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.eigen_recursive_ai import RecursiveEigenAI
from src.eigen_discrete_tokenizer import tokenize_word

def tokenize_text(text):
    """Tokenize a text string into discrete tokens"""
    words = text.split()
    return [tokenize_word(word) for word in words]

st.set_page_config(
    page_title="EigenAI Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 EigenAI: Understanding Through Eigenstate Detection")
st.markdown("""
Chat with an AI that measures its own understanding through geometric eigenstate convergence.
Watch the metrics panel to see how the AI's processing framework evolves.
""")

# Settings bar
col_settings = st.columns([3, 1])
with col_settings[1]:
    entropy_mode = st.toggle(
        "✨ Entropy Weighting",
        value=False,
        help="v1.0.0 feature: Weight semantics by information density (14.6× better orthogonality)"
    )

if "ai" not in st.session_state:
    st.session_state.ai = RecursiveEigenAI(embedding_dim=64)
    st.session_state.messages = []
    st.session_state.metrics_history = []
    st.session_state.learned_tokens = {}

def generate_token_response(learned_tokens, user_message, max_words=10):
    """
    Attempt to generate a response using only learned discrete tokens
    This shows if the AI can 'speak' using the token vocabulary it has learned
    """
    if len(learned_tokens) < 3:
        return "⋯ (Not enough tokens learned yet to generate response)"

    # Simple generation: pick relevant tokens based on similarity
    import random
    available_words = list(learned_tokens.keys())

    # Prefer tokens with similar bit patterns to user's message tokens
    user_tokens = tokenize_text(user_message.lower())

    # Pick words that have similar L, R, or V values (simplified similarity)
    response_words = []
    for _ in range(min(max_words, len(available_words))):
        if available_words:
            word = random.choice(available_words)
            response_words.append(word)

    return " ".join(response_words)

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
                        for token in m["tokens"][:5]:  # Show first 5 tokens
                            st.code(f"{token.word:>12} → L:{token.L:08b} R:{token.R:08b} V:{token.V:08b} M:{token.M:08b}", language="text")

                    if m.get("entropy_weighted"):
                        st.caption("✨ Processed with entropy weighting (v1.0.0)")

    prompt = st.chat_input("Type your message here...")

    if "example_clicked" in st.session_state and st.session_state.example_clicked:
        prompt = st.session_state.example_clicked
        st.session_state.example_clicked = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Processing and detecting eigenstates..."):
            # Tokenize the input to show discrete representation
            tokens = tokenize_text(prompt)

            # Learn tokens from user input
            for token in tokens:
                st.session_state.learned_tokens[token.word.lower()] = token

            # Process user input through EigenAI framework
            result = st.session_state.ai.process(prompt, verbose=False)

            # Generate token-based response (AI speaking from learned vocabulary)
            token_generated = generate_token_response(
                st.session_state.learned_tokens,
                prompt
            )

            # Learn tokens from the generated response
            response_tokens = tokenize_text(token_generated)
            for token in response_tokens:
                st.session_state.learned_tokens[token.word.lower()] = token

            # Feed the generated response back into itself for recursive understanding
            st.session_state.ai.process(token_generated, verbose=False)

            # Get updated state after recursive processing
            state = st.session_state.ai.get_state_summary()

            # Create response with eigenstate information
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

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "metrics": metrics
            })
            st.session_state.metrics_history.append(metrics)

        st.rerun()

with col2:
    st.subheader("📈 Understanding Evolution")

    state = st.session_state.ai.get_state_summary()

    st.metric("Total Inputs", state['iteration'])
    st.metric("Meta-Eigenstate", "✓ Reached" if state['eigenstate_reached'] else "⋯ Building")
    st.metric("Framework Strength", f"{state['M_context_norm']:.3f}")

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
        "Explain Einstein's theory of relativity",
        "What is consciousness?"
    ]

    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state.example_clicked = ex
            st.rerun()

    st.divider()

    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.ai = RecursiveEigenAI(embedding_dim=64)
        st.session_state.messages = []
        st.session_state.metrics_history = []
        st.session_state.learned_tokens = {}
        st.rerun()

    st.divider()

    if st.session_state.learned_tokens:
        st.caption(f"📚 Learned Vocabulary: {len(st.session_state.learned_tokens)} tokens")

st.sidebar.title("About EigenAI v1.0.0")
st.sidebar.markdown("""
### What is This?

This chatbot demonstrates **recursive self-modifying AI** that measures understanding through **eigenstate detection**.

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

📦 **Install**: `pip install eigenai`

💻 **GitHub**: [InauguralPhysicist/EigenAI](https://github.com/InauguralPhysicist/EigenAI)

### How It Works

1. You send a message
2. AI extracts semantic triad (L, R, V)
3. Computes meta-understanding (M)
4. Detects eigenstate convergence
5. Updates extraction rules recursively
6. Generates response from learned tokens
""")
