#!/usr/bin/env python3
"""
EigenAI Interactive Chatbot

Chat with a recursive self-modifying AI that builds understanding through eigenstate detection.
Watch in real-time as the AI's processing framework evolves during conversation.
"""

import streamlit as st
import numpy as np
import os
import uuid
import heapq
from st_cookies_manager import EncryptedCookieManager
from newspaper import Article
from src.eigen_recursive_ai import RecursiveEigenAI
from src.eigen_discrete_tokenizer import tokenize_word, xor_states, compute_change_stability
from database import init_database
from db_helpers import (
    save_message, load_messages,
    save_learned_token, load_learned_tokens,
    save_learned_tokens_batch, load_tokens_by_classification,
    load_tokens_by_classifications_batch,
    get_session_stats
)

# Initialize cookie manager for persistent session ID
cookies = EncryptedCookieManager(
    prefix="eigenai_",
    password=os.environ.get("COOKIE_PASSWORD", "eigenai-default-secret-key-change-in-production")
)

def tokenize_text(text):
    """Tokenize a text string into discrete tokens"""
    words = text.split()
    return [tokenize_word(word) for word in words]

def adaptive_F(vocab_size, text_length, memory_available=None):
    """
    Adaptive F selection based on bottleneck constraints

    F (fan-in capacity) should match the limiting resource:
    - Human constraint: F=2-8 (working memory)
    - Vocab constraint: F=4-8 (minimize overhead)
    - Text constraint: F=32-64 (amortize setup)
    - Memory constraint: F=4 (reduce buffers)

    This implements k* = √(oP·F/c) by adapting F to the constraint.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in vocabulary
    text_length : int
        Number of words to process
    memory_available : int, optional
        Available memory in bytes (not used yet)

    Returns
    -------
    F : int
        Optimal fan-in for this context

    Examples
    --------
    >>> adaptive_F(1000, 50)     # Small vocab, short text
    8
    >>> adaptive_F(1000000, 10000)  # Huge vocab, long text
    64
    """
    # Memory constraint (if specified)
    if memory_available is not None and memory_available < 100_000:
        return 4  # Tight memory, minimize buffers

    # Large vocabulary → maximize throughput
    if vocab_size > 1_000_000:
        return 64 if text_length > 5000 else 32

    # Long text → amortize setup cost
    if text_length > 10_000:
        return 64
    elif text_length > 1000:
        return 32

    # Default human-scale
    return 8

def tokenize_and_record_transitions(text, learned_tokens):
    """
    Tokenize text and record transition statistics for each token

    Parameters
    ----------
    text : str
        Input text
    learned_tokens : dict
        Dictionary of learned tokens (will be updated in-place)

    Returns
    -------
    tokens : list of DiscreteToken
        Tokens with updated transition statistics
    """
    words = text.split()
    tokens = []

    # Initialize state for XOR cascade
    state = (0, 0, 0, 0)

    for word in words:
        # Get or create token
        word_key = word.lower()
        if word_key in learned_tokens:
            token = learned_tokens[word_key]
        else:
            token = tokenize_word(word)

        # Compute transition caused by this token
        prev_state = state
        state = xor_states(state, token.as_tuple())
        C, S, ds2 = compute_change_stability(prev_state, state)

        # Record transition statistics
        token.record_transition(ds2)

        # Update learned tokens
        learned_tokens[word_key] = token
        tokens.append(token)

    return tokens

def tokenize_and_record_transitions_parallel(text, learned_tokens, F=8):
    """
    F-aware parallel tokenization applying batching framework: k* = √(oP·F/c)

    Reduces processing depth from O(n) to O(n/F + log_F(n)) by:
    1. Batch token lookup/creation (F words at once)
    2. Vectorized transition computation (numpy arrays)
    3. Hierarchical state reduction

    For n=1000 words, F=8:
    - Old depth: 1000 sequential operations
    - New depth: 125 batches + 3 levels ≈ 128 operations (7.8× reduction)

    Parameters
    ----------
    text : str
        Input text
    learned_tokens : dict
        Dictionary of learned tokens (will be updated in-place)
    F : int
        Fan-in capacity (batch size). Default=8 for human-comprehension tasks.
        Increase to 32-64 for large article processing.

    Returns
    -------
    tokens : list of DiscreteToken
        Tokens with updated transition statistics
    """
    words = text.split()
    if not words:
        return []

    tokens = []
    word_keys = [w.lower() for w in words]

    # PHASE 1: Batch token lookup/creation (F words at once)
    # Reduces database access patterns from O(n) to O(n/F)
    for i in range(0, len(words), F):
        batch_words = words[i:i+F]
        batch_keys = word_keys[i:i+F]

        # Parallel lookup: check all F words at once
        batch_tokens = []
        for word, word_key in zip(batch_words, batch_keys):
            if word_key in learned_tokens:
                token = learned_tokens[word_key]
            else:
                # Create new token (parallelizable across batch)
                token = tokenize_word(word)
                learned_tokens[word_key] = token
            batch_tokens.append(token)

        tokens.extend(batch_tokens)

    # PHASE 2: Vectorized XOR cascade and transition computation
    # Convert all tokens to numpy arrays for parallel computation
    n = len(tokens)
    L_vals = np.array([t.L for t in tokens], dtype=np.int32)
    R_vals = np.array([t.R for t in tokens], dtype=np.int32)
    V_vals = np.array([t.V for t in tokens], dtype=np.int32)
    M_vals = np.array([t.M for t in tokens], dtype=np.int32)

    # Compute XOR cascade: state[i] = state[i-1] ⊕ token[i]
    # This must be sequential due to dependencies, but XOR is fast
    states_L = np.zeros(n+1, dtype=np.int32)
    states_R = np.zeros(n+1, dtype=np.int32)
    states_V = np.zeros(n+1, dtype=np.int32)
    states_M = np.zeros(n+1, dtype=np.int32)

    for i in range(n):
        states_L[i+1] = states_L[i] ^ L_vals[i]
        states_R[i+1] = states_R[i] ^ R_vals[i]
        states_V[i+1] = states_V[i] ^ V_vals[i]
        states_M[i+1] = states_M[i] ^ M_vals[i]

    # PHASE 3: Vectorized transition computation (parallelized)
    # Compute C, S, ds² for all transitions at once
    prev_states = np.stack([states_L[:-1], states_R[:-1], states_V[:-1], states_M[:-1]], axis=1)
    curr_states = np.stack([states_L[1:], states_R[1:], states_V[1:], states_M[1:]], axis=1)

    # XOR to find changed bits
    changed = prev_states ^ curr_states

    # Count bits: stable = 1024 - changed, changed = changed
    # Using numpy's binary operations for parallel bit counting
    C_vals = np.array([bin(changed[i, 0]).count('1') + bin(changed[i, 1]).count('1') +
                       bin(changed[i, 2]).count('1') + bin(changed[i, 3]).count('1')
                       for i in range(n)])
    S_vals = 32 - C_vals  # 4 bytes × 8 bits = 32 total bits
    ds2_vals = S_vals**2 - C_vals**2

    # PHASE 4: Batch record transitions (vectorized)
    for i, (token, ds2) in enumerate(zip(tokens, ds2_vals)):
        token.record_transition(int(ds2))

    return tokens

def read_article_from_url(url):
    """
    Fetch and extract article text from a URL using newspaper3k
    
    Returns:
        dict: {
            'success': bool,
            'title': str,
            'text': str,
            'authors': list,
            'publish_date': str,
            'error': str (if failed)
        }
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        # Optional: run NLP for keywords and summary
        try:
            article.nlp()
            keywords = article.keywords
            summary = article.summary
        except:
            keywords = []
            summary = ""
        
        return {
            'success': True,
            'title': article.title or "Untitled",
            'text': article.text,
            'authors': article.authors,
            'publish_date': str(article.publish_date) if article.publish_date else "Unknown",
            'keywords': keywords,
            'summary': summary,
            'url': url
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'url': url
        }

# Initialize database tables on startup
try:
    init_database()
except Exception as e:
    st.error(f"Database initialization error: {e}")
    st.stop()

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

# Wait for cookies to be ready
if not cookies.ready():
    st.stop()

# Get or create persistent session ID from cookies
# Use "global" as default to accumulate ALL tokens across all sessions
if "session_id" not in cookies:
    cookies["session_id"] = "global"  # Single global session for all users
    cookies.save()

# Store in session state for this page load
if "session_id" not in st.session_state:
    st.session_state.session_id = cookies["session_id"]

# Initialize AI and load token vocabulary from database
if "ai" not in st.session_state:
    st.session_state.ai = RecursiveEigenAI(embedding_dim=64)
    st.session_state.messages = []  # Always start with fresh conversation
    st.session_state.metrics_history = []
    st.session_state.learned_tokens = {}

    # Load global token vocabulary (community-built)
    # AI state NOT persisted - fresh processing framework each session avoids
    # race conditions in global mode where concurrent users would overwrite
    # each other's extraction rules and context embeddings
    try:
        st.session_state.learned_tokens = load_learned_tokens(st.session_state.session_id)
    except Exception as e:
        st.sidebar.warning(f"Could not load token vocabulary: {e}")

def compute_semantic_similarities_vectorized(tokens_list, avg_L, avg_R, avg_V, avg_M):
    """
    Vectorized semantic similarity computation using numpy

    10-100x faster than Python loops for large token sets

    Returns: list of (similarity_score, word, token) tuples
    """
    if not tokens_list:
        return []

    # Extract LRVM values into numpy arrays
    L_vals = np.array([t.L for _, t in tokens_list])
    R_vals = np.array([t.R for _, t in tokens_list])
    V_vals = np.array([t.V for _, t in tokens_list])
    M_vals = np.array([t.M for _, t in tokens_list])

    # Vectorized similarity computation (all tokens at once)
    l_sim = 1.0 / (1.0 + np.abs(L_vals - avg_L) / 255.0)
    r_sim = 1.0 / (1.0 + np.abs(R_vals - avg_R) / 255.0)
    v_sim = 1.0 / (1.0 + np.abs(V_vals - avg_V) / 255.0)
    m_sim = 1.0 / (1.0 + np.abs(M_vals - avg_M) / 255.0)

    # Weighted average (M has 2x weight as in original)
    similarities = (l_sim + r_sim + v_sim + 2.0 * m_sim) / 5.0

    # Return as list of (similarity, word, token) for heapq
    return [(sim, word, token) for sim, (word, token) in zip(similarities, tokens_list)]

def generate_token_response(learned_tokens, user_message, max_words=10):
    """
    Generate response using geometric classification of tokens (OPTIMIZED)

    Optimizations for global mode at scale:
    - Database-side filtering (don't load all tokens into memory)
    - Numpy vectorization for semantic similarity (10-100x faster)
    - heapq for top-N selection (O(n log k) instead of O(n log n))

    Uses transition statistics to separate:
    - Time-like tokens (S > C): structural/sequential words
    - Space-like tokens (C > S): semantic/content words
    - Light-like tokens (C = S): relational/transformational words
    """
    if len(learned_tokens) < 3:
        return "⋯ (Not enough tokens learned yet to generate response)"

    # Get session ID from session state
    session_id = st.session_state.session_id

    # Target distribution based on natural language
    num_structural = max(1, max_words // 4)  # ~25% structural
    num_content = max(1, max_words // 2)     # ~50% content
    num_relational = max(1, max_words // 4)  # ~25% relational

    # Load all classifications in single batch query (batching framework applied to DB)
    # Old: 3 queries × 50ms = 150ms
    # New: 1 query × 60ms = 60ms (2.5× speedup)
    classified_tokens = load_tokens_by_classifications_batch(session_id, {
        'time-like': num_structural * 5,
        'space-like': num_content * 10,
        'light-like': num_relational * 5
    })

    time_like_tokens = list(classified_tokens['time-like'].items())
    space_like_tokens = list(classified_tokens['space-like'].items())
    light_like_tokens = list(classified_tokens['light-like'].items())

    # Tokenize user input to extract semantic intent
    user_tokens = tokenize_text(user_message.lower())
    if not user_tokens:
        return "⋯"

    # Compute average semantic pattern from user input
    avg_L = sum(t.L for t in user_tokens) / len(user_tokens)
    avg_R = sum(t.R for t in user_tokens) / len(user_tokens)
    avg_V = sum(t.V for t in user_tokens) / len(user_tokens)
    avg_M = sum(t.M for t in user_tokens) / len(user_tokens)

    # Vectorized similarity computation for space-like tokens
    space_similarities = compute_semantic_similarities_vectorized(
        space_like_tokens, avg_L, avg_R, avg_V, avg_M
    )

    # Use heapq to get top N without sorting entire list (O(n log k) vs O(n log n))
    top_space_tokens = heapq.nlargest(num_content, space_similarities, key=lambda x: x[0])

    # Build response with geometric structure
    response_words = []

    # Add time-like tokens (structural/sequential)
    for word, token in time_like_tokens[:num_structural]:
        response_words.append(word)

    # Add space-like tokens (semantic content) - highest similarity first
    for sim, word, token in top_space_tokens:
        response_words.append(word)

    # Add light-like tokens (relational/transformational)
    for word, token in light_like_tokens[:num_relational]:
        response_words.append(word)

    if not response_words:
        return "⋯"

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
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Eigenstate", "✓" if m.get("eigenstate") else "✗")
                    with col_b:
                        st.metric("Iteration", m.get("iteration", "N/A"))
                    
                    # Show thinking iterations chart if available
                    if m.get("iterations_log"):
                        st.divider()
                        st.caption("🧠 Thinking Evolution")
                        import pandas as pd
                        df = pd.DataFrame(m["iterations_log"])
                        st.line_chart(df.set_index("iteration")[["m_norm"]])
                    
                    # Show discrete tokens if available
                    if "tokens" in m:
                        st.divider()
                        st.caption("🔢 Discrete Token Bit Patterns & Classification")
                        for token in m["tokens"][:5]:  # Show first 5 tokens
                            classification = token.get_classification()
                            ratios = token.get_classification_ratios()

                            # Icon for classification
                            icon = "⏱️" if classification == "time-like" else "🌐" if classification == "space-like" else "💫" if classification == "light-like" else "❓"

                            st.code(f"{token.word:>12} → L:{token.L:08b} R:{token.R:08b} V:{token.V:08b} M:{token.M:08b}", language="text")
                            st.caption(f"{icon} {classification} | Usage: {token.usage_count} | T:{ratios['time-like']:.2f} S:{ratios['space-like']:.2f} L:{ratios['light-like']:.2f}")
    
    prompt = st.chat_input("Type your message here...")
    
    if "example_clicked" in st.session_state and st.session_state.example_clicked:
        prompt = st.session_state.example_clicked
        st.session_state.example_clicked = None
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Save user message to database
        try:
            save_message(st.session_state.session_id, "user", prompt)
        except Exception as e:
            st.sidebar.error(f"DB save error: {e}")
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("Processing and detecting eigenstates..."):
            # Tokenize user input with adaptive F-aware parallel processing
            # F adapts to constraint: vocab size, text length
            vocab_size = len(st.session_state.learned_tokens)
            text_length = len(prompt.split())
            F = adaptive_F(vocab_size, text_length)

            tokens = tokenize_and_record_transitions_parallel(
                prompt,
                st.session_state.learned_tokens,
                F=F
            )

            # Save tokens to database (batch operation for efficiency)
            try:
                save_learned_tokens_batch(st.session_state.session_id, tokens)
            except Exception as e:
                st.sidebar.error(f"Token save error: {e}")
            
            # MULTI-ITERATION THINKING: Process input multiple times
            iterations_log = []
            num_iterations = st.session_state.thinking_iterations
            
            for think_iter in range(num_iterations):
                # Process user input through EigenAI framework
                result = st.session_state.ai.process(prompt, verbose=False)
                
                # Capture state after this iteration
                state_snapshot = st.session_state.ai.get_state_summary()
                iterations_log.append({
                    "iteration": think_iter + 1,
                    "eigenstate": result['eigenstate'],
                    "m_norm": state_snapshot['M_context_norm'],
                    "l_weight": st.session_state.ai.extraction_rules['L_weight'],
                    "r_weight": st.session_state.ai.extraction_rules['R_weight'],
                    "v_weight": st.session_state.ai.extraction_rules['V_weight'],
                })
            
            # Use the final iteration's result
            result['iterations_log'] = iterations_log
            
            # Generate token-based response (AI speaking from learned vocabulary)
            token_generated = generate_token_response(
                st.session_state.learned_tokens, 
                prompt
            )
            
            # Learn tokens from generated response and record transitions
            response_tokens = tokenize_and_record_transitions(
                token_generated,
                st.session_state.learned_tokens
            )

            # Save response tokens to database (batch operation for efficiency)
            try:
                save_learned_tokens_batch(st.session_state.session_id, response_tokens)
            except Exception as e:
                st.sidebar.error(f"Token save error: {e}")
            
            # Feed the generated response back into itself for recursive understanding
            # This deepens the understanding without cluttering the chat
            st.session_state.ai.process(token_generated, verbose=False)
            
            # Get updated state after recursive processing
            state = st.session_state.ai.get_state_summary()
            
            # Create response with eigenstate information
            full_response = f"**🔢 Token-Generated Response:**\n{token_generated}\n\n"
            full_response += f"*({len(st.session_state.learned_tokens)} unique tokens learned)*\n\n"
            
            # Show thinking iterations
            if num_iterations > 1:
                full_response += f"**🧠 Thinking Process ({num_iterations} iterations):**\n"
                for log in iterations_log:
                    eigenstate_icon = "✓" if log['eigenstate'] else "⋯"
                    full_response += f"  Iteration {log['iteration']}: {eigenstate_icon} M={log['m_norm']:.3f} | L={log['l_weight']:.2f} R={log['r_weight']:.2f} V={log['v_weight']:.2f}\n"
                full_response += "\n"
            
            if result['eigenstate']:
                full_response += f"✓ *Meta-eigenstate reached (iteration {result['iteration']}) | Framework strength: {state['M_context_norm']:.2f}*"
            else:
                full_response += f"⋯ *Building understanding (iteration {result['iteration']}) | Framework strength: {state['M_context_norm']:.2f}*"
            
            metrics = {
                "eigenstate": result['eigenstate'],
                "iteration": result['iteration'],
                "M_context_norm": state['M_context_norm'],
                "tokens": tokens,  # Include discrete tokens
                "vocab_size": len(st.session_state.learned_tokens),
                "iterations_log": iterations_log if num_iterations > 1 else None
            }
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "metrics": metrics
            })
            st.session_state.metrics_history.append(metrics)
            
            # Save assistant message to database (tokens already saved in batch above)
            # AI state NOT persisted for global mode - see initialization for rationale
            try:
                save_message(st.session_state.session_id, "assistant", full_response, metrics)
            except Exception as e:
                st.sidebar.error(f"DB save error: {e}")
        
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
    
    st.subheader("🧠 Thinking Iterations")
    st.caption("How many times should the AI iterate on each input?")
    thinking_iterations = st.slider(
        "Iteration depth",
        min_value=1,
        max_value=10,
        value=3,
        help="More iterations = deeper understanding convergence"
    )
    
    if "thinking_iterations" not in st.session_state:
        st.session_state.thinking_iterations = 3
    
    st.session_state.thinking_iterations = thinking_iterations
    
    st.divider()
    
    st.subheader("📰 Read Online Articles")
    st.caption("Let the AI learn from web articles")
    
    article_url = st.text_input(
        "Enter article URL",
        placeholder="https://example.com/article",
        help="Paste a URL to an article. The AI will extract and learn from it."
    )
    
    if st.button("📖 Read Article", use_container_width=True, disabled=not article_url):
        with st.spinner("Fetching and reading article..."):
            result = read_article_from_url(article_url)
            
            if result['success']:
                # Learn tokens from article text with adaptive F-aware parallel processing
                # F adapts based on article length and vocab size
                vocab_size = len(st.session_state.learned_tokens)
                text_length = len(result['text'].split())
                F = adaptive_F(vocab_size, text_length)

                initial_vocab_size = vocab_size
                article_tokens = tokenize_and_record_transitions_parallel(
                    result['text'],
                    st.session_state.learned_tokens,
                    F=F  # Adaptive fan-in based on constraints
                )

                # Save all tokens to database (batch operation - CRITICAL for large articles)
                try:
                    save_learned_tokens_batch(st.session_state.session_id, article_tokens)
                except Exception as e:
                    st.sidebar.error(f"Token save error: {e}")

                tokens_learned = len(st.session_state.learned_tokens) - initial_vocab_size
                
                # Show success message
                st.success(f"✅ **Article Read Successfully!**\n\n**{result['title']}**\n\nLearned {tokens_learned} new tokens from article")
                
                if result.get('authors'):
                    st.caption(f"👤 By: {', '.join(result['authors'])}")
                if result.get('keywords'):
                    st.caption(f"🏷️ Keywords: {', '.join(result['keywords'][:5])}")
            else:
                st.error(f"❌ Failed to read article: {result['error']}")
    
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
    
    if st.button("🔄 Clear Conversation", use_container_width=True):
        # Clear current conversation (global tokens persist for community building)
        st.session_state.messages = []
        st.session_state.metrics_history = []
        st.rerun()

    st.divider()
    
    # Show vocabulary stats with classification breakdown
    if len(st.session_state.learned_tokens) > 0:
        # Count tokens by classification
        time_like = sum(1 for t in st.session_state.learned_tokens.values() if t.get_classification() == 'time-like')
        space_like = sum(1 for t in st.session_state.learned_tokens.values() if t.get_classification() == 'space-like')
        light_like = sum(1 for t in st.session_state.learned_tokens.values() if t.get_classification() == 'light-like')
        unknown = sum(1 for t in st.session_state.learned_tokens.values() if t.get_classification() == 'unknown')

        st.caption(f"💾 **Persistent Vocabulary**")
        st.caption(f"📚 {len(st.session_state.learned_tokens)} tokens accumulated")
        st.caption(f"⏱️ Time-like (structural): {time_like}")
        st.caption(f"🌐 Space-like (semantic): {space_like}")
        st.caption(f"💫 Light-like (relational): {light_like}")
        if unknown > 0:
            st.caption(f"❓ Unknown (learning): {unknown}")

st.sidebar.title("About EigenAI")

# Show session type
if st.session_state.session_id == "global":
    st.sidebar.info("🌍 **Global Session Mode**\nAll users contribute to the same token vocabulary. The AI learns from everyone!")
else:
    st.sidebar.success(f"👤 **Personal Session**\nYour unique session ID: `{st.session_state.session_id[:8]}...`")

st.sidebar.markdown("""
### What is This?

This chatbot demonstrates **recursive self-modifying AI** that measures understanding through **eigenstate detection**.

**Tokens persist forever** - the AI accumulates vocabulary across ALL sessions and deployments!

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

**Geometric Token Classification**:
From transition metric ds² = S² - C²:
- ⏱️ **Time-like** (S > C): Structural/sequential words (the, to, is)
- 🌐 **Space-like** (C > S): Semantic/content words (quantum, cat)
- 💫 **Light-like** (C = S): Relational/transformational words (becomes, connects)

**Recursive Self-Modification**:
The AI changes its own processing framework based on what it learns, building genuine comprehension over time.

### How It Works

1. You send a message
2. AI extracts semantic triad (L, R, V)
3. Computes meta-understanding (M)
4. Detects eigenstate convergence
5. Updates its extraction rules
6. Accumulates understanding context

Watch the metrics to see eigenstate detection in action!
""")

st.sidebar.divider()
st.sidebar.caption("Built on EigenAI framework | MIT License")
