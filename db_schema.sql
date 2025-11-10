-- EigenAI Chatbot State Persistence Schema
-- PostgreSQL database schema for maintaining conversation state across sessions

-- Main session table
CREATE TABLE IF NOT EXISTS chatbot_sessions (
    session_id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_iterations INTEGER DEFAULT 0,
    framework_strength REAL DEFAULT 1.0,
    eigenstate_reached BOOLEAN DEFAULT FALSE,
    ai_state JSONB,  -- Serialized RecursiveEigenAI state
    extraction_rules JSONB  -- L, R, V, Context weights
);

-- Learned tokens table
CREATE TABLE IF NOT EXISTS learned_tokens (
    token_id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES chatbot_sessions(session_id) ON DELETE CASCADE,
    word TEXT NOT NULL,
    l_bits INTEGER,
    r_bits INTEGER,
    v_bits INTEGER,
    m_bits INTEGER,
    learned_at_iteration INTEGER,
    UNIQUE(session_id, word)
);

-- Conversation messages table
CREATE TABLE IF NOT EXISTS messages (
    message_id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES chatbot_sessions(session_id) ON DELETE CASCADE,
    role TEXT NOT NULL,  -- 'user' or 'assistant'
    content TEXT NOT NULL,
    iteration INTEGER,
    metrics JSONB,  -- eigenstate, vocab_size, entropy_weighted, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Metrics history table
CREATE TABLE IF NOT EXISTS metrics_history (
    metric_id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES chatbot_sessions(session_id) ON DELETE CASCADE,
    iteration INTEGER NOT NULL,
    eigenstate BOOLEAN,
    m_context_norm REAL,
    vocab_size INTEGER,
    entropy_weighted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_sessions_last_updated ON chatbot_sessions(last_updated DESC);
CREATE INDEX IF NOT EXISTS idx_tokens_session ON learned_tokens(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_metrics_session ON metrics_history(session_id, iteration);

-- View for latest session
CREATE OR REPLACE VIEW latest_session AS
SELECT * FROM chatbot_sessions
ORDER BY last_updated DESC
LIMIT 1;
