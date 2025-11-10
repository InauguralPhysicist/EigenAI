# EigenAI Chatbot Setup Guide

## Overview

The EigenAI chatbot demonstrates recursive self-modifying AI with eigenstate detection. This guide covers setup with PostgreSQL persistence for Replit deployment.

## Features

- 🧠 **Eigenstate Detection**: Real-time visualization of understanding convergence
- 💾 **Persistent Memory**: Conversations and learned tokens saved across sessions
- ✨ **Entropy Weighting**: v1.0.0 feature for improved semantic extraction
- 📊 **Metrics Tracking**: Monitor extraction rules and framework evolution

## Quick Start (Replit)

### 1. Install Dependencies

```bash
pip install -r requirements-chatbot.txt
```

### 2. Set Up PostgreSQL Database

#### Option A: Replit Database (Recommended)

1. Go to your Replit project
2. Click **"Database"** in the left sidebar
3. Replit will automatically provision a PostgreSQL database
4. The connection URL is available as `DATABASE_URL` environment variable

#### Option B: External PostgreSQL

Set the `DATABASE_URL` environment variable:

```bash
export DATABASE_URL="postgresql://user:password@host:port/dbname"
```

### 3. Initialize Database Schema

The schema is automatically created on first run. To manually initialize:

```bash
python3 -c "from chatbot_db import ChatbotDatabase; ChatbotDatabase()"
```

### 4. Run the Chatbot

```bash
streamlit run streamlit_chatbot_persistent.py
```

## Database Schema

### Tables

- **chatbot_sessions**: Main session metadata
  - `session_id`: Primary key
  - `total_iterations`: Current iteration count
  - `eigenstate_reached`: Boolean flag
  - `ai_state`: Serialized RecursiveEigenAI object
  - `extraction_rules`: L, R, V, Context weights

- **learned_tokens**: Discrete token vocabulary
  - `word`: Token text
  - `l_bits`, `r_bits`, `v_bits`, `m_bits`: Bit representations

- **messages**: Conversation history
  - `role`: 'user' or 'assistant'
  - `content`: Message text
  - `metrics`: Eigenstate metrics (JSON)

- **metrics_history**: Iteration metrics
  - `iteration`: Iteration number
  - `eigenstate`: Convergence status
  - `m_context_norm`: Framework strength
  - `vocab_size`: Number of learned tokens

## Usage

### Continue Previous Session

Click **"📂 Continue Last Session"** to load your most recent conversation with all learned tokens and extraction rules.

### Start New Session

Click **"✨ New Session"** to start fresh while keeping previous sessions in the database.

### Enable Entropy Weighting

Toggle **"✨ Entropy Weighting"** to use information-density-based semantic extraction (14.6× better orthogonality).

## Session Persistence

State automatically saves after each interaction:
- ✅ All learned tokens (word → L, R, V, M bit patterns)
- ✅ Complete conversation history
- ✅ Extraction rule evolution
- ✅ Metrics and eigenstate progression
- ✅ RecursiveEigenAI internal state

## Deployment

### Replit Deployment

1. Install requirements: `pip install -r requirements-chatbot.txt`
2. Enable PostgreSQL database in Replit
3. Run: `streamlit run streamlit_chatbot_persistent.py`
4. Deploy using Replit's deployment feature

### Local Development

For local development without PostgreSQL, use the non-persistent version:

```bash
streamlit run streamlit_chatbot.py
```

State will reset between sessions but the chatbot remains fully functional.

## Troubleshooting

### Database Connection Errors

If you see database connection errors:

1. Check `DATABASE_URL` environment variable is set
2. Verify PostgreSQL is running
3. Ensure `psycopg2-binary` is installed
4. Check firewall/network settings

### Session Not Loading

If previous sessions don't load:

1. Check database contains data: `SELECT * FROM chatbot_sessions;`
2. Verify schema was initialized properly
3. Look for errors in Streamlit console

### Performance Issues

For large vocabularies (>5000 tokens):

1. Database queries are indexed for performance
2. Consider archiving old sessions
3. Use session-specific token loading

## Architecture

```
streamlit_chatbot_persistent.py  ← Main Streamlit app
         ↓
chatbot_db.py                    ← Database manager
         ↓
db_schema.sql                    ← PostgreSQL schema
         ↓
PostgreSQL Database              ← Persistent storage
```

## API Reference

### ChatbotDatabase

```python
from chatbot_db import ChatbotDatabase

# Initialize
db = ChatbotDatabase()

# Create session
session_id = db.create_session()

# Save state
db.update_session(session_id, ai_state, extraction_rules, iterations, eigenstate)

# Load state
ai_state = db.load_ai_state(session_id)
tokens = db.load_learned_tokens(session_id)
messages = db.load_messages(session_id)

# Save data
db.save_learned_token(session_id, token, iteration)
db.save_message(session_id, role, content, iteration, metrics)
db.save_metrics(session_id, iteration, eigenstate, norm, vocab_size)
```

## Contributing

To add new persistence features:

1. Update `db_schema.sql` with new tables/columns
2. Add methods to `ChatbotDatabase` class
3. Update Streamlit app to use new persistence
4. Test locally before deploying

## License

MIT License - see main EigenAI repository
