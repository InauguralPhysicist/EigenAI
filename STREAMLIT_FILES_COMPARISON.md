# EigenAI Streamlit Files Comparison

## Quick Reference Table

| Aspect | streamlit_app.py | streamlit_chatbot.py | app.py |
|--------|------------------|----------------------|--------|
| **Lines of Code** | 468 | 534 | 777 |
| **Layout** | Wide (2:1) with tabs | Wide (2:1) | Wide (2:1) |
| **Tabs** | 3 (Chatbot, Geometric, About) | None | None |
| **Database** | No | No | Yes |
| **Persistence** | Session only | Session only | Global + Sessions |
| **Articles** | No | No | Yes |
| **Thinking Iterations** | No | No | Yes (1-10 slider) |
| **Charts** | 1 (line chart) | 0 | 1 (thinking evolution) |
| **Metrics** | Basic | Basic | Enhanced |
| **Complexity** | Medium | Medium | High |

---

## Detailed Feature Comparison

### Core Features
All three share:
- Chat interface with message history
- Token extraction and learning
- Discrete token visualization (bit patterns)
- Eigenstate detection
- Extraction rules display
- Token classification (time/space/light-like)
- Example prompts
- Reset functionality
- Sidebar documentation

### Unique Features

#### streamlit_app.py
- **Multi-tab design**: Separates chatbot, geometric tests, about
- **Geometric tests tab**: Monte Carlo sampling for Prince Rupert's Cube
- **About tab**: Educational content, comprehensive documentation
- **Entropy weighting toggle**: Checkbox in left column
- **Metrics history chart**: Single line chart (streamlit_app only)

#### streamlit_chatbot.py
- **Single-page focus**: Simpler cognitive load
- **Token classification counts**: Time-like, Space-like, Light-like breakdown
- **Entropy toggle**: In top settings bar
- **Cleaner layout**: No tabs, all features accessible
- **Sidebar focus**: Educational content in sidebar

#### app.py (Most Advanced)
- **Database persistence**: Saves all tokens and messages
- **Global session mode**: Community-built vocabulary (all users share tokens)
- **Multi-iteration thinking**: Slider to control 1-10 iterations per input
- **Article reading**: URL input to fetch and learn from articles
- **Thinking evolution chart**: Shows iterations and convergence
- **Session management**: Cookie-based persistent session IDs
- **Adaptive F-aware tokenization**: Dynamic batch sizing based on vocabulary
- **Database optimizations**: Batch saves for efficiency
- **Persistent vocabulary**: Tokens persist across sessions

---

## Code Structure Analysis

### Imports and Dependencies

**streamlit_app.py**:
```python
import streamlit as st
import numpy as np
import sys
import os
from src.eigen_recursive_ai import RecursiveEigenAI
from src.eigen_discrete_tokenizer import tokenize_word
```

**streamlit_chatbot.py**:
```python
import streamlit as st
import numpy as np
import heapq  # ← Added for optimized selection
import sys
import os
from src.eigen_recursive_ai import RecursiveEigenAI
from src.eigen_discrete_tokenizer import tokenize_word, xor_states, compute_change_stability
```

**app.py**:
```python
import streamlit as st
import numpy as np
import os
import uuid
import heapq
from st_cookies_manager import EncryptedCookieManager  # ← New
from newspaper import Article  # ← New (article fetching)
from src.eigen_recursive_ai import RecursiveEigenAI
from database import init_database  # ← New
from db_helpers import (  # ← New (database operations)
    save_message, load_messages,
    save_learned_token, load_learned_tokens,
    ...
)
```

---

## Session State Variables

### All Three Files
```python
st.session_state.ai                    # RecursiveEigenAI instance
st.session_state.messages              # Chat history
st.session_state.metrics_history       # Metrics for each turn
st.session_state.learned_tokens        # Token vocabulary
```

### streamlit_chatbot.py Only
- (None unique)

### app.py Only
```python
st.session_state.session_id            # Persistent session identifier
st.session_state.thinking_iterations   # User-selected iteration depth
```

---

## Layout Organization

### streamlit_app.py
```
Page
├─ Title
├─ Tabs (3)
│  ├─ Tab 1: Chatbot
│  │  ├─ Columns (2:1)
│  │  │  ├─ Left: Conversation + entropy toggle
│  │  │  └─ Right: Metrics dashboard
│  │  └─ Components:
│  │     ├─ Chat history (expandable metrics)
│  │     ├─ Chat input
│  │     ├─ KPI metrics (3)
│  │     ├─ Extraction rules (4 progress bars)
│  │     ├─ Metrics history (line chart)
│  │     ├─ Example prompts (5 buttons)
│  │     ├─ Reset button
│  │     └─ Vocabulary summary
│  │
│  ├─ Tab 2: Geometric Tests
│  │  ├─ Parameters section (2 sliders)
│  │  ├─ Test button
│  │  └─ Results display
│  │
│  └─ Tab 3: About
│     └─ Educational content
│
└─ Sidebar
   └─ Quick reference + About
```

### streamlit_chatbot.py
```
Page
├─ Title + description
├─ Entropy toggle (top right)
├─ Columns (2:1)
│  ├─ Left: Conversation
│  │  ├─ Chat history (expandable metrics)
│  │  └─ Chat input
│  │
│  └─ Right: Dashboard
│     ├─ KPI metrics (3)
│     ├─ Extraction rules (4 progress bars)
│     ├─ Example prompts (5 buttons)
│     ├─ Reset button
│     └─ Vocabulary classification breakdown
│
└─ Sidebar
   └─ About + Concepts
```

### app.py
```
Page
├─ Title + description
├─ Session info (🌍 Global or 👤 Personal)
├─ Columns (2:1)
│  ├─ Left: Conversation
│  │  ├─ Chat history (expandable metrics with thinking evolution)
│  │  └─ Chat input
│  │
│  └─ Right: Dashboard
│     ├─ KPI metrics (3)
│     ├─ Extraction rules (4 progress bars)
│     ├─ 🧠 Thinking Iterations slider
│     ├─ 📰 Read Articles section
│     │  ├─ URL input
│     │  └─ Read button
│     ├─ Example prompts (5 buttons)
│     ├─ Clear conversation button
│     └─ Persistent vocabulary breakdown
│
└─ Sidebar
   └─ Session type + About + Concepts
```

---

## Message Processing Flow

### streamlit_app.py & streamlit_chatbot.py
```
User Input
    ↓
Tokenize with tokenize_and_record_transitions_parallel(F=8)
    ↓
AI.process(prompt, verbose=False)
    ↓
generate_token_response()
    ↓
Tokenize response
    ↓
AI.process(response, verbose=False)
    ↓
Display response + metrics
```

### app.py
```
User Input
    ↓
Tokenize with adaptive F (4-64 based on vocab/text)
    ↓
Save tokens to database (batch operation)
    ↓
Multi-iteration thinking (1-10 times):
    ├─ AI.process(prompt, verbose=False)
    ├─ Capture state snapshot
    └─ Log iteration metrics
    ↓
generate_token_response() from database
    ↓
Tokenize response + save to database
    ↓
AI.process(response, verbose=False)
    ↓
Display response + thinking evolution
```

---

## Visualization Components Used

### streamlit_app.py
- st.metric() × 3 (KPIs)
- st.progress() × 4 (extraction rules)
- st.line_chart() × 1 (metrics history)
- st.expander() (per message)
- st.code() (tokens)
- st.columns() (layout)
- st.divider() (separation)
- st.checkbox() (entropy toggle)
- st.button() × 5+ (examples, reset)

### streamlit_chatbot.py
- st.metric() × 3 (KPIs)
- st.progress() × 4 (extraction rules)
- st.expander() (per message)
- st.code() (tokens)
- st.columns() (layout)
- st.divider() (separation)
- st.toggle() (entropy toggle)
- st.button() × 5+ (examples, reset)
- st.caption() (vocab classification)

### app.py
- st.metric() × 3 (KPIs)
- st.progress() × 4 (extraction rules)
- st.line_chart() × 1 (thinking evolution)
- st.expander() × multiple (message metrics)
- st.code() (tokens)
- st.columns() (layout)
- st.divider() (separation)
- st.slider() × 1 (thinking iterations)
- st.text_input() × 1 (article URL)
- st.button() × 5+ (examples, read article, clear)
- st.caption() (vocab breakdown)
- pandas + st.line_chart() (thinking evolution visualization)

---

## Performance Considerations

### streamlit_app.py
- In-memory token storage
- No database overhead
- Suitable for: Single-session demonstrations

### streamlit_chatbot.py
- In-memory token storage
- Optimized token response generation (heapq)
- Vectorized similarity computation (numpy)
- Suitable for: Quick testing, education

### app.py
- Database-backed persistence
- Batch operations for efficiency
- Adaptive F-aware tokenization
- Async article fetching possible
- Suitable for: Production use, long-term learning

---

## Which Version to Use?

### Use streamlit_app.py if you want:
- Educational multi-tab experience
- Geometric property demonstrations
- Clear separation of concepts
- Comprehensive about/documentation tab
- Single-session learning

### Use streamlit_chatbot.py if you want:
- Focused conversation experience
- No distractions (single page)
- Quick iteration testing
- Smaller mental model
- Lightweight implementation

### Use app.py if you want:
- Persistent learning across sessions
- Community-built vocabulary
- Multi-iteration thinking control
- Article integration for learning
- Database backup and analysis
- Production-ready implementation
- Global collaborative learning

---

## Code Statistics

```
streamlit_app.py:
- Total lines: 468
- Functions: 4 (tokenize_text, generate_token_response, tokenize_text, etc)
- Classes: 0 (uses imported classes)
- Comments: ~40 lines
- Code: ~350 lines
- UI: ~120 lines

streamlit_chatbot.py:
- Total lines: 534
- Functions: 6 (tokenize_text, tokenize_and_record_transitions, etc)
- Classes: 0
- Comments: ~50 lines
- Code: ~380 lines
- UI: ~100 lines

app.py:
- Total lines: 777
- Functions: 8 (tokenize_text, adaptive_F, read_article_from_url, etc)
- Classes: 0
- Comments: ~80 lines
- Code: ~550 lines
- UI: ~150 lines
- Database: ~50 lines
```

---

## File Size and Complexity

| File | Size | Complexity | Maintainability |
|------|------|-----------|-----------------|
| streamlit_app.py | 468 lines | Medium | High |
| streamlit_chatbot.py | 534 lines | Medium | High |
| app.py | 777 lines | High | Medium |

---

## Recommended Deployment

**Development**: Use `streamlit_chatbot.py` or `streamlit_app.py`
```bash
streamlit run streamlit_chatbot.py
```

**Production**: Use `app.py` with database
```bash
streamlit run app.py
```

**Education**: Use `streamlit_app.py` for tabs and demonstrations
```bash
streamlit run streamlit_app.py
```

---

## Future Enhancement Opportunities

**Common to all**:
- Add context accumulation layer visualization
- Implement interactive charts (Plotly)
- Add metric time series
- Token space visualization
- Radar chart for extraction rules

**app.py specific**:
- Message full-text search
- Batch article processing
- Token analytics dashboard
- Export conversation history
- Collaborative features (user tags)

