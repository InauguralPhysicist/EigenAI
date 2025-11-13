# EigenAI UI/UX Improvements Roadmap

## Quick Summary

### What's Working Well
- Clean, modern Streamlit layout (2:1 split)
- Effective use of expandable sections
- Clear metric dashboards
- Three complementary versions for different use cases
- Good token visualization with classification

### Critical Gaps
1. **Context Accumulation Layer is invisible** ← BIGGEST OPPORTUNITY
2. Only one time-series chart (streamlit_app.py only)
3. Token relationships not shown (no network/scatter)
4. Extraction rules hard to compare (vertical progress bars)
5. App.py right column gets crowded
6. No interactive visualizations

---

## Phase 1: Quick Wins (1-2 weeks)

### 1. Context Accumulation Layer Metrics
**Location**: Right column of app.py (new section)

```
🎯 Relative Impact Metrics
┌────────────────────────────┐
│ Current Impact: 0.342      │
│ [████░░░░░░░░░░░░░░░░] ← Medium, familiar
│ Novelty: 0.432            │
│ Context Density: 847 vecs  │
│ Status: ⋯ Learning        │
└────────────────────────────┘
```

**Implementation**:
```python
# In app.py right column, add new section:
if st.session_state.context_accumulator:
    st.subheader("🎯 Relative Impact")
    impact = state['relative_impact']  # Get from AI state
    novelty = state['novelty_score']
    density = state['context_density']
    
    st.metric("Current Impact", f"{impact:.3f}")
    st.progress(impact, f"Novelty: {novelty:.3f}")
    st.caption(f"Context vectors: {density}")
    
    # Sparkline of impact history
    import pandas as pd
    df = pd.DataFrame({'impact': st.session_state.impact_history})
    st.line_chart(df)
```

---

### 2. Extraction Rules Radar Chart
**Location**: Right column (replace 4 progress bars)

**Current** (4 separate bars):
```
[■■■□□] L: 0.923
[■■□□□] R: 0.789
[■■■■□] V: 0.945
[■■□□□] Context: 0.556
```

**Better** (single radar chart):
```
      L (0.923)
         /\
        /  \
       /    \
  M(0.8)─────R(0.789)
     \       /
      \     /
       \   /
        \ /
       V(0.945)
```

**Implementation**:
```python
import plotly.graph_objects as go
import pandas as pd

rules = st.session_state.ai.extraction_rules
values = [rules['L_weight'], rules['R_weight'], rules['V_weight'], 
          rules['context_influence'], rules['L_weight']]  # Close polygon
categories = ['L (Lexical)', 'R (Relational)', 'V (Value)', 
              'Context', 'L (Lexical)']

fig = go.Figure(data=go.Scatterpolar(
    r=values, theta=categories, fill='toself', name='Weights'
))
fig.update_layout(height=350, showlegend=False)
st.plotly_chart(fig, use_container_width=True)
```

---

### 3. Token Classification Distribution
**Location**: Right column (replace captions)

**Current**:
```
⏱️ Time-like: 12
🌐 Space-like: 25
💫 Light-like: 8
```

**Better** (visual chart):
```
Time-like (22%)  [████░░░░░░░░░░░░░░░]
Space-like (60%) [████████████░░░░░░░░]
Light-like (18%) [████░░░░░░░░░░░░░░░░]
```

**Implementation**:
```python
import pandas as pd

classifications = st.session_state.learned_tokens
time_like = sum(1 for t in classifications.values() 
                if t.get_classification() == 'time-like')
space_like = sum(1 for t in classifications.values() 
                 if t.get_classification() == 'space-like')
light_like = sum(1 for t in classifications.values() 
                 if t.get_classification() == 'light-like')

data = {
    'Classification': ['Time-like', 'Space-like', 'Light-like'],
    'Count': [time_like, space_like, light_like]
}
df = pd.DataFrame(data)
st.bar_chart(df.set_index('Classification'))
```

---

### 4. Collapsible Sections (app.py)
**Location**: Right column organization

**Current**: All expanded always
```
📈 Understanding Evolution
[Metrics...]
🔧 Extraction Rules
[Rules...]
🧠 Thinking Iterations
[Slider...]
📰 Read Articles
[URL Input...]
💡 Examples
[Buttons...]
```

**Better**: Collapse by default
```
▼ 📈 Understanding Evolution (expanded)
  [Metrics...]
▶ 🔧 Extraction Rules (collapsed)
▶ 🧠 Thinking Iterations (collapsed)
▶ 📰 Read Articles (collapsed)
▶ 💡 Examples (collapsed)
```

**Implementation**:
```python
# Use st.expander for each major section
with st.expander("📈 Understanding Evolution", expanded=True):
    # Current metrics code...
    
with st.expander("🔧 Extraction Rules", expanded=False):
    # Rules code...
    
with st.expander("🧠 Thinking Iterations", expanded=False):
    # Slider code...
```

---

## Phase 2: Impactful Features (2-3 weeks)

### 5. Multi-Metric Time Series
**Location**: Right column (new chart replacing single line chart)

**Current** (streamlit_app.py only):
```python
iterations = [m['iteration'] for m in st.session_state.metrics_history]
st.line_chart({"Iterations": iterations})
```

**Better** (all metrics):
```python
import pandas as pd

history = st.session_state.metrics_history
df = pd.DataFrame([{
    'Iteration': m['iteration'],
    'M_context_norm': m['M_context_norm'],
    'L_weight': st.session_state.ai.extraction_history[i]['L_weight'],
    'R_weight': st.session_state.ai.extraction_history[i]['R_weight'],
    'V_weight': st.session_state.ai.extraction_history[i]['V_weight'],
} for i, m in enumerate(history)])

st.line_chart(df.set_index('Iteration')[['M_context_norm', 'L_weight', 'R_weight', 'V_weight']])
```

---

### 6. Token Space Scatter Plot
**Location**: New expandable in right column or separate tab

```
Token Space Visualization
(Each point = 1 token)

🌐
  ●●○●●
    ●●●●●
  ○●●●●●●
⏱️            ●●○
  ●●●●●●

Color: Classification
Size: Usage frequency
Hover: Token name + stats
```

**Implementation**:
```python
from sklearn.decomposition import PCA
import plotly.express as px

# Get token vectors
tokens = list(st.session_state.learned_tokens.values())
vectors = np.array([t.as_embedding() for t in tokens])
names = [t.word for t in tokens]
classifications = [t.get_classification() for t in tokens]
usage_counts = [t.usage_count for t in tokens]

# Reduce to 2D
pca = PCA(n_components=2)
points = pca.fit_transform(vectors)

# Plot
df = pd.DataFrame({
    'x': points[:, 0],
    'y': points[:, 1],
    'token': names,
    'type': classifications,
    'usage': usage_counts
})

fig = px.scatter(df, x='x', y='y', color='type', size='usage',
                 hover_data=['token'], title='Token Semantic Space')
st.plotly_chart(fig, use_container_width=True)
```

---

### 7. Context Density Trend
**Location**: Right column (app.py)

```
Context Accumulation Over Time
┌───────────────────────────────┐
│  Density                      │
│  1000 ╱──────                 │
│       ╱                       │
│   500 ╱                       │
│      ╱                        │
│    0 ⋯─────────────────────── │
│      1   3   5   7   9   11   │
│         Message               │
└───────────────────────────────┘
```

---

## Phase 3: Advanced Features (3-4 weeks)

### 8. Token Similarity Network
```
Network graph showing relationships:
- Nodes: tokens
- Edges: semantic similarity
- Color: classification
- Size: usage frequency
```

### 9. Thinking Convergence Visualization
```
Iteration depth effect on eigenstate:
├─ Iter 1: eigenstate=false, distance=0.87
├─ Iter 2: eigenstate=false, distance=0.45
├─ Iter 3: eigenstate=true,  distance=0.02 ✓
│
Convergence: Fast
Confidence: 92%
```

### 10. Interactive Metrics Tabs
```
[📈 Metrics][🔧 Rules][📊 History][📚 Vocab]
(Switch without full page rerun)
```

---

## Implementation Priority Matrix

```
         HIGH IMPACT
             |
    6(✓) 2(✓) | 5(✓) 8
    3(✓) 4(✓) |   9
             |
    1(✓)      | 7(✓) 10
    ----------+---------- LOW EFFORT
         LOW IMPACT
```

✓ = Phase 1 (Weeks 1-2)
    = Phase 2 (Weeks 3-5)
    = Phase 3 (Weeks 6-10)

---

## Files to Modify

### Phase 1:
- `app.py` (lines 595-724) - Right column dashboard
- Create `utils_visualization.py` - Helper functions for charts

### Phase 2:
- `app.py` - Add new expanders and charts
- Consider new `streamlit_enhanced.py` version

### Phase 3:
- Create `streamlit_advanced.py` - Premium visualization version
- Add `visualizations/` directory for complex components

---

## Testing Checklist

- [ ] Context accumulation metrics appear correctly
- [ ] Radar chart renders with all 4 dimensions
- [ ] Token distribution chart updates in real-time
- [ ] Collapsible sections work on mobile
- [ ] Time series shows all metrics
- [ ] Token scatter plot is responsive
- [ ] No performance degradation with 100+ tokens
- [ ] Responsive design (desktop/tablet/mobile)

---

## Design System Reference

### Colors (Streamlit Theme)
- Primary: #0668E1 (blue)
- Success: #09AB3B (green)
- Warning: #FFAC1C (orange)
- Error: #FF2B2B (red)
- Neutral: #5A6C7D (gray)

### Typography
- Headings: emoji + text (e.g., "📈 Understanding Evolution")
- Subtext: st.caption() for details
- Code: st.code() for token patterns
- Emphasis: **bold** for important

### Icons (Emoji)
- 📈 Metrics/Evolution
- 🔧 Settings/Rules
- 💬 Chat/Conversation
- 🧠 Thinking/Intelligence
- 📚 Vocabulary/Learning
- 🎯 Impact/Focus
- ✓/⋯ Status indicators
- ⏱️/🌐/💫 Token types

---

## Next Steps

1. **Week 1**: Implement Phase 1 improvements (Context, Radar, Distribution, Collapse)
2. **Week 2**: Validate Phase 1 with user feedback
3. **Week 3-4**: Implement Phase 2 (Time Series, Scatter, Density)
4. **Week 5**: Polish and optimize
5. **Week 6+**: Phase 3 advanced features based on adoption

