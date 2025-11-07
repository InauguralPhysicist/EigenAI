# How to Run EigenAI

Complete guide to using the Eigen text understanding system.

---

## Quick Start

### 1. Run Basic Demo

```bash
cd /home/user/EigenAI
python examples/simple_demo.py
```

**Output**: Analyzes several sentences, shows convergence metrics.

---

### 2. Run Interactive Demo

```bash
python examples/interactive_demo.py
```

**Output**: Detailed analysis with semantic triad extraction, method comparison, and eigenstate detection.

---

### 3. Run Tests

```bash
python tests/test_core.py
```

**Output**: Validates all core functions (should show "ALL TESTS PASSED ✓").

---

## Try Your Own Sentences

### Option A: Edit and Run Script

1. Edit `examples/try_your_own.py`
2. Add your sentences to the `sentences_to_analyze` list
3. Run:
   ```bash
   python examples/try_your_own.py
   ```

### Option B: Python Interactive

```python
import sys
sys.path.insert(0, '/home/user/EigenAI')

from src.eigen_text_core import understanding_loop

# Analyze a sentence
text = "Your sentence here"
M, history, metrics = understanding_loop(text, verbose=True)

# Check results
print(f"Converged: {metrics['converged']}")
print(f"Eigenstate: {metrics['eigenstate_type']}")
print(f"Regime: {metrics['final_regime']}")
```

### Option C: Command Line One-Liner

```bash
cd /home/user/EigenAI
python -c "
import sys
sys.path.insert(0, '/home/user/EigenAI')
from src.eigen_text_core import understanding_loop

M, hist, m = understanding_loop('The wind bends the tree', verbose=True)
print(f'\nEigenstate: {m[\"eigenstate_type\"]}')
"
```

---

## Understanding the Output

### Example Output

```
Iteration 1: alignment=1.000, C=2, S=298, ds²=88800, regime=time-like
Iteration 2: alignment=1.000, C=0, S=300, ds²=90000, regime=time-like
Fixed-point eigenstate reached at iteration 2

RESULTS:
  Converged: True
  Iterations: 3
  Eigenstate: fixed-point
  Final regime: time-like (settled/understood)
  Final alignment: 1.0000
  Final ds²: 90000
```

### Key Metrics Explained

| Metric | Meaning | Good Value |
|--------|---------|------------|
| **Alignment** | How similar M is between iterations (0-1) | >0.99 = converged |
| **C (Change)** | Number of dimensions changing | 0 = stable |
| **S (Stability)** | Number of dimensions stable | High = understood |
| **ds²** | S² - C² metric signature | Large positive = understood |
| **Regime** | Understanding state | "time-like" = settled |

### Eigenstate Types

1. **Fixed-point**: Understanding converged to stable meaning
   - Most common for clear sentences
   - C → 0, S → max, ds² → large positive

2. **Periodic** (period-2, period-8, etc.): Oscillating interpretation
   - Happens with ambiguous sentences
   - System cycles between interpretations
   - Example: "Time flies like an arrow" (multiple parse trees)

3. **None**: No convergence
   - Fundamental ambiguity
   - Contradictory or nonsensical input

### Regimes

```
ds² = S² - C²
```

| Regime | ds² Value | Interpretation |
|--------|-----------|----------------|
| **Space-like** | ds² < 0 | Exploring, ambiguous, not understood |
| **Light-like** | ds² ≈ 0 | Critical transition, "aha moment" |
| **Time-like** | ds² > 0 | Settled, clear, understood |

---

## Advanced Usage

### Compare Methods

```python
from src.eigen_text_core import understanding_loop

text = "Water flows downhill"

# Geometric method (continuous)
M_geo, _, metrics_geo = understanding_loop(text, method='geometric')

# XOR method (discrete)
M_xor, _, metrics_xor = understanding_loop(text, method='xor')

print(f"Geometric: {metrics_geo['iterations']} iterations")
print(f"XOR: {metrics_xor['iterations']} iterations")
```

### Extract Semantic Triad

```python
from src.eigen_text_core import extract_LRV_from_sentence

triad = extract_LRV_from_sentence("The cat sat on the mat")

print(f"L (Lexical/Subject): {triad.L.shape}")
print(f"R (Relational/Verb): {triad.R.shape}")
print(f"V (Value/Object): {triad.V.shape}")
```

### Compute Understanding Directly

```python
from src.eigen_text_core import (
    extract_LRV_from_sentence,
    compute_M_geometric,
    compute_M_xor
)

# Extract triad
triad = extract_LRV_from_sentence("The wind bends the tree")

# Compute M using geometric method
M_geo = compute_M_geometric(triad.L, triad.R, triad.V)

# Or using XOR method
M_xor = compute_M_xor(triad.L, triad.R, triad.V)
```

### Track Convergence History

```python
M, history, metrics = understanding_loop("Example sentence", verbose=False)

# Access iteration history
print(f"Total iterations: {len(history)}")
print(f"Alignment history: {metrics['alignment_history']}")
print(f"C history: {metrics['C_history']}")
print(f"S history: {metrics['S_history']}")
print(f"ds² history: {metrics['ds2_history']}")
print(f"Regime history: {metrics['regime_history']}")
```

---

## Parameters

### `understanding_loop()`

```python
understanding_loop(
    text,                    # Input sentence
    max_iterations=100,      # Maximum refinement steps
    method='geometric',      # 'geometric' or 'xor'
    learning_rate=0.1,       # Refinement step size
    verbose=False            # Print iteration details
)
```

**Returns**:
- `M_final`: Final understanding vector (numpy array)
- `M_history`: List of M at each iteration
- `metrics`: Dictionary with convergence statistics

---

## Troubleshooting

### "No module named 'numpy'"

```bash
pip install numpy
```

### Import errors

Make sure to add the repo to path:
```python
import sys
sys.path.insert(0, '/home/user/EigenAI')
```

### Sentence doesn't converge

This is **expected** for ambiguous or contradictory sentences. The system may:
- Find a periodic eigenstate (oscillating interpretation)
- Not converge at all (fundamental ambiguity)

Try:
- Increase `max_iterations`
- Check the `eigenstate_type` in metrics
- Look for periodic orbits (period-2, period-8)

---

## Examples

### Clear Sentence
```python
M, _, m = understanding_loop("The cat sat on the mat")
# Usually: 2-3 iterations, fixed-point, time-like
```

### Ambiguous Sentence
```python
M, _, m = understanding_loop("Time flies like an arrow")
# May: take more iterations or show periodic behavior
```

### Nonsensical Input
```python
M, _, m = understanding_loop("Colorless green ideas sleep furiously")
# May: not converge or take many iterations
```

---

## What the System Does

1. **Extract (L, R, V)** from text
   - L = Lexical (subject/agent)
   - R = Relational (verb/transformation)
   - V = Value (object/patient)

2. **Compute M = L ⊕ R ⊕ V**
   - 45° bisection creates curvature
   - Two 45° angles (tangent + normal)

3. **Iterate until eigenstate**
   - Refine (L,R,V) based on M
   - Check convergence (alignment > 0.99)
   - Detect periodic orbits (period-2 through period-8)

4. **Classify regime**
   - ds² = S² - C²
   - Space-like / Light-like / Time-like

5. **Return understanding**
   - Fixed-point: stable meaning
   - Periodic: oscillating interpretation
   - None: ambiguous/contradictory

---

## Next Steps

- Try ambiguous sentences
- Compare geometric vs XOR methods
- Examine convergence trajectories
- Integrate with Eigen-TM VM (upcoming)

---

For questions or issues, see README.md or contact mcreynolds.jon@gmail.com
