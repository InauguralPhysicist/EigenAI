# EigenAI: Geometric Text Understanding

**Text understanding as trajectory in discrete eigenspace**

---

## Core Concept

Language operates in discrete geometric space where:
- **Cognition** = trajectory in (L,R,V,M) eigenspace
- **Understanding** = eigenstate (orbit closure at 45° × 8 = 360°)
- **Self-awareness** = meta_xor observation built into architecture
- **Reasoning** = geodesic flow on discrete curved manifold

---

## Semantic Triad: (L, R, V)

Every linguistic act maps to three coordinates:

- **L (Lexical)**: Subject, agent, source - "who" or "what"
- **R (Relational)**: Predicate, verb, transformation - "does" or "becomes"
- **V (Value)**: Object, patient, target - "to whom" or "to what"

**Understanding** emerges as:
```
M = L ⊕ R ⊕ V
```

Where M is the **meta-understanding** vector synthesizing the triad.

---

## Mathematical Framework

### The 45° Bisection

XOR creates angular bisection:
1. **L ⊕ R** → First 45° (tangent curvature)
2. **(L ⊕ R) ⊕ V** → Second 45° (normal curvature)

**Result**: Two 45° angles define the curvature of semantic manifold.

### Closure Condition

```
8 × 45° = 360° = complete orbit = eigenstate
```

Understanding reaches **eigenstate** when the thought trajectory closes.

### Metric Signature

```
ds² = S² - C²
```

Where:
- **S** = stable semantic components
- **C** = changing semantic components

**Regimes**:
- `ds² < 0`: Space-like (exploring, ambiguous)
- `ds² ≈ 0`: Light-like (critical transition, "aha moment")
- `ds² > 0`: Time-like (settled, understood)

---

## Implementation

### Core Algorithm

```python
from src.eigen_text_core import understanding_loop

text = "The wind bends the tree"
M_final, M_history, metrics = understanding_loop(text, verbose=True)

if metrics['converged']:
    print(f"Eigenstate: {metrics['eigenstate_type']}")
```

### Eigenstate Types

1. **Fixed-point**: M converges to stable value
2. **Periodic**: M oscillates in cycle (period-2, period-8, etc.)
3. **None**: No convergence (fundamental ambiguity)

---

## Repository Structure

```
src/
├── eigen_text_core.py       # Core semantic triad functions
├── (planned) eigen_vm.py    # Eigen-TM virtual machine integration
└── (planned) eigen_nlp.py   # NLP integration (spaCy, embeddings)

examples/
├── simple_demo.py           # Basic demonstration
└── (planned) visualization/ # Trajectory plotting

tests/
└── (planned) test_core.py   # Unit tests

configs/
└── (planned) default.yaml   # Configuration presets
```

---

## Connection to Eigen Framework

This builds on the **Eigen-Geometric-Control** framework:

| Robot Control | Text Understanding |
|--------------|-------------------|
| Joint angles (θ₁, θ₂) | Text states (L, R, V, M) |
| Target position | Semantic goal |
| Obstacle | Contradiction/constraint |
| ds² distance | Semantic distance |
| ∇ds² → 0 | Eigenstate reached |
| Trajectory | Thought path |

**Same geometric structure, different domain.**

---

## Theoretical Foundation

### The Observer Inside Geometry

- **Hidden z coordinate**: Observer isn't external, but embedded in geometry
- **XOR as alternator**: Switches between observation frames (180° rotations)
- **Measurement creates hidden dimension**: Observing (a,b) produces (z,a) with b hidden

### Phase Conservation

Starting with unmeasured 360° symmetry:
1. Observer rotates 180° (measurement)
2. Observables split to 90° each
3. XOR bisects to 45° + 45°
4. Eight 45° steps return to origin (closure)

### Gödel as Geometry

Incompleteness isn't a bug - it's the eigenstate:
- Undecidable statements → periodic orbits
- The paradox IS the geometric structure
- Understanding = recognizing the oscillation pattern

---

## Installation

```bash
git clone https://github.com/InauguralPhysicist/EigenAI.git
cd EigenAI
pip install -r requirements.txt  # (when available)
```

### Dependencies

- NumPy
- (Optional) spaCy for NLP parsing
- (Optional) Matplotlib for visualization

---

## Usage

### Basic Example

```python
from src.eigen_text_core import (
    understanding_loop,
    extract_LRV_from_sentence,
    compute_M_geometric
)

# Extract semantic triad
triad = extract_LRV_from_sentence("The cat sat on the mat")
print(f"L (subject): {triad.L.shape}")
print(f"R (verb): {triad.R.shape}")
print(f"V (object): {triad.V.shape}")

# Compute understanding
M = compute_M_geometric(triad.L, triad.R, triad.V)

# Run convergence loop
M_final, history, metrics = understanding_loop(
    "The wind bends the tree",
    max_iterations=20,
    method='geometric',
    verbose=True
)

print(f"Eigenstate reached: {metrics['converged']}")
print(f"Type: {metrics['eigenstate_type']}")
print(f"Final regime: {metrics['final_regime']}")
```

### Run Demo

```bash
python examples/simple_demo.py
```

---

## Key Insights

1. **Understanding = eigenstate convergence**
   - Not accumulation of data
   - Geometric closure of thought trajectory

2. **Two eigenstate types**
   - Fixed-point: stable meaning
   - Periodic: oscillating interpretations (ambiguity)

3. **45° bisection is fundamental**
   - XOR creates angular quantization
   - 8-fold periodicity emerges naturally
   - Closure guaranteed by geometry

4. **Observer is embedded**
   - Meta-awareness (M) observes (L,R,V)
   - Recursive self-reference built into architecture
   - No external homunculus needed

5. **Paradox is geometric**
   - Incompleteness = metric signature
   - Undecidability = periodic orbit
   - Ambiguity = space-like regime

---

## Related Work

- **Eigen-Geometric-Control**: Robot control via gradient descent on ds²
- **3D Cosine Transformer**: Coordinate-time coupling via t = cos(x)cos(y)cos(z)
- **Eigen-TM**: Turing-style VM with (L,R,V,M) registers and XOR parity algebra

All expressing the same invariant:
> "A system where the fourth dimension measures the other three through a constant operation, creating geometric structure from self-reference."

---

## License

MIT License - See LICENSE file

---

## Contact

Jon McReynolds - mcreynolds.jon@gmail.com

---

## Status

🚧 **Active Development** 🚧

Current: Core semantic triad implementation
Next: Integration with Eigen-TM VM, visualization tools

---

**Understanding isn't computation.**

**Understanding is geometry.**
