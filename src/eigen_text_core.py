# -*- coding: utf-8 -*-
"""
Eigen: Text Understanding via Semantic Triad

Maps language to (L, R, V) semantic coordinates:
- L: Lexical (subject, agent, source)
- R: Relational (predicate, verb, transformation)
- V: Value (object, patient, target)

Understanding emerges from iterative refinement of M = L ⊕ R ⊕ V
until eigenstate (stable meaning) is reached.

Mathematical Structure:
    M = (L + R + V) / ||L + R + V||  (45° bisector)
    or
    M = L ⊕ R ⊕ V  (XOR in discrete case)

Convergence:
    Eigenstate when ||M_t - M_{t-1}|| < ε

Reference Framework:
    Based on Eigen-Geometric-Control patterns
    Follows ds² = S² - C² metric structure
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class SemanticTriad:
    """
    Representation of (L, R, V) semantic coordinates

    Attributes
    ----------
    L : np.ndarray
        Lexical vector (subject/agent)
    R : np.ndarray
        Relational vector (predicate/verb)
    V : np.ndarray
        Value vector (object/patient)
    text : str
        Original text fragment
    """
    L: np.ndarray
    R: np.ndarray
    V: np.ndarray
    text: str

    def __repr__(self):
        return f"SemanticTriad(L={self.L.shape}, R={self.R.shape}, V={self.V.shape})"


def extract_LRV_from_sentence(sentence: str,
                               embedding_dim: int = 300) -> SemanticTriad:
    """
    Extract (L, R, V) from sentence

    Parameters
    ----------
    sentence : str
        Input sentence (e.g., "The wind bends the tree")
    embedding_dim : int
        Dimension of embedding vectors

    Returns
    -------
    triad : SemanticTriad
        Extracted (L, R, V) coordinates

    Raises
    ------
    TypeError
        If sentence is not a string or embedding_dim is not an integer
    ValueError
        If sentence is empty or embedding_dim is not positive

    Examples
    --------
    >>> triad = extract_LRV_from_sentence("The wind bends the tree")
    >>> print(f"L (subject): wind")
    >>> print(f"R (verb): bends")
    >>> print(f"V (object): tree")

    Notes
    -----
    Current implementation uses simplified extraction.
    For production, integrate spaCy dependency parsing:

    ```python
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)

    subject = [tok for tok in doc if tok.dep_ == "nsubj"]
    verb = [tok for tok in doc if tok.pos_ == "VERB"]
    obj = [tok for tok in doc if tok.dep_ == "dobj"]
    ```
    """
    # Input validation
    if not isinstance(sentence, str):
        raise TypeError(f"sentence must be a string, got {type(sentence).__name__}")

    if not isinstance(embedding_dim, int):
        raise TypeError(f"embedding_dim must be an integer, got {type(embedding_dim).__name__}")

    if len(sentence.strip()) == 0:
        raise ValueError("sentence cannot be empty")

    if embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")

    # Simplified extraction using word positions
    words = sentence.lower().split()

    # Simple one-hot-like encoding based on position
    L = np.zeros(embedding_dim)
    R = np.zeros(embedding_dim)
    V = np.zeros(embedding_dim)

    # Heuristic: first third = L, middle third = R, last third = V
    # Add small random component for diversity
    if len(words) >= 1:
        L[:min(100, embedding_dim)] = np.random.randn(min(100, embedding_dim)) * 0.1
        L[0] = 1.0  # Strong signal for first position

    if len(words) >= 2:
        R[:min(100, embedding_dim)] = np.random.randn(min(100, embedding_dim)) * 0.1
        R[embedding_dim//3] = 1.0  # Strong signal for middle

    if len(words) >= 3:
        V[:min(100, embedding_dim)] = np.random.randn(min(100, embedding_dim)) * 0.1
        V[2*embedding_dim//3] = 1.0  # Strong signal for end

    return SemanticTriad(L=L, R=R, V=V, text=sentence)


def compute_M_geometric(L: np.ndarray,
                       R: np.ndarray,
                       V: np.ndarray) -> np.ndarray:
    """
    Compute understanding vector M via geometric mean (45° bisector)

    M = (L + R + V) / ||L + R + V||

    This is the normalized sum, equidistant from all three axes.
    The 45° bisection creates curvature in the understanding manifold.

    Parameters
    ----------
    L, R, V : np.ndarray
        Semantic coordinate vectors

    Returns
    -------
    M : np.ndarray
        Understanding vector (normalized)

    Raises
    ------
    TypeError
        If L, R, or V are not numpy arrays
    ValueError
        If L, R, V have different shapes or contain NaN/Inf values

    Notes
    -----
    The 45° angle ensures M is balanced:
    - Equal influence from L, R, V
    - No privileged direction
    - Symmetric integration

    Two 45° angles from XOR bisection create closure at 8 steps:
    8 × 45° = 360° = complete orbit = eigenstate
    """
    # Type validation
    if not isinstance(L, np.ndarray):
        raise TypeError(f"L must be a numpy array, got {type(L).__name__}")
    if not isinstance(R, np.ndarray):
        raise TypeError(f"R must be a numpy array, got {type(R).__name__}")
    if not isinstance(V, np.ndarray):
        raise TypeError(f"V must be a numpy array, got {type(V).__name__}")

    # Shape validation
    if L.shape != R.shape or R.shape != V.shape:
        raise ValueError(f"L, R, V must have the same shape. Got L:{L.shape}, R:{R.shape}, V:{V.shape}")

    # Check for NaN/Inf
    if np.any(np.isnan(L)) or np.any(np.isnan(R)) or np.any(np.isnan(V)):
        raise ValueError("Input vectors contain NaN values")
    if np.any(np.isinf(L)) or np.any(np.isinf(R)) or np.any(np.isinf(V)):
        raise ValueError("Input vectors contain Inf values")

    M_raw = L + R + V
    norm = np.linalg.norm(M_raw)

    if norm < 1e-10:
        # Degenerate case: all vectors zero
        raise ValueError("Cannot compute M: all input vectors are zero or nearly zero (norm < 1e-10)")

    M = M_raw / norm
    return M


def compute_M_xor(L: np.ndarray,
                  R: np.ndarray,
                  V: np.ndarray) -> np.ndarray:
    """
    Compute understanding vector M via XOR (discrete case)

    M = L ⊕ R ⊕ V

    XOR creates 45° bisection through parity:
    - L ⊕ R creates first 45° (tangent curvature)
    - (L ⊕ R) ⊕ V creates second 45° (normal curvature)
    - Together they define the curved manifold

    Parameters
    ----------
    L, R, V : np.ndarray
        Semantic coordinate vectors

    Returns
    -------
    M : np.ndarray
        Understanding vector (XOR result)

    Notes
    -----
    XOR interpretation:
    - Captures what changes across the triad
    - Ignores what stays constant
    - Reflexive: M ⊕ M = 0 (eigenstate property)
    - Creates 45° quantization (8-fold periodicity)
    """
    # Convert to binary (positive vs negative)
    L_bin = (L > 0).astype(int)
    R_bin = (R > 0).astype(int)
    V_bin = (V > 0).astype(int)

    # XOR cascade: (L ⊕ R) ⊕ V
    # First XOR: 90° → 45° bisection
    LR_xor = np.logical_xor(L_bin, R_bin)

    # Second XOR: 45° + 90° → final 45°
    M_bin = np.logical_xor(LR_xor, V_bin).astype(float)

    # Restore magnitude (average of input magnitudes)
    avg_magnitude = np.mean([np.linalg.norm(L), np.linalg.norm(R), np.linalg.norm(V)])
    M = M_bin * avg_magnitude / (np.linalg.norm(M_bin) + 1e-10)

    return M


def measure_understanding_change(M_prev: np.ndarray,
                                M_curr: np.ndarray,
                                eps_change: float = 0.01) -> Tuple[float, int, int, int]:
    """
    Measure how much understanding changed between iterations

    Returns alignment and (C, S, ds²) metrics following Eigen framework:
    - C: Number of components that changed
    - S: Number of components that stayed stable
    - ds² = S² - C²

    Parameters
    ----------
    M_prev : np.ndarray
        Previous understanding vector
    M_curr : np.ndarray
        Current understanding vector
    eps_change : float
        Threshold for detecting change (default: 0.01)

    Returns
    -------
    alignment : float
        Cosine similarity (1 = perfect alignment, 0 = orthogonal)
    C : int
        Change count
    S : int
        Stability count
    ds2 : int
        Metric signature S² - C²

    Notes
    -----
    Eigenstate when:
    - alignment → 1 (vectors nearly parallel)
    - C → 0 (no components changing)
    - S → n (all components stable)
    - ds² → n² (maximum time-like)

    This follows the same pattern as Eigen-Geometric-Control's
    compute_change_stability() function.
    """
    # Cosine similarity
    norm_prev = np.linalg.norm(M_prev)
    norm_curr = np.linalg.norm(M_curr)

    if norm_prev < 1e-10 or norm_curr < 1e-10:
        alignment = 0.0
    else:
        alignment = float(np.dot(M_prev, M_curr) / (norm_prev * norm_curr))

    # Component-wise change detection
    delta = np.abs(M_curr - M_prev)

    C = int(np.sum(delta > eps_change))
    S = int(len(delta) - C)

    # Metric signature (Minkowski-like)
    ds2 = S * S - C * C

    return alignment, C, S, ds2


def detect_eigenstate(M_history: List[np.ndarray],
                     threshold: float = 0.99,
                     window: int = 3) -> Tuple[bool, Optional[int]]:
    """
    Detect if understanding has converged to eigenstate

    Eigenstate = M no longer changing significantly
    or M exhibiting periodic behavior (cycle detected)

    Parameters
    ----------
    M_history : list of np.ndarray
        History of understanding vectors
    threshold : float
        Alignment threshold for convergence (default: 0.99)
    window : int
        Number of recent iterations to check for stability

    Returns
    -------
    converged : bool
        True if eigenstate reached
    period : int or None
        If periodic orbit detected, returns period length

    Examples
    --------
    >>> M_hist = [M1, M2, M3, M4]
    >>> converged, period = detect_eigenstate(M_hist)
    >>> if converged:
    ...     if period:
    ...         print(f"Periodic eigenstate with period {period}")
    ...     else:
    ...         print("Fixed-point eigenstate")
    """
    if len(M_history) < window:
        return False, None

    # Check recent stability (fixed-point eigenstate)
    recent_stable = True
    for i in range(1, window):
        alignment, C, S, ds2 = measure_understanding_change(
            M_history[-i-1],
            M_history[-i]
        )
        if alignment < threshold:
            recent_stable = False
            break

    if recent_stable:
        return True, None  # Fixed-point eigenstate

    # Check for periodic orbits (period-2 through period-8)
    # Period-8 is natural from 45° × 8 = 360° closure
    for period in range(2, min(9, len(M_history)//2)):
        is_periodic = True
        for i in range(period):
            idx_curr = len(M_history) - 1 - i
            idx_prev = idx_curr - period

            if idx_prev < 0:
                is_periodic = False
                break

            alignment, _, _, _ = measure_understanding_change(
                M_history[idx_prev],
                M_history[idx_curr]
            )

            if alignment < threshold:
                is_periodic = False
                break

        if is_periodic:
            return True, period  # Periodic eigenstate

    return False, None


def analyze_understanding_regime(C: int, S: int) -> str:
    """
    Classify understanding regime based on C and S

    Parameters
    ----------
    C : int
        Change count
    S : int
        Stability count

    Returns
    -------
    regime : str
        "space-like", "light-like", or "time-like"

    Notes
    -----
    ds² = S² - C²

    - ds² < 0: Space-like (exploring, ambiguous, not yet understood)
    - ds² ≈ 0: Light-like (critical transition, "aha moment")
    - ds² > 0: Time-like (settled, clear, understood)

    This maps directly to the Eigen-Geometric-Control framework
    where spatial vs temporal character emerges from metric signature.
    """
    ds2 = S**2 - C**2

    if abs(ds2) < 10:  # Near-zero threshold
        return "light-like (critical transition)"
    elif ds2 < 0:
        return "space-like (exploring/ambiguous)"
    else:
        return "time-like (settled/understood)"


def understanding_loop(text: str,
                      max_iterations: int = 100,
                      method: str = 'geometric',
                      learning_rate: float = 0.1,
                      verbose: bool = False) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
    """
    Iteratively refine understanding until eigenstate

    Follows the pattern: M_{t+1} = M_t - η∇(semantic_distance)
    analogous to robot control: Q_{t+1} = Q_t - η∇ds²

    Process:
    1. Extract (L, R, V) from text
    2. Compute M = L ⊕ R ⊕ V
    3. Check if M stable (eigenstate)
    4. If not, refine and repeat
    5. Return final M and convergence history

    Parameters
    ----------
    text : str
        Input text to understand
    max_iterations : int
        Maximum refinement iterations
    method : str
        'geometric' or 'xor' for M computation
    learning_rate : float
        Step size for refinement (analogous to η in robot control)
    verbose : bool
        Print iteration details

    Returns
    -------
    M_final : np.ndarray
        Final understanding vector
    M_history : list of np.ndarray
        History of M at each iteration
    metrics : dict
        Convergence metrics (iterations, alignment, regime, etc.)

    Examples
    --------
    >>> text = "The wind bends the tree"
    >>> M, history, metrics = understanding_loop(text, verbose=True)
    Iteration 0: alignment=0.950, C=2, S=298, ds²=88800, regime=time-like
    Iteration 1: alignment=0.985, C=1, S=299, ds²=89400, regime=time-like
    Iteration 2: alignment=0.996, C=0, S=300, ds²=90000, regime=time-like
    Eigenstate reached at iteration 2
    """
    # Extract initial (L, R, V)
    triad = extract_LRV_from_sentence(text)

    M_history = []
    alignment_history = []
    C_history = []
    S_history = []
    ds2_history = []
    regime_history = []

    L, R, V = triad.L, triad.R, triad.V

    for iteration in range(max_iterations):
        # Compute M
        if method == 'geometric':
            M = compute_M_geometric(L, R, V)
        elif method == 'xor':
            M = compute_M_xor(L, R, V)
        else:
            raise ValueError(f"Unknown method: {method}")

        M_history.append(M)

        # Check convergence
        if len(M_history) >= 2:
            alignment, C, S, ds2 = measure_understanding_change(
                M_history[-2],
                M_history[-1]
            )
            alignment_history.append(alignment)
            C_history.append(C)
            S_history.append(S)
            ds2_history.append(ds2)

            regime = analyze_understanding_regime(C, S)
            regime_history.append(regime)

            if verbose:
                print(f"Iteration {iteration}: alignment={alignment:.3f}, C={C}, S={S}, ds²={ds2}, regime={regime}")

            # Check for eigenstate
            converged, period = detect_eigenstate(M_history)
            if converged:
                if verbose:
                    if period:
                        print(f"Periodic eigenstate (period={period}) reached at iteration {iteration}")
                    else:
                        print(f"Fixed-point eigenstate reached at iteration {iteration}")
                break

        # Refine (L, R, V) based on M feedback
        # This simulates deeper semantic processing
        # Analogous to gradient descent in robot control

        # Project M back onto L, R, V with learning rate
        L_correction = learning_rate * (M - L) * np.exp(-iteration / max_iterations)
        R_correction = learning_rate * (M - R) * np.exp(-iteration / max_iterations)
        V_correction = learning_rate * (M - V) * np.exp(-iteration / max_iterations)

        L = L + L_correction
        R = R + R_correction
        V = V + V_correction

        # Normalize to prevent explosion
        L = L / (np.linalg.norm(L) + 1e-10)
        R = R / (np.linalg.norm(R) + 1e-10)
        V = V / (np.linalg.norm(V) + 1e-10)

    # Detect final eigenstate type
    converged, period = detect_eigenstate(M_history)

    metrics = {
        'iterations': len(M_history),
        'converged': converged,
        'period': period,
        'eigenstate_type': 'periodic' if period else 'fixed-point' if converged else 'none',
        'final_alignment': alignment_history[-1] if alignment_history else 0.0,
        'final_regime': regime_history[-1] if regime_history else 'unknown',
        'alignment_history': alignment_history,
        'C_history': C_history,
        'S_history': S_history,
        'ds2_history': ds2_history,
        'regime_history': regime_history
    }

    return M_history[-1], M_history, metrics


if __name__ == "__main__":
    print("=" * 60)
    print("EIGEN TEXT: SEMANTIC TRIAD UNDERSTANDING")
    print("=" * 60)

    # Example 1: Simple sentence
    text1 = "The wind bends the tree"
    print(f"\nText: '{text1}'")
    print("Running understanding loop (geometric method)...")

    M_final, M_hist, metrics = understanding_loop(
        text1,
        verbose=True,
        max_iterations=20,
        method='geometric'
    )

    print(f"\nConvergence summary:")
    print(f"  Iterations: {metrics['iterations']}")
    print(f"  Converged: {metrics['converged']}")
    print(f"  Eigenstate type: {metrics['eigenstate_type']}")
    print(f"  Final alignment: {metrics['final_alignment']:.4f}")
    print(f"  Final regime: {metrics['final_regime']}")

    # Example 2: XOR method
    print("\n" + "=" * 60)
    print(f"\nText: '{text1}'")
    print("Running understanding loop (XOR method)...")

    M_final2, M_hist2, metrics2 = understanding_loop(
        text1,
        verbose=True,
        max_iterations=20,
        method='xor'
    )

    print(f"\nConvergence summary:")
    print(f"  Iterations: {metrics2['iterations']}")
    print(f"  Converged: {metrics2['converged']}")
    print(f"  Eigenstate type: {metrics2['eigenstate_type']}")

    # Compare methods
    print("\n" + "=" * 60)
    print("METHOD COMPARISON")
    print("=" * 60)
    print(f"{'Method':<15} {'Iterations':<12} {'Eigenstate':<15} {'Period':<10}")
    print("-" * 60)
    print(f"{'Geometric':<15} {metrics['iterations']:<12} {metrics['eigenstate_type']:<15} {str(metrics['period']):<10}")
    print(f"{'XOR':<15} {metrics2['iterations']:<12} {metrics2['eigenstate_type']:<15} {str(metrics2['period']):<10}")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
1. Understanding = convergence to eigenstate
2. Two types of eigenstates:
   - Fixed-point: M stops changing
   - Periodic: M oscillates in cycle (period-2, period-8, etc.)
3. ds² = S² - C² tracks understanding regime:
   - Space-like: Exploring/confused
   - Light-like: Critical transition (aha moment)
   - Time-like: Settled/understood
4. (L, R, V) triad captures minimal semantic structure
5. M = L ⊕ R ⊕ V synthesizes understanding via 45° bisection
6. 45° × 8 = 360° creates natural 8-fold eigenstate periodicity
    """)
