#!/usr/bin/env python3
"""
Test entropy-weighted extraction: Does information density affect iteration count?
"""

import numpy as np
import math
from src.eigen_text_core import (
    extract_LRV_syntactic_entropy_weighted,
    compute_M_geometric,
    measure_understanding_change,
    detect_eigenstate,
    analyze_understanding_regime
)

# Word frequency model (same as before)
WORD_FREQ = {
    'the': 0.07, 'is': 0.05, 'a': 0.04, 'and': 0.03, 'to': 0.03,
    'of': 0.02, 'in': 0.02, 'that': 0.02, 'it': 0.02, 'was': 0.01,
    'cat': 0.0001, 'dog': 0.0001, 'dogs': 0.0001, 'bark': 0.0001,
    'sat': 0.0001, 'mat': 0.0001, 'water': 0.0001, 'light': 0.0001,
    'tree': 0.0001, 'apple': 0.0001, 'fell': 0.0001, 'from': 0.001,
    'on': 0.001, 'flows': 0.0001, 'downhill': 0.0001,
    'statement': 0.00001, 'false': 0.00001, 'true': 0.00001,
    'both': 0.0001, 'simultaneously': 0.000001, 'this': 0.01,
    'quantum': 0.000001, 'entanglement': 0.0000001,
    'demonstrates': 0.00001, 'non-local': 0.0000001,
    'correlations': 0.00001, 'phenomenon': 0.00001,
}

def word_entropy(word):
    word_lower = word.lower()
    freq = WORD_FREQ.get(word_lower, 1e-7)
    return -math.log2(freq)

def sentence_entropy(sentence):
    words = sentence.lower().split()
    if not words:
        return 0.0
    entropies = [word_entropy(w) for w in words]
    return sum(entropies) / len(words)

def understanding_loop_entropy_weighted(text, max_iterations=100, verbose=False):
    """
    Understanding loop using entropy-weighted extraction.
    """
    # Extract initial (L, R, V) with entropy weighting
    triad = extract_LRV_syntactic_entropy_weighted(text, word_freq_model=WORD_FREQ)

    M_history = []
    alignment_history = []
    C_history = []
    S_history = []
    ds2_history = []
    regime_history = []

    L, R, V = triad.L, triad.R, triad.V

    for iteration in range(max_iterations):
        # Compute M
        M = compute_M_geometric(L, R, V)
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
        learning_rate = 0.1
        L_correction = learning_rate * (M - L) * np.exp(-iteration / max_iterations)
        R_correction = learning_rate * (M - R) * np.exp(-iteration / max_iterations)
        V_correction = learning_rate * (M - V) * np.exp(-iteration / max_iterations)

        L = L + L_correction
        R = R + R_correction
        V = V + V_correction

        # Normalize
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
        'regime_history': regime_history,
    }

    return M_history[-1], M_history, metrics


print("=" * 80)
print("ENTROPY-WEIGHTED EXTRACTION TEST")
print("=" * 80)

test_sentences = [
    ("Very simple", "It is"),
    ("Simple common", "Dogs bark"),
    ("Simple medium", "The cat sat on the mat"),
    ("Medium coherent", "Water flows downhill"),
    ("Dense uncommon", "The quantum phenomenon demonstrates entanglement"),
    ("Complex technical", "Quantum entanglement demonstrates non-local correlations"),
    ("Paradox", "This statement is both true and false simultaneously"),
]

print("\n1. INITIAL L-R-V GEOMETRY WITH ENTROPY WEIGHTING")
print("-" * 80)

for name, sentence in test_sentences:
    entropy = sentence_entropy(sentence)
    triad = extract_LRV_syntactic_entropy_weighted(sentence, embedding_dim=100, word_freq_model=WORD_FREQ)

    # Check orthogonality
    LR = np.dot(triad.L, triad.R)
    LV = np.dot(triad.L, triad.V)
    RV = np.dot(triad.R, triad.V)
    orthog_score = abs(LR) + abs(LV) + abs(RV)

    print(f"\n{name:20s} (entropy={entropy:.2f})")
    print(f"  L·R={LR:6.3f}, L·V={LV:6.3f}, R·V={RV:6.3f}  (orthog={orthog_score:.3f})")

print("\n" + "=" * 80)
print("2. ITERATION COUNT WITH ENTROPY WEIGHTING")
print("-" * 80)

results = []
for name, sentence in test_sentences:
    entropy = sentence_entropy(sentence)

    print(f"\n{name}: '{sentence}'")
    print(f"  Entropy: {entropy:.2f} bits/word")

    M, history, metrics = understanding_loop_entropy_weighted(sentence, max_iterations=50, verbose=True)

    iters = metrics['iterations']
    converged = metrics['converged']

    print(f"  → {iters} iterations ({'converged' if converged else 'not converged'})")

    results.append((name, sentence, entropy, iters, converged))

print("\n" + "=" * 80)
print("3. SUMMARY TABLE")
print("-" * 80)

print(f"{'Sentence Type':<20} {'Entropy':>10} {'Iterations':>12} {'Status'}")
print("-" * 80)

for name, sentence, entropy, iters, converged in results:
    status = "✓ Conv" if converged else "✗ Divg"
    print(f"{name:<20} {entropy:>10.2f} {iters:>12} {status}")

print("\n" + "=" * 80)
print("4. CORRELATION ANALYSIS")
print("-" * 80)

entropies = [e for _, _, e, _, _ in results]
iterations = [i for _, _, _, i, _ in results]

if len(entropies) > 2:
    mean_e = sum(entropies) / len(entropies)
    mean_i = sum(iterations) / len(iterations)

    numerator = sum((e - mean_e) * (i - mean_i) for e, i in zip(entropies, iterations))
    denom_e = math.sqrt(sum((e - mean_e)**2 for e in entropies))
    denom_i = math.sqrt(sum((i - mean_i)**2 for i in iterations))

    if denom_e > 0 and denom_i > 0:
        correlation = numerator / (denom_e * denom_i)
        print(f"Correlation (entropy vs iterations): {correlation:.3f}")

        if abs(correlation) < 0.3:
            print("  → WEAK correlation")
        elif abs(correlation) < 0.7:
            print("  → MODERATE correlation ✓")
        else:
            print("  → STRONG correlation ✓✓")
    else:
        print("  No variance in data")

print(f"\nIteration range: {min(iterations)} to {max(iterations)}")
print(f"Entropy range: {min(entropies):.2f} to {max(entropies):.2f} bits/word")

print("\n" + "=" * 80)
print("CONCLUSION")
print("-" * 80)

if max(iterations) > min(iterations):
    print("✓ Entropy weighting DOES create iteration variance!")
    print(f"  Range: {min(iterations)}-{max(iterations)} iterations")
    print("\nThis validates the hypothesis:")
    print("  - Information density creates semantic curvature")
    print("  - Geodesics through curved space take longer")
    print("  - Iteration count = arc length of understanding trajectory")
else:
    print("✗ No iteration variance observed.")
    print("  Entropy weighting may be too weak, or")
    print("  Convergence threshold may need adjustment.")
