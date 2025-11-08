#!/usr/bin/env python3
"""
Alternative test: Instead of measuring iteration count,
measure the ARC LENGTH of the trajectory through understanding space.

Hypothesis: High-entropy sentences create curved paths (longer arcs)
even if they converge in same number of iterations.
"""

import numpy as np
import math
from src.eigen_text_core import (
    extract_LRV_syntactic_entropy_weighted,
    compute_M_geometric,
)

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


def understanding_trajectory_with_perturbation(text, perturbation_scale=0.0, steps=20):
    """
    Trace understanding trajectory with initial perturbation.

    Instead of iterating until convergence, take fixed number of steps
    and measure total arc length.
    """
    # Extract with entropy weighting
    triad = extract_LRV_syntactic_entropy_weighted(text, word_freq_model=WORD_FREQ)

    L, R, V = triad.L.copy(), triad.R.copy(), triad.V.copy()

    # Add perturbation (noise proportional to sentence entropy)
    if perturbation_scale > 0:
        entropy = sentence_entropy(text)
        noise_strength = perturbation_scale * (entropy / 15.0)  # Normalize

        L = L + np.random.randn(len(L)) * noise_strength
        R = R + np.random.randn(len(R)) * noise_strength
        V = V + np.random.randn(len(V)) * noise_strength

        # Renormalize
        L = L / (np.linalg.norm(L) + 1e-10)
        R = R / (np.linalg.norm(R) + 1e-10)
        V = V / (np.linalg.norm(V) + 1e-10)

    # Trace trajectory
    M_trajectory = []
    learning_rate = 0.1

    for step in range(steps):
        M = compute_M_geometric(L, R, V)
        M_trajectory.append(M.copy())

        # Refine
        decay = np.exp(-step / steps)
        L = L + learning_rate * (M - L) * decay
        R = R + learning_rate * (M - R) * decay
        V = V + learning_rate * (M - V) * decay

        # Normalize
        L = L / (np.linalg.norm(L) + 1e-10)
        R = R / (np.linalg.norm(R) + 1e-10)
        V = V / (np.linalg.norm(V) + 1e-10)

    # Calculate arc length
    arc_length = 0.0
    for i in range(1, len(M_trajectory)):
        segment = np.linalg.norm(M_trajectory[i] - M_trajectory[i-1])
        arc_length += segment

    # Calculate curvature (sum of direction changes)
    curvature = 0.0
    if len(M_trajectory) >= 3:
        for i in range(1, len(M_trajectory)-1):
            v1 = M_trajectory[i] - M_trajectory[i-1]
            v2 = M_trajectory[i+1] - M_trajectory[i]

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 1e-10 and norm2 > 1e-10:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                curvature += angle

    return {
        'trajectory': M_trajectory,
        'arc_length': arc_length,
        'curvature': curvature,
        'final_M': M_trajectory[-1],
    }


print("=" * 80)
print("CURVATURE HYPOTHESIS TEST")
print("=" * 80)
print("\nInstead of counting iterations, we measure:")
print("  - ARC LENGTH: Total distance traveled through understanding space")
print("  - CURVATURE: Sum of direction changes (bending)")
print("\nHypothesis: High-entropy sentences create curved trajectories")

test_sentences = [
    ("Very simple", "It is"),
    ("Simple common", "Dogs bark"),
    ("Simple medium", "The cat sat on the mat"),
    ("Medium coherent", "Water flows downhill"),
    ("Dense uncommon", "The quantum phenomenon demonstrates entanglement"),
    ("Complex technical", "Quantum entanglement demonstrates non-local correlations"),
]

perturbation_levels = [0.0, 0.1, 0.3]

for perturb in perturbation_levels:
    print("\n" + "=" * 80)
    print(f"PERTURBATION = {perturb}")
    print("=" * 80)

    results = []

    for name, sentence in test_sentences:
        entropy = sentence_entropy(sentence)

        # Run 5 trials and average (due to random perturbation)
        arc_lengths = []
        curvatures = []

        for trial in range(5):
            metrics = understanding_trajectory_with_perturbation(
                sentence,
                perturbation_scale=perturb,
                steps=20
            )
            arc_lengths.append(metrics['arc_length'])
            curvatures.append(metrics['curvature'])

        avg_arc = sum(arc_lengths) / len(arc_lengths)
        avg_curve = sum(curvatures) / len(curvatures)

        results.append((name, sentence, entropy, avg_arc, avg_curve))

        print(f"\n{name:20s} (entropy={entropy:.2f})")
        print(f"  Arc length: {avg_arc:.6f}")
        print(f"  Curvature:  {avg_curve:.6f} radians")

    # Correlation analysis
    print("\n" + "-" * 80)
    print("CORRELATION ANALYSIS")
    print("-" * 80)

    entropies = [e for _, _, e, _, _ in results]
    arc_lengths = [a for _, _, _, a, _ in results]
    curvatures = [c for _, _, _, _, c in results]

    # Entropy vs arc length
    if len(entropies) > 2:
        mean_e = sum(entropies) / len(entropies)
        mean_a = sum(arc_lengths) / len(arc_lengths)

        num = sum((e - mean_e) * (a - mean_a) for e, a in zip(entropies, arc_lengths))
        denom_e = math.sqrt(sum((e - mean_e)**2 for e in entropies))
        denom_a = math.sqrt(sum((a - mean_a)**2 for a in arc_lengths))

        if denom_e > 0 and denom_a > 0:
            corr_arc = num / (denom_e * denom_a)
            print(f"Correlation (entropy vs arc_length): {corr_arc:.3f}")
        else:
            corr_arc = 0.0
            print("Correlation (entropy vs arc_length): No variance")

        # Entropy vs curvature
        mean_c = sum(curvatures) / len(curvatures)
        num = sum((e - mean_e) * (c - mean_c) for e, c in zip(entropies, curvatures))
        denom_c = math.sqrt(sum((c - mean_c)**2 for c in curvatures))

        if denom_e > 0 and denom_c > 0:
            corr_curve = num / (denom_e * denom_c)
            print(f"Correlation (entropy vs curvature):  {corr_curve:.3f}")
        else:
            corr_curve = 0.0
            print("Correlation (entropy vs curvature):  No variance")

        # Assessment
        if abs(corr_arc) > 0.5 or abs(corr_curve) > 0.5:
            print("\n✓ MODERATE-STRONG correlation detected!")
        elif abs(corr_arc) > 0.3 or abs(corr_curve) > 0.3:
            print("\n✓ WEAK correlation detected")
        else:
            print("\n✗ No significant correlation")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
If correlation emerges with perturbation:
  → High-entropy sentences DO create curved trajectories
  → The curvature was hidden because we started too close to eigenstate
  → Arc length (not iteration count) measures information geometry

If no correlation even with perturbation:
  → Entropy weighting affects initial geometry but not trajectory
  → Need different mechanism (e.g., adaptive learning rate)
  → Or hypothesis needs refinement
""")
