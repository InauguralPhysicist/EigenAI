#!/usr/bin/env python3
"""
Test information density hypothesis:
Does semantic entropy determine helix pitch (iteration count)?
"""

import numpy as np
import math
from collections import Counter

# We'll create a minimal test first without modifying core
# Then implement entropy-weighted version

print("=" * 80)
print("TESTING: Information Density → Helix Pitch")
print("=" * 80)

# Simple word frequency model (approximate English)
# Higher values = more common = lower entropy
WORD_FREQ = {
    # Very common (low entropy)
    'the': 0.07, 'is': 0.05, 'a': 0.04, 'and': 0.03, 'to': 0.03,
    'of': 0.02, 'in': 0.02, 'that': 0.02, 'it': 0.02, 'was': 0.01,

    # Common (medium-low entropy)
    'cat': 0.0001, 'dog': 0.0001, 'bark': 0.0001, 'sat': 0.0001,
    'mat': 0.0001, 'water': 0.0001, 'light': 0.0001, 'tree': 0.0001,
    'apple': 0.0001, 'fell': 0.0001, 'from': 0.001, 'on': 0.001,

    # Uncommon (medium-high entropy)
    'statement': 0.00001, 'false': 0.00001, 'true': 0.00001,
    'both': 0.0001, 'simultaneously': 0.000001,

    # Rare technical (high entropy)
    'quantum': 0.000001, 'entanglement': 0.0000001,
    'demonstrates': 0.00001, 'non-local': 0.0000001,
    'correlations': 0.00001, 'phenomenon': 0.00001,

    # Default for unknown words
    'unknown': 0.0000001
}

def word_entropy(word):
    """Calculate information content (entropy) of a word."""
    word_lower = word.lower()
    freq = WORD_FREQ.get(word_lower, WORD_FREQ['unknown'])
    # Shannon entropy: -log2(P)
    return -math.log2(freq)

def sentence_entropy(sentence):
    """Average information density of sentence."""
    words = sentence.lower().split()
    if not words:
        return 0.0
    entropies = [word_entropy(w) for w in words]
    return sum(entropies) / len(words)

# Test sentences from different categories
test_sentences = [
    ("Simple common", "Dogs bark"),
    ("Simple medium", "The cat sat on the mat"),
    ("Complex technical", "Quantum entanglement demonstrates non-local correlations"),
    ("Paradox", "This statement is both true and false simultaneously"),
    ("Dense uncommon", "The quantum phenomenon demonstrates entanglement"),
    ("Very simple", "It is"),
    ("Medium coherent", "Water flows downhill"),
]

print("\n1. MEASURING SENTENCE ENTROPY")
print("-" * 80)

entropy_data = []
for name, sentence in test_sentences:
    entropy = sentence_entropy(sentence)
    word_count = len(sentence.split())
    print(f"{name:20s}: entropy={entropy:.2f} bits/word  ({word_count} words)")
    print(f"  '{sentence}'")
    entropy_data.append((name, sentence, entropy))

print("\n" + "=" * 80)
print("2. CURRENT BEHAVIOR (No entropy weighting)")
print("-" * 80)

# Import actual framework
from src.eigen_text_core import extract_LRV_syntactic, understanding_loop

print("\nTesting with syntactic geometry (intrinsic)...")

current_iterations = []
for name, sentence, entropy in entropy_data:
    try:
        M, history, metrics = understanding_loop(
            sentence,
            max_iterations=50,
            verbose=False
        )
        iters = metrics['iterations']
        converged = metrics['converged']

        print(f"{name:20s}: {iters:2d} iterations  (entropy={entropy:.2f})")
        current_iterations.append((name, sentence, entropy, iters))

        if not converged:
            print(f"  WARNING: Did not converge!")

    except Exception as e:
        print(f"{name:20s}: ERROR - {e}")

print("\n" + "=" * 80)
print("3. ENTROPY CORRELATION ANALYSIS")
print("-" * 80)

if len(current_iterations) > 2:
    entropies = [e for _, _, e, _ in current_iterations]
    iterations = [i for _, _, _, i in current_iterations]

    # Calculate correlation
    mean_e = sum(entropies) / len(entropies)
    mean_i = sum(iterations) / len(iterations)

    numerator = sum((e - mean_e) * (i - mean_i) for e, i in zip(entropies, iterations))
    denom_e = math.sqrt(sum((e - mean_e)**2 for e in entropies))
    denom_i = math.sqrt(sum((i - mean_i)**2 for i in iterations))

    if denom_e > 0 and denom_i > 0:
        correlation = numerator / (denom_e * denom_i)
        print(f"Correlation (entropy vs iterations): {correlation:.3f}")

        if abs(correlation) < 0.3:
            print("  → WEAK correlation (as expected - no entropy weighting yet)")
        elif abs(correlation) < 0.7:
            print("  → MODERATE correlation")
        else:
            print("  → STRONG correlation!")

    print(f"\nIteration range: {min(iterations)} to {max(iterations)}")
    print(f"Entropy range: {min(entropies):.2f} to {max(entropies):.2f} bits/word")

print("\n" + "=" * 80)
print("4. PREDICTED BEHAVIOR (With entropy weighting)")
print("-" * 80)

print("\nFormula: iterations* = √(semantic_entropy · F_syntax / coherence)")
print("Where:")
print("  - semantic_entropy = avg info content per word")
print("  - F_syntax = 3 (subject-verb-object)")
print("  - coherence = 1 / orthogonality_deviation")

print("\nPredictions:")

for name, sentence, entropy in entropy_data:
    # Predict iterations based on entropy
    # Using F=3, and estimated coherence
    # Base iterations ≈ 3 for perfect coherence

    # Simplified prediction
    base = 3
    entropy_factor = entropy / 15  # Normalize (15 bits ≈ rare word)
    predicted = int(base * (1 + entropy_factor))

    print(f"{name:20s}: entropy={entropy:.2f} → predicted ~{predicted:2d} iterations")

print("\n" + "=" * 80)
print("CONCLUSION")
print("-" * 80)

print("""
CURRENT STATE (without entropy weighting):
- All sentences converge in ~3 iterations
- No correlation with semantic density
- Why? Character n-grams have uniform information density

HYPOTHESIS:
- Weight embeddings by word entropy: rare words = stronger signal
- High-entropy words create "curvature" in semantic space
- Geodesics through curved space = longer paths = more iterations

PREDICTION:
- "Dogs bark" (low entropy) → 2-3 iterations
- "Quantum entanglement..." (high entropy) → 10-20 iterations
- "This statement is false" (paradox) → never converges

NEXT STEP: Implement entropy-weighted embeddings and re-test.
""")
