#!/usr/bin/env python3
"""Quick debug script to see what's happening with embeddings"""

import numpy as np
from src.eigen_text_core import extract_LRV_from_sentence, understanding_loop

# Test different sentences
sentences = [
    "apple apple apple",  # Repetition
    "The cat sat on the mat",  # Meaningful
    "Cat the mat on sat the",  # Scrambled
    "xkqz mplv jfgh wert",  # Random noise
]

print("=" * 80)
print("TESTING INITIAL L, R, V DIFFERENCES")
print("=" * 80)

for sent in sentences:
    triad = extract_LRV_from_sentence(sent, embedding_dim=100)

    # Check if L, R, V are actually different
    LR_diff = np.linalg.norm(triad.L - triad.R)
    LV_diff = np.linalg.norm(triad.L - triad.V)
    RV_diff = np.linalg.norm(triad.R - triad.V)

    print(f"\n'{sent}':")
    print(f"  ||L - R|| = {LR_diff:.4f}")
    print(f"  ||L - V|| = {LV_diff:.4f}")
    print(f"  ||R - V|| = {RV_diff:.4f}")

print("\n" + "=" * 80)
print("TESTING CONVERGENCE BEHAVIOR")
print("=" * 80)

for sent in sentences:
    print(f"\n'{sent}':")
    M, hist, metrics = understanding_loop(sent, max_iterations=10, verbose=True)
    print(f"  Converged: {metrics['converged']} in {metrics['iterations']} iterations")
    if len(metrics['C_history']) > 0:
        print(f"  C_history: {metrics['C_history'][:5]}")
        print(f"  S_history: {metrics['S_history'][:5]}")
