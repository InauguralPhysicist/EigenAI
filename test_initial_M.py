#!/usr/bin/env python3
"""Test if initial M values differ for different sentences"""

import numpy as np
from src.eigen_text_core import extract_LRV_from_sentence, compute_M_geometric

sentences = [
    "apple apple apple",  # Repetition
    "The cat sat on the mat",  # Meaningful
    "Cat the mat on sat the",  # Scrambled (same words, different order)
    "xkqz mplv jfgh wert",  # Random noise
]

print("=" * 80)
print("INITIAL M VALUES FOR DIFFERENT SENTENCES")
print("=" * 80)

M_values = []
for sent in sentences:
    triad = extract_LRV_from_sentence(sent, embedding_dim=100)
    M = compute_M_geometric(triad.L, triad.R, triad.V)
    M_values.append(M)

    print(f"\n'{sent}':")
    print(f"  M norm: {np.linalg.norm(M):.6f}")
    print(f"  M[0:5]: {M[0:5]}")

print("\n" + "=" * 80)
print("PAIRWISE M DIFFERENCES (cosine similarity)")
print("=" * 80)

for i, sent1 in enumerate(sentences):
    for j, sent2 in enumerate(sentences):
        if i < j:
            # Cosine similarity (1 = identical, 0 = orthogonal, -1 = opposite)
            similarity = np.dot(M_values[i], M_values[j])
            distance = np.linalg.norm(M_values[i] - M_values[j])
            print(f"\n'{sent1}' vs '{sent2}':")
            print(f"  Cosine similarity: {similarity:.6f}")
            print(f"  Euclidean distance: {distance:.6f}")
