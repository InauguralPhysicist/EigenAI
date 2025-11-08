#!/usr/bin/env python3
"""
Comparison test: Do final M values distinguish between
meaningful vs meaningless, grammatical vs ungrammatical?
"""

import numpy as np
from src.eigen_text_core import understanding_loop

print("=" * 80)
print("FALSIFICATION TEST: Can M distinguish different types of input?")
print("=" * 80)

# Test pairs from falsification tests
test_pairs = [
    ("Repetition vs Meaningful",
     "apple apple apple apple apple",
     "The apple fell from the tree"),

    ("Grammatical vs Ungrammatical",
     "The cat sat on the mat",
     "Cat the mat on sat the"),

    ("Coherent vs Contradiction",
     "Water flows downhill due to gravity",
     "This statement is both true and false simultaneously"),

    ("Meaningful vs Random Noise",
     "Light travels through space",
     "xkqz mplv jfgh wert yuio"),
]

print("\nFor each pair, comparing final M values:")
print("(Lower similarity = better distinction)\n")

for name, sent1, sent2 in test_pairs:
    M1, _, metrics1 = understanding_loop(sent1, max_iterations=5, verbose=False)
    M2, _, metrics2 = understanding_loop(sent2, max_iterations=5, verbose=False)

    similarity = np.dot(M1, M2)
    distance = np.linalg.norm(M1 - M2)

    print(f"\n{name}:")
    print(f"  Sent1: '{sent1}'")
    print(f"  Sent2: '{sent2}'")
    print(f"  Cosine similarity: {similarity:.6f}")
    print(f"  Euclidean distance: {distance:.6f}")

    if similarity > 0.95:
        result = "❌ TOO SIMILAR (can't distinguish)"
    elif similarity > 0.7:
        result = "⚠️  Somewhat similar"
    elif similarity > 0.3:
        result = "✓  Moderately different"
    else:
        result = "✓✓ VERY different (good distinction!)"

    print(f"  Result: {result}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("If similarities are < 0.95, the embeddings CAN distinguish!")
print("If similarities are ~0.99+, the embeddings CANNOT distinguish.")
