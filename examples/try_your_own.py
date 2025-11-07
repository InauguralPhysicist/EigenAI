#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Try your own sentences through the Eigen text understanding system
"""

import sys
sys.path.insert(0, '/home/user/EigenAI')

from src.eigen_text_core import understanding_loop


# ===== PUT YOUR SENTENCES HERE =====

sentences_to_analyze = [
    "The cat sat on the mat",
    "Time flies like an arrow",
    "I think therefore I am",
    # Add your own sentences below:

]

# ====================================


def main():
    print("=" * 70)
    print("EIGEN TEXT UNDERSTANDING - YOUR SENTENCES")
    print("=" * 70)

    for i, text in enumerate(sentences_to_analyze, 1):
        print(f"\n[{i}/{len(sentences_to_analyze)}] '{text}'")
        print("-" * 70)

        M, history, metrics = understanding_loop(
            text,
            max_iterations=20,
            method='geometric',  # or 'xor'
            verbose=True
        )

        print(f"\n✓ {metrics['eigenstate_type']} eigenstate")
        print(f"  Iterations: {metrics['iterations']}")
        print(f"  Regime: {metrics['final_regime']}")

        if not metrics['converged']:
            print("  ⚠ Did not converge (ambiguous?)")

        print()


if __name__ == "__main__":
    main()
