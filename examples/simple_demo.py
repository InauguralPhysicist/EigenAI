#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple demonstration of Eigen text understanding

Shows how semantic triad (L, R, V) converges to eigenstate M
"""

import sys
sys.path.insert(0, '/home/user/EigenAI')

from src.eigen_text_core import understanding_loop

def main():
    print("=" * 70)
    print("EIGEN TEXT UNDERSTANDING - SIMPLE DEMO")
    print("=" * 70)

    # Test sentences
    sentences = [
        "The wind bends the tree",
        "Water flows downhill",
        "The cat sleeps",
    ]

    for text in sentences:
        print(f"\n{'='*70}")
        print(f"TEXT: '{text}'")
        print(f"{'='*70}\n")

        # Run understanding loop
        M_final, M_history, metrics = understanding_loop(
            text,
            max_iterations=15,
            method='geometric',
            verbose=True
        )

        print(f"\n{'─'*70}")
        print("RESULTS:")
        print(f"{'─'*70}")
        print(f"  Converged: {metrics['converged']}")
        print(f"  Iterations: {metrics['iterations']}")
        print(f"  Eigenstate: {metrics['eigenstate_type']}")
        if metrics['period']:
            print(f"  Period: {metrics['period']}")
        print(f"  Final regime: {metrics['final_regime']}")
        print(f"  Final alignment: {metrics['final_alignment']:.4f}")

        if metrics['ds2_history']:
            final_ds2 = metrics['ds2_history'][-1]
            print(f"  Final ds²: {final_ds2}")

if __name__ == "__main__":
    main()
