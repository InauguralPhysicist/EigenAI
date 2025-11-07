#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive demonstration of Eigen text understanding

Run your own sentences through the semantic triad framework
"""

import sys
sys.path.insert(0, '/home/user/EigenAI')

from src.eigen_text_core import (
    understanding_loop,
    extract_LRV_from_sentence,
    compute_M_geometric,
    compute_M_xor
)


def analyze_sentence(text, method='geometric', verbose=True):
    """Analyze a single sentence"""
    print("=" * 70)
    print(f"TEXT: '{text}'")
    print("=" * 70)

    # Extract triad
    triad = extract_LRV_from_sentence(text)
    print(f"\nSemantic Triad:")
    print(f"  L (Lexical):    {triad.L.shape} vector")
    print(f"  R (Relational): {triad.R.shape} vector")
    print(f"  V (Value):      {triad.V.shape} vector")

    # Compute M
    if method == 'geometric':
        M = compute_M_geometric(triad.L, triad.R, triad.V)
        print(f"\nM (Meta) via geometric bisection: {M.shape}")
    else:
        M = compute_M_xor(triad.L, triad.R, triad.V)
        print(f"\nM (Meta) via XOR: {M.shape}")

    # Run understanding loop
    print(f"\nRunning understanding loop ({method})...\n")
    M_final, M_history, metrics = understanding_loop(
        text,
        max_iterations=20,
        method=method,
        verbose=verbose
    )

    # Results
    print("\n" + "─" * 70)
    print("RESULTS:")
    print("─" * 70)
    print(f"  Converged:        {metrics['converged']}")
    print(f"  Iterations:       {metrics['iterations']}")
    print(f"  Eigenstate type:  {metrics['eigenstate_type']}")

    if metrics['period']:
        print(f"  Period:           {metrics['period']}")

    print(f"  Final regime:     {metrics['final_regime']}")
    print(f"  Final alignment:  {metrics['final_alignment']:.4f}")

    if metrics['ds2_history']:
        print(f"  Final ds²:        {metrics['ds2_history'][-1]}")
        print(f"  Final C (change): {metrics['C_history'][-1]}")
        print(f"  Final S (stable): {metrics['S_history'][-1]}")

    return metrics


def compare_methods(text):
    """Compare geometric vs XOR methods"""
    print("\n" + "=" * 70)
    print("COMPARING METHODS")
    print("=" * 70)

    print("\n### GEOMETRIC METHOD ###\n")
    metrics_geo = analyze_sentence(text, method='geometric', verbose=False)

    print("\n### XOR METHOD ###\n")
    metrics_xor = analyze_sentence(text, method='xor', verbose=False)

    print("\n" + "─" * 70)
    print("COMPARISON:")
    print("─" * 70)
    print(f"{'Method':<12} {'Iters':<8} {'Converged':<12} {'Type':<15} {'Regime':<30}")
    print("─" * 70)
    print(f"{'Geometric':<12} {metrics_geo['iterations']:<8} {str(metrics_geo['converged']):<12} {metrics_geo['eigenstate_type']:<15} {metrics_geo['final_regime']:<30}")
    print(f"{'XOR':<12} {metrics_xor['iterations']:<8} {str(metrics_xor['converged']):<12} {metrics_xor['eigenstate_type']:<15} {metrics_xor['final_regime']:<30}")


def main():
    """Main interactive demo"""
    print("\n" + "=" * 70)
    print("EIGEN TEXT UNDERSTANDING - INTERACTIVE DEMO")
    print("=" * 70)
    print("\nThis demonstrates how text maps to (L,R,V,M) eigenspace")
    print("and converges to eigenstates.\n")

    # Example 1: Clear sentence
    analyze_sentence("The wind bends the tree", method='geometric')

    print("\n\n")

    # Example 2: Method comparison
    compare_methods("Water flows downhill")

    print("\n\n")

    # Example 3: Different sentence
    analyze_sentence("Light travels through space", method='xor')

    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS:")
    print("=" * 70)
    print("""
1. Simple sentences converge quickly (2-3 iterations)
2. Understanding reaches 'time-like' regime (settled)
3. ds² becomes large and positive (S >> C)
4. Fixed-point eigenstate is most common
5. Both geometric and XOR methods work

Try your own sentences:
  from src.eigen_text_core import understanding_loop
  M, history, metrics = understanding_loop("Your sentence here")
    """)


if __name__ == "__main__":
    main()
