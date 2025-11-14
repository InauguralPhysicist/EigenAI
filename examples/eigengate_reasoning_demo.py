#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eigengate Reasoning Demo - Deterministic Semantic Analysis

Demonstrates how Eigengate logic Q25 = (A ⊕ B) ∨ (D ⊙ C) enables deterministic
reasoning within EigenAI's semantic eigenstate framework.

Key Demonstrations:
1. Direct eigengate computation and balance detection
2. 5W1H (Who, What, When, Where, Why, How) extraction from semantic states
3. Light-like resolution of time-like/space-like oscillations
4. Integration with discrete token vocabulary
5. Semantic state analysis using Boolean logic
6. Truth table exploration

Usage:
    python examples/eigengate_reasoning_demo.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.eigen_gate_reasoning import (
    eigengate_Q25,
    semantic_to_eigengate,
    analyze_balance,
    resolve_oscillation,
    classify_regime_eigengate,
    print_truth_table,
    generate_truth_table
)
from src.eigen_discrete_tokenizer import tokenize_word, DiscreteToken


def demo_1_basic_eigengate():
    """Demo 1: Basic Eigengate Q25 Computation"""
    print("=" * 80)
    print("Demo 1: Basic Eigengate Q25 Computation")
    print("=" * 80)
    print()
    print("Formula: Q = (A ⊕ B) ∨ (D ⊙ C)")
    print("  Where ⊕ = XOR (asymmetry), ⊙ = XNOR (symmetry)")
    print()

    # Test cases from your examples
    test_cases = [
        (1, 0, 1, 0, "Asymmetry in both pairs"),
        (0, 1, 0, 1, "Asymmetry in both pairs (inverse)"),
        (1, 1, 1, 1, "Symmetry in both pairs"),
        (0, 0, 0, 0, "Symmetry in A-B, Asymmetry in D-C (imbalanced!)"),
    ]

    for A, B, D, C, description in test_cases:
        state = eigengate_Q25(A, B, D, C)
        print(f"Inputs: A={A}, B={B}, D={D}, C={C}")
        print(f"  Description: {description}")
        print(f"  XOR(A,B) = {state.xor_AB} (asymmetry: {state.has_AB_asymmetry()})")
        print(f"  XNOR(D,C) = {state.xnor_DC} (symmetry: {state.has_DC_symmetry()})")
        print(f"  Q = {state.Q} → {'BALANCED ✓' if state.is_balanced() else 'IMBALANCED ✗'}")
        print()


def demo_2_5w1h_analysis():
    """Demo 2: 5W1H Analysis from Eigengate States"""
    print("=" * 80)
    print("Demo 2: 5W1H (Who, What, When, Where, Why, How) Analysis")
    print("=" * 80)
    print()

    # Balanced state example
    print("Example 1: Balanced State (A=1, B=0, D=1, C=0)")
    print("-" * 80)
    state = eigengate_Q25(A=1, B=0, D=1, C=0)
    analysis = analyze_balance(state)

    for key in ['what', 'who', 'when', 'where', 'why', 'how']:
        print(f"{key.upper():>6}: {analysis[key]}")
    print()

    # Imbalanced state example
    print("Example 2: Imbalanced State (A=0, B=0, D=0, C=1)")
    print("-" * 80)
    state = eigengate_Q25(A=0, B=0, D=0, C=1)
    analysis = analyze_balance(state)

    for key in ['what', 'who', 'when', 'where', 'why', 'how']:
        print(f"{key.upper():>6}: {analysis[key]}")
    print()


def demo_3_semantic_state_resolution():
    """Demo 3: Semantic State Resolution via Eigengate"""
    print("=" * 80)
    print("Demo 3: Semantic State Resolution (L, R, V, M → Q)")
    print("=" * 80)
    print()
    print("Maps semantic states to Boolean inputs, then computes eigengate balance.")
    print("Threshold = 128: values ≥128 → 1, values <128 → 0")
    print()

    test_states = [
        (200, 50, 180, 200, "High L, low R, high V, high M"),
        (100, 100, 50, 50, "Medium L-R, low V-M"),
        (255, 255, 255, 255, "Maximum all components"),
        (50, 50, 50, 50, "Low all components"),
    ]

    for L, R, V, M, description in test_states:
        print(f"Semantic State: L={L}, R={R}, V={V}, M={M}")
        print(f"  Description: {description}")

        # Map to binary
        A, B, D, C = semantic_to_eigengate(L, R, V, M, threshold=128)
        print(f"  Binary: A={A}, B={B}, D={D}, C={C}")

        # Compute eigengate
        resolved, analysis = resolve_oscillation(L, R, V, M, threshold=128)
        print(f"  Resolved: {resolved}")
        print(f"  WHAT: {analysis['what']}")
        print(f"  WHY: {analysis['why']}")
        print()


def demo_4_token_integration():
    """Demo 4: Integration with DiscreteToken Vocabulary"""
    print("=" * 80)
    print("Demo 4: Eigengate Analysis of Discrete Tokens")
    print("=" * 80)
    print()
    print("Tokenize words and analyze their eigengate balance properties.")
    print()

    words = ["quantum", "physics", "understanding", "eigenstate", "balance"]

    for word in words:
        token = tokenize_word(word)
        print(f"Word: '{word}'")
        print(f"  Token: L={token.L:3d}, R={token.R:3d}, V={token.V:3d}, M={token.M:3d}")

        # Compute eigengate balance
        balanced, analysis = token.compute_eigengate_balance(threshold=128)
        regime = token.resolve_oscillation_eigengate(threshold=128)

        print(f"  Balanced: {balanced}")
        print(f"  Regime: {regime}")
        print(f"  WHAT: {analysis['what']}")
        print()


def demo_5_regime_classification():
    """Demo 5: Regime Classification via Eigengate"""
    print("=" * 80)
    print("Demo 5: Regime Classification (Light-like, Time-like, Space-like)")
    print("=" * 80)
    print()
    print("Eigengate interprets Q output as regime indicator:")
    print("  Q=1: Light-like (resolved, balanced)")
    print("  Q=0: Time-like/Space-like/Oscillating (context-dependent)")
    print()

    # Generate all 16 states
    table = generate_truth_table()

    # Group by regime
    regimes = {}
    for row in table:
        regime = row['regime']
        if regime not in regimes:
            regimes[regime] = []
        regimes[regime].append((row['A'], row['B'], row['D'], row['C'], row['Q']))

    for regime, states in sorted(regimes.items()):
        print(f"{regime.upper()} States: {len(states)}")
        for A, B, D, C, Q in states[:3]:  # Show first 3
            print(f"  A={A}, B={B}, D={D}, C={C} → Q={Q}")
        if len(states) > 3:
            print(f"  ... and {len(states)-3} more")
        print()


def demo_6_oscillation_resolution():
    """Demo 6: Light-like Resolution of Oscillations"""
    print("=" * 80)
    print("Demo 6: Light-like (Q) Resolves Time-like/Space-like Oscillations")
    print("=" * 80)
    print()
    print("When Q=1, oscillations resolve (light-like measurement).")
    print("When Q=0, oscillations continue (time-like/space-like conflict).")
    print()

    # Simulate oscillating states
    oscillating_states = [
        (100, 150, 100, 150, "Symmetric A-B (low), Asymmetric D-C → oscillates"),
        (200, 50, 180, 200, "Asymmetric A-B → resolves"),
        (180, 180, 50, 50, "Symmetric A-B (high), Symmetric D-C → resolves"),
    ]

    for L, R, V, M, description in oscillating_states:
        print(f"State: L={L}, R={R}, V={V}, M={M}")
        print(f"  Scenario: {description}")

        resolved, analysis = resolve_oscillation(L, R, V, M, threshold=128)

        if resolved:
            print(f"  ✓ RESOLVED: {analysis['when']}")
        else:
            print(f"  ✗ OSCILLATING: {analysis['when']}")

        print(f"  Mechanism: {analysis['how']}")
        print()


def demo_7_truth_table():
    """Demo 7: Complete Truth Table"""
    print("Demo 7: Complete Eigengate Q25 Truth Table")
    print()
    print_truth_table()


def interactive_mode():
    """Interactive Eigengate Explorer"""
    print("=" * 80)
    print("Interactive Eigengate Explorer")
    print("=" * 80)
    print()
    print("Enter semantic state values (L, R, V, M) to analyze eigengate balance.")
    print("Type 'quit' to exit.")
    print()

    while True:
        try:
            print("-" * 80)
            user_input = input("Enter L R V M (0-255 each), or 'quit': ").strip()

            if user_input.lower() in ['quit', 'q', 'exit']:
                print("Exiting interactive mode.")
                break

            values = [int(x) for x in user_input.split()]

            if len(values) != 4:
                print("Error: Please enter exactly 4 values (L R V M)")
                continue

            L, R, V, M = values

            if not all(0 <= v <= 255 for v in [L, R, V, M]):
                print("Error: All values must be in range 0-255")
                continue

            # Analyze
            print()
            print(f"Analyzing: L={L}, R={R}, V={V}, M={M}")
            print()

            A, B, D, C = semantic_to_eigengate(L, R, V, M, threshold=128)
            print(f"Binary mapping (threshold=128): A={A}, B={B}, D={D}, C={C}")
            print()

            state = eigengate_Q25(A, B, D, C)
            print(f"Eigengate Q25: {state.Q} → {'BALANCED' if state.is_balanced() else 'IMBALANCED'}")
            print(f"  XOR(A,B) = {state.xor_AB} (asymmetry in L-R)")
            print(f"  XNOR(D,C) = {state.xnor_DC} (symmetry in V-M)")
            print()

            analysis = analyze_balance(state)
            print("5W1H Analysis:")
            for key in ['what', 'who', 'when', 'where', 'why', 'how']:
                print(f"  {key.upper():>6}: {analysis[key]}")
            print()

            regime = classify_regime_eigengate(state)
            print(f"Regime: {regime}")
            print()

        except ValueError:
            print("Error: Invalid input. Please enter 4 integers (0-255).")
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print(" " * 20 + "EIGENGATE REASONING DEMONSTRATIONS")
    print(" " * 15 + "Deterministic Semantic Analysis via Boolean Logic")
    print("=" * 80 + "\n")

    demos = [
        ("Basic Eigengate Computation", demo_1_basic_eigengate),
        ("5W1H Analysis", demo_2_5w1h_analysis),
        ("Semantic State Resolution", demo_3_semantic_state_resolution),
        ("Token Integration", demo_4_token_integration),
        ("Regime Classification", demo_5_regime_classification),
        ("Oscillation Resolution", demo_6_oscillation_resolution),
        ("Truth Table", demo_7_truth_table),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        demo_func()
        if i < len(demos):
            input(f"\nPress Enter to continue to Demo {i+1}...")
            print("\n" * 2)

    # Offer interactive mode
    print("\n" * 2)
    response = input("Would you like to enter interactive mode? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        print()
        interactive_mode()

    print("\n" + "=" * 80)
    print(" " * 25 + "Demonstrations Complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
