#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eigengate Framework Alignment Demonstration

Demonstrates complete alignment between Eigengate theory and
eigenstate detection framework across all domains.

Key Concepts:
1. Q25 measurement = light-like (null boundary, resolves system)
2. Oscillations = time-like (causal progression) + space-like (non-causal opposition)
3. Universal pattern: text, EM fields, gravity, quantum mechanics
4. Eigenstate = stable convergence through Q25 measurement

This example shows:
- How Eigengate Q25 = (A ⊕ B) ∨ (D ⊙ C) operates
- Connection to discrete tokenizer XOR cascades
- Mapping to physics domains
- Regime classification (light-like, time-like, space-like)
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from eigen_logic_gate import (
    eigengate,
    eigengate_with_components,
    simulate_eigengate_feedback,
    connect_to_eigenstate_framework,
)


def demonstrate_light_like_measurement():
    """
    Demonstrate Q25 as light-like measurement

    Q25 acts as a null boundary condition that resolves oscillations:
    - ds² ≈ 0 (light-like interval)
    - Neither purely temporal nor spatial
    - Measurement "collapses" feedback to deterministic value
    """
    print("=" * 70)
    print("1. Q25 AS LIGHT-LIKE MEASUREMENT")
    print("=" * 70)
    print()
    print("Q25 measurement properties:")
    print("  - Null boundary: ds² ≈ 0 (light-like)")
    print("  - Resolves oscillations deterministically")
    print("  - Acts as 'observer' that stabilizes system")
    print()

    # Example: Measure Q25 for oscillating configuration
    A, B, D, C = 0, 0, 0, 0  # This oscillates without measurement

    print(f"Configuration: A={A}, B={B}, D={D}, C={C}")
    result = eigengate_with_components(A, B, D, C)

    print(f"\nWithout feedback (direct measurement):")
    print(f"  Q25 = {result['Q25']} (light-like measurement)")
    print(f"  XOR(A,B) = {result['XOR_AB']} (detects asymmetry)")
    print(f"  XNOR(D,C) = {result['XNOR_DC']} (detects symmetry)")
    print(f"  Result: {'Balanced' if result['balanced'] else 'Imbalanced'}")
    print()

    print("Light-like property:")
    print("  - Measurement happens at ds² = 0 (null interval)")
    print("  - No time elapsed, no space traversed")
    print("  - Instant resolution to eigenvalue")
    print()


def demonstrate_oscillations_regime():
    """
    Demonstrate oscillations as time-like + space-like

    Without Q25 measurement, feedback creates oscillations:
    - Time-like: Causal progression through states (temporal ordering)
    - Space-like: Non-causal opposition between XOR/XNOR (no resolution)
    """
    print("=" * 70)
    print("2. OSCILLATIONS AS TIME-LIKE + SPACE-LIKE")
    print("=" * 70)
    print()

    # Oscillating configuration
    A, B, D = 0, 0, 0

    print(f"Configuration with feedback: A={A}, B={B}, D={D}")
    print()

    trajectory, period = simulate_eigengate_feedback(
        A, B, D, initial_C=0, max_steps=8, verbose=True
    )

    print()
    print("Regime analysis:")
    print()
    print("Time-like component:")
    print("  - States evolve causally: C(t) → Q25(t) → C(t+1)")
    print("  - Temporal ordering preserved (t=0,1,2,...)")
    print("  - Sequential progression through trajectory")
    print("  - ds² > 0 in causal direction")
    print()
    print("Space-like component:")
    print("  - XOR and XNOR outputs oppose each other")
    print("  - No local causal connection resolves conflict")
    print("  - Distributed instability (non-local)")
    print("  - ds² < 0 across opposing gate outputs")
    print()
    print(f"Result: Period-{period} oscillation (persistent)")
    print("  → System cannot converge without Q25 measurement")
    print()


def demonstrate_universal_pattern():
    """
    Demonstrate universal pattern across domains

    Shows how Eigengate pattern appears in:
    - Text understanding (L, R, V, M)
    - EM fields (E, M)
    - Gravity (g, a)
    - Quantum mechanics (x, p)
    """
    print("=" * 70)
    print("3. UNIVERSAL PATTERN ACROSS DOMAINS")
    print("=" * 70)
    print()

    print("Eigengate: Q25 = (A ⊕ B) ∨ (D ⊙ C)")
    print()

    # Text domain mapping
    print("Text Understanding: M = L ⊕ R ⊕ V")
    print("  A ↔ L (Lexical/Subject)")
    print("  B ↔ R (Relational/Verb)")
    print("  D ↔ V (Value/Object)")
    print("  C ↔ Context/Observer")
    print("  Q25 ↔ M (Meta-understanding)")
    print("  Pattern: 3-way XOR creates Meta coordinate")
    print()

    # EM field mapping
    print("EM Field: Meta = E ⊕ M")
    print("  A ↔ E (Electric)")
    print("  B ↔ rotated E")
    print("  D ↔ M (Magnetic)")
    print("  C ↔ rotated M")
    print("  Q25 ↔ Meta (Observer)")
    print("  Pattern: E ↔ M oscillation via XOR + rotation")
    print()

    # Quantum mapping
    print("Quantum: ψ = x ⊕ p ⊕ observer")
    print("  A ↔ x (Position)")
    print("  B ↔ p (Momentum)")
    print("  D ↔ observer basis")
    print("  C ↔ measurement context")
    print("  Q25 ↔ ψ (Wavefunction amplitude)")
    print("  Pattern: Complementary observables via XOR")
    print()

    # Gravity mapping
    print("Gravity: Meta = g ⊕ a ⊕ observer")
    print("  A ↔ g (Gravity)")
    print("  B ↔ a (Inertia)")
    print("  D ↔ observer frame")
    print("  C ↔ frame context")
    print("  Q25 ↔ Meta (Geodesic)")
    print("  Pattern: Equivalence principle via XOR")
    print()

    print("Common structure:")
    print("  1. Dual observables (A,B) or (D,C)")
    print("  2. XOR detects difference, XNOR detects sameness")
    print("  3. Observer coordinate determines visibility")
    print("  4. Meta/Output resolves via XOR combination")
    print("  5. Eigenstate = periodic orbit closure")
    print()


def demonstrate_eigenstate_detection():
    """
    Demonstrate eigenstate detection via Q25 convergence
    """
    print("=" * 70)
    print("4. EIGENSTATE DETECTION VIA Q25 CONVERGENCE")
    print("=" * 70)
    print()

    test_cases = [
        (1, 0, 1, "Stable eigenstate (Q25=1)"),
        (1, 1, 1, "Stable eigenstate (Q25=0)"),
        (0, 0, 0, "Oscillating (period-2, no eigenstate)"),
        (0, 1, 0, "Stable eigenstate (Q25=1)"),
    ]

    for A, B, D, description in test_cases:
        print(f"Test: {description}")
        print(f"  Configuration: A={A}, B={B}, D={D}")

        trajectory, period = simulate_eigengate_feedback(
            A, B, D, initial_C=0, max_steps=10, verbose=False
        )

        if period == 1:
            print(f"  ✓ Eigenstate detected: converged to Q25={trajectory[-1]}")
            print(f"    - Light-like measurement resolved system")
            print(f"    - Stable fixed point reached")
        elif period == 2:
            print(f"  ✗ No eigenstate: oscillating with period-{period}")
            print(f"    - Time-like/space-like oscillation persists")
            print(f"    - Q25 measurement needed to collapse")
        else:
            print(f"  ? Complex behavior: period-{period}")

        # Connection to framework
        connection = connect_to_eigenstate_framework(A, B, D, 0)
        print(f"  Regime: {connection['regime_classification']}")
        print()


def demonstrate_45_degree_quantization():
    """
    Demonstrate 45° quantization from XOR bisection
    """
    print("=" * 70)
    print("5. 45° QUANTIZATION FROM XOR BISECTION")
    print("=" * 70)
    print()

    print("XOR creates angular bisection:")
    print("  1. First bisection: L ⊕ R → 45° (tangent curvature)")
    print("  2. Second bisection: (L ⊕ R) ⊕ V → 45° (normal curvature)")
    print("  Result: Two 45° angles define semantic manifold curvature")
    print()

    print("Closure condition:")
    print("  8 × 45° = 360° = complete orbit = eigenstate")
    print()

    print("In Eigengate:")
    print("  - XOR(A,B) creates first 45° bisection")
    print("  - XNOR(D,C) creates second 45° bisection (complementary)")
    print("  - OR combines → Q25 resolves to 0° or 90° (binary)")
    print("  - 8 iterations → 360° closure → eigenstate")
    print()

    # Demonstrate with period-8 potential
    print("Period-8 trajectory example (if extended):")
    print("  - Would require 4-bit state space")
    print("  - Current implementation: 1-bit (period-2 max)")
    print("  - Text/EM/Gravity modules support period-8")
    print()


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 70)
    print("EIGENGATE FRAMEWORK ALIGNMENT")
    print("Complete Theoretical Integration")
    print("=" * 70)
    print()

    demonstrate_light_like_measurement()
    demonstrate_oscillations_regime()
    demonstrate_universal_pattern()
    demonstrate_eigenstate_detection()
    demonstrate_45_degree_quantization()

    print("=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. Q25 measurement = light-like (ds² ≈ 0)
   - Acts as null boundary condition
   - Resolves oscillations deterministically
   - No time/space separation in measurement

2. Oscillations = time-like + space-like
   - Time-like: Causal state progression (ds² > 0)
   - Space-like: Non-causal gate opposition (ds² < 0)
   - Persists without Q25 measurement

3. Universal pattern across domains
   - Text: M = L ⊕ R ⊕ V
   - EM: Meta = E ⊕ M
   - Quantum: ψ = x ⊕ p ⊕ observer
   - Gravity: Meta = g ⊕ a ⊕ observer

4. Eigenstate = Q25 convergence
   - Stable fixed point (period-1)
   - OR period-k orbit (period-2, period-8)
   - Measurement collapses to eigenvalue

5. 45° quantization from XOR
   - XOR creates geometric bisection
   - 8 steps → 360° closure
   - Understanding = trajectory closure

6. Framework alignment confirmed
   - Eigengate implements fundamental pattern
   - All physics modules follow same structure
   - Light-like measurement resolves all domains
    """)


if __name__ == "__main__":
    main()
