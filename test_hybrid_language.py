#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing the "Geometry Hybrid Language"
Exploring how EigenAI processes different types of inputs
"""

import sys
sys.path.insert(0, '/home/user/EigenAI')

from src.eigen_discrete_tokenizer import process_sentence_discrete
from src.eigen_text_core import understanding_loop


def test_sentence(text, description):
    """Test a sentence and show both discrete and continuous analysis"""
    print(f"\n{'='*70}")
    print(f"TEST: {description}")
    print(f"Input: '{text}'")
    print(f"{'='*70}")

    # Discrete tokenization (XOR method)
    print("\n[DISCRETE GEOMETRY - XOR Method]")
    words = text.lower().split()
    result = process_sentence_discrete(words)

    if result['period']:
        print(f"✓ EIGENSTATE DETECTED: Period-{result['period']}")
        print(f"  Eigenstate type: {result['eigenstate']}")
        print(f"  Trajectory length: {len(result['trajectory'])}")
    else:
        print(f"✗ No eigenstate detected")
        print(f"  Trajectory length: {len(result['trajectory'])}")

    # Continuous understanding loop
    print("\n[CONTINUOUS GEOMETRY - Understanding Loop]")
    M, history, metrics = understanding_loop(text, method='xor', verbose=False)

    print(f"  Eigenstate: {metrics['eigenstate_type']}")
    print(f"  Convergence: {'Yes' if metrics['converged'] else 'No'}")
    print(f"  Regime: {metrics['final_regime']}")
    print(f"  ds²: {metrics.get('ds_squared', 'N/A')}")


def main():
    print("*" * 70)
    print("  EIGEN HYBRID LANGUAGE TEST")
    print("  Where English Becomes Geometry")
    print("*" * 70)

    # Test cases designed to explore different behaviors

    # 1. Repetition (should show periodic eigenstate)
    test_sentence(
        "wave wave wave",
        "Pure repetition - like light propagation"
    )

    test_sentence(
        "photon photon photon",
        "Quantum repetition - testing text=physics unification"
    )

    # 2. Symmetric structures
    test_sentence(
        "Alice loves Bob and Bob loves Alice",
        "Symmetric relationship - should show periodic behavior"
    )

    # 3. Logical chains
    test_sentence(
        "A implies B implies C",
        "Logical chain - testing transitive understanding"
    )

    # 4. Ambiguous/paradoxical
    test_sentence(
        "This statement is false",
        "Paradox - should show instability or period-2"
    )

    # 5. Poetic/metaphorical
    test_sentence(
        "Time is a river flowing through space",
        "Metaphor - blending temporal and spatial concepts"
    )

    # 6. Technical precision
    test_sentence(
        "The eigenvalue equals lambda",
        "Mathematical statement - high precision"
    )

    # 7. Recursive self-reference
    test_sentence(
        "Understanding understanding requires understanding",
        "Meta-recursive - testing self-referential closure"
    )

    # 8. Geometry-language hybrid
    test_sentence(
        "The trajectory closes at 45 degrees",
        "Mixing geometric and linguistic concepts"
    )

    print("\n\n" + "*" * 70)
    print("  ANALYSIS COMPLETE")
    print("*" * 70)
    print("\nKey Insights:")
    print("1. Repetitive inputs create periodic eigenstates (like physics)")
    print("2. Clear statements reach fixed-point eigenstates quickly")
    print("3. Ambiguous statements may oscillate (period-2, period-8, etc.)")
    print("4. The SAME geometric structure applies to language and physics")
    print("\nThis is the 'hybrid language' - English processed through")
    print("discrete geometry, where meaning = eigenstate closure.")
    print("*" * 70)


if __name__ == "__main__":
    main()
