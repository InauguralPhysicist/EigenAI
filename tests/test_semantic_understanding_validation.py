#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Critical Validation: Does Semantic Transformer Solve the Understanding Problem?

Original Challenge (from test_understanding_vs_stability.py):
- Original discrete tokenizer: FAILED to distinguish "the cat sat" from "cat the sat"
- Both scored 0.223 (identical)
- Conclusion: Detecting pattern stability, NOT understanding

This test validates whether the semantic-geometric transformer with 3D cosine
coupling NOW successfully correlates eigenstate formation with actual understanding.

Test Strategy:
1. Run same test cases as original validation
2. Use semantic transformer instead of discrete tokenizer
3. Check if eigenstates NOW correlate with:
   - Grammatical structure
   - Semantic coherence
   - Truth/meaning
4. Validate metric signatures (ds²) distinguish understanding regimes
"""

import sys
# Add project root to path (works in any environment)
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.eigen_semantic_transformer import (
    SemanticGeometricTransformer,
    compute_grammatical_score,
    compute_ds2_semantic,
    classify_regime
)


def test_grammatical_vs_ungrammatical_semantic():
    """
    Test 1: Does SEMANTIC transformer detect grammatical structure?

    Original result: FAILED (both scored 0.223)
    Expected now: SUCCESS (grammatical should show better metrics)
    """
    print("=" * 80)
    print("TEST 1: GRAMMATICAL vs UNGRAMMATICAL (Semantic Transformer)")
    print("=" * 80)
    print()
    print("Question: Does semantic eigenstate formation correlate with grammar?")
    print()

    transformer = SemanticGeometricTransformer(embedding_dim=100)

    test_cases = [
        # Grammatical
        (["the", "cat", "sat"], "grammatical", "Simple valid sentence"),
        (["light", "travels", "fast"], "grammatical", "Valid sentence"),

        # Ungrammatical (scrambled)
        (["cat", "the", "sat"], "ungrammatical", "Scrambled words"),
        (["travels", "light", "fast"], "ungrammatical", "Wrong order"),

        # Nonsense (valid grammar, no meaning)
        (["the", "light", "sat"], "nonsense", "Grammatical but nonsensical"),
    ]

    results = []

    for words, category, description in test_cases:
        text = " ".join(words)

        # Process with semantic transformer
        trajectory, period = transformer.process_sequence(words, verbose=False)
        coherence = transformer.compute_semantic_coherence(words)

        # Get grammatical score and coupling
        gram_score = compute_grammatical_score(words, transformer)
        coupling = 0.5 + 4.5 * gram_score

        # Compute ds² metric
        ds2_values = compute_ds2_semantic(trajectory, grammatical_coupling=coupling)
        avg_ds2 = np.mean(ds2_values) if ds2_values else 0.0
        regime = classify_regime(avg_ds2)

        print(f"{category.upper():15s} | '{text}'")
        print(f"  Description: {description}")
        print(f"  Eigenstate: {'✓' if period else '✗'} ", end="")
        if period:
            print(f"(period-{period})")
        else:
            print()
        print(f"  Coherence: {coherence:.3f}")
        print(f"  Grammatical score: {gram_score:.3f}")
        print(f"  Coupling: {coupling:.2f}x")
        print(f"  ds²: {avg_ds2:.4f} [{regime}]")
        print()

        results.append({
            'text': text,
            'category': category,
            'eigenstate': period is not None,
            'period': period,
            'coherence': coherence,
            'gram_score': gram_score,
            'ds2': avg_ds2,
            'regime': regime
        })

    # Analysis
    print("=" * 80)
    print("ANALYSIS:")
    print("=" * 80)
    print()

    grammatical = [r for r in results if r['category'] == 'grammatical']
    ungrammatical = [r for r in results if r['category'] == 'ungrammatical']

    gram_avg_coherence = np.mean([r['coherence'] for r in grammatical])
    ungram_avg_coherence = np.mean([r['coherence'] for r in ungrammatical])

    gram_avg_ds2 = np.mean([r['ds2'] for r in grammatical])
    ungram_avg_ds2 = np.mean([r['ds2'] for r in ungrammatical])

    print(f"Grammatical:    avg coherence = {gram_avg_coherence:.3f}, avg ds² = {gram_avg_ds2:.4f}")
    print(f"Ungrammatical:  avg coherence = {ungram_avg_coherence:.3f}, avg ds² = {ungram_avg_ds2:.4f}")
    print()

    # Check if there's meaningful difference
    coherence_diff = abs(gram_avg_coherence - ungram_avg_coherence)
    ds2_diff = abs(gram_avg_ds2 - ungram_avg_ds2)

    if coherence_diff > 0.1:
        print(f"✓ COHERENCE: Significant difference detected ({coherence_diff:.3f})")
        print("  → Semantic transformer DOES detect grammatical structure")
    else:
        print(f"✗ COHERENCE: No significant difference ({coherence_diff:.3f})")
        print("  → Semantic transformer does NOT detect grammar")

    print()

    if ds2_diff > 0.5:
        print(f"✓ METRIC SIGNATURE: Significant ds² difference ({ds2_diff:.4f})")
        print("  → Coupling constant distinguishes grammatical from scrambled")
    else:
        print(f"✗ METRIC SIGNATURE: No significant ds² difference ({ds2_diff:.4f})")

    print()
    return results


def test_semantic_coherence_semantic():
    """
    Test 2: Does semantic transformer detect semantic coherence?

    Coherent: concepts that fit together
    Incoherent: concepts that don't fit
    """
    print("\n\n" + "=" * 80)
    print("TEST 2: SEMANTIC COHERENCE (Semantic Transformer)")
    print("=" * 80)
    print()
    print("Question: Does eigenstate distinguish meaningful vs nonsensical?")
    print()

    transformer = SemanticGeometricTransformer(embedding_dim=100)

    test_cases = [
        # Coherent
        (["light", "travels", "fast"], True, "Physically coherent"),

        # Incoherent
        (["light", "fast", "travels"], False, "Scrambled (incoherent order)"),
    ]

    results = []

    for words, is_coherent, description in test_cases:
        text = " ".join(words)

        trajectory, period = transformer.process_sequence(words, verbose=False)
        coherence = transformer.compute_semantic_coherence(words)

        gram_score = compute_grammatical_score(words, transformer)
        coupling = 0.5 + 4.5 * gram_score

        ds2_values = compute_ds2_semantic(trajectory, grammatical_coupling=coupling)
        avg_ds2 = np.mean(ds2_values) if ds2_values else 0.0
        regime = classify_regime(avg_ds2)

        label = "COHERENT  " if is_coherent else "INCOHERENT"

        print(f"{label} | '{text}'")
        print(f"  {description}")
        print(f"  Eigenstate: {'✓' if period else '✗'}")
        print(f"  Coherence: {coherence:.3f}")
        print(f"  ds²: {avg_ds2:.4f} [{regime}]")
        print()

        results.append({
            'text': text,
            'coherent': is_coherent,
            'eigenstate': period is not None,
            'coherence': coherence,
            'ds2': avg_ds2,
            'regime': regime
        })

    # Analysis
    print("=" * 80)
    print("ANALYSIS:")
    print("=" * 80)
    print()

    coherent = [r for r in results if r['coherent']]
    incoherent = [r for r in results if not r['coherent']]

    coherent_avg = np.mean([r['coherence'] for r in coherent]) if coherent else 0
    incoherent_avg = np.mean([r['coherence'] for r in incoherent]) if incoherent else 0

    print(f"Coherent:    avg coherence = {coherent_avg:.3f}")
    print(f"Incoherent:  avg coherence = {incoherent_avg:.3f}")
    print()

    if abs(coherent_avg - incoherent_avg) > 0.1:
        print("✓ SIGNIFICANT DIFFERENCE detected")
        print("  → Semantic transformer DOES correlate with semantic coherence")
    else:
        print("✗ NO SIGNIFICANT DIFFERENCE")
        print("  → Semantic transformer does NOT detect semantic meaning")

    print()
    return results


def test_the_core_question_semantic():
    """
    The ACTUAL test: "The cat sat" vs "Cat the sat"

    This is the original challenge that the discrete tokenizer FAILED.

    Original result:
      - "the cat sat": score 0.223
      - "cat the sat": score 0.223 (IDENTICAL - FAILED)

    Expected with semantic transformer: DIFFERENT scores/metrics
    """
    print("\n\n" + "=" * 80)
    print("THE CORE QUESTION: DOES SEMANTIC TRANSFORMER SOLVE IT?")
    print("=" * 80)
    print()
    print("Original challenge: Distinguish 'the cat sat' from 'cat the sat'")
    print("Original discrete tokenizer result: FAILED (both scored 0.223)")
    print()
    print("Testing with semantic-geometric transformer...")
    print()

    transformer = SemanticGeometricTransformer(embedding_dim=100)

    grammatical = ["the", "cat", "sat"]
    scrambled = ["cat", "the", "sat"]

    # Grammatical
    print("TEST A: 'The cat sat' (grammatical)")
    gram_traj, gram_period = transformer.process_sequence(grammatical, verbose=False)
    gram_coherence = transformer.compute_semantic_coherence(grammatical)
    gram_score = compute_grammatical_score(grammatical, transformer)
    gram_coupling = 0.5 + 4.5 * gram_score
    gram_ds2 = compute_ds2_semantic(gram_traj, grammatical_coupling=gram_coupling)
    gram_avg_ds2 = np.mean(gram_ds2) if gram_ds2 else 0.0
    gram_regime = classify_regime(gram_avg_ds2)

    print(f"  Eigenstate: {'✓' if gram_period else '✗'}", end="")
    if gram_period:
        print(f" (period-{gram_period})")
    else:
        print()
    print(f"  Coherence: {gram_coherence:.3f}")
    print(f"  Grammatical score: {gram_score:.3f}")
    print(f"  Coupling: {gram_coupling:.2f}x")
    print(f"  ds²: {gram_avg_ds2:.4f} [{gram_regime}]")
    print()

    # Scrambled
    print("TEST B: 'Cat the sat' (ungrammatical)")
    scram_traj, scram_period = transformer.process_sequence(scrambled, verbose=False)
    scram_coherence = transformer.compute_semantic_coherence(scrambled)
    scram_score = compute_grammatical_score(scrambled, transformer)
    scram_coupling = 0.5 + 4.5 * scram_score
    scram_ds2 = compute_ds2_semantic(scram_traj, grammatical_coupling=scram_coupling)
    scram_avg_ds2 = np.mean(scram_ds2) if scram_ds2 else 0.0
    scram_regime = classify_regime(scram_avg_ds2)

    print(f"  Eigenstate: {'✓' if scram_period else '✗'}", end="")
    if scram_period:
        print(f" (period-{scram_period})")
    else:
        print()
    print(f"  Coherence: {scram_coherence:.3f}")
    print(f"  Grammatical score: {scram_score:.3f}")
    print(f"  Coupling: {scram_coupling:.2f}x")
    print(f"  ds²: {scram_avg_ds2:.4f} [{scram_regime}]")
    print()

    print("=" * 80)
    print("VERDICT:")
    print("=" * 80)
    print()

    # Compare results
    coherence_diff = gram_coherence - scram_coherence
    ds2_sign_flip = (gram_avg_ds2 < 0 and scram_avg_ds2 > 0)

    print(f"Coherence difference: {coherence_diff:.3f}")
    print(f"ds² signature flip: {ds2_sign_flip}")
    print()

    if coherence_diff > 0.1 and ds2_sign_flip:
        print("✓✓ YES - SEMANTIC TRANSFORMER SOLVES THE PROBLEM!")
        print()
        print("Evidence:")
        print(f"  1. Coherence: Grammatical scores {coherence_diff:.3f} higher")
        print(f"  2. Metric signature FLIPS:")
        print(f"     - Grammatical: ds² = {gram_avg_ds2:.4f} ({gram_regime})")
        print(f"     - Scrambled:   ds² = {scram_avg_ds2:.4f} ({scram_regime})")
        print()
        print("Conclusion:")
        print("  ✓ Semantic transformer CAN distinguish meaningful from scrambled")
        print("  ✓ 3D cosine coupling creates proper metric signatures")
        print("  ✓ Grammatical structure amplifies coupling constant")
        print()
        print("This validates the claim:")
        print("  'Understanding = eigenstate detection' (with semantic encoding)")

    elif coherence_diff > 0.05:
        print("⋯ PARTIAL SUCCESS - Coherence differs but no clear metric signature")
        print()
        print(f"  Coherence difference: {coherence_diff:.3f} (detectable)")
        print(f"  ds² signature: {gram_regime} vs {scram_regime}")
        print()
        print("  System can detect some difference, but metric signature unclear")

    else:
        print("✗ NO - Still cannot distinguish")
        print()
        print(f"  Coherence difference: {coherence_diff:.3f} (too small)")
        print("  Semantic transformer does NOT solve the core problem")

    print()
    print("=" * 80)
    print("COMPARISON TO ORIGINAL:")
    print("=" * 80)
    print()
    print("Original discrete tokenizer:")
    print("  'the cat sat': score 0.223")
    print("  'cat the sat': score 0.223 (IDENTICAL)")
    print("  Result: FAILED to distinguish")
    print()
    print("Semantic transformer:")
    print(f"  'the cat sat': coherence {gram_coherence:.3f}, ds² {gram_avg_ds2:.4f}")
    print(f"  'cat the sat': coherence {scram_coherence:.3f}, ds² {scram_avg_ds2:.4f}")

    if coherence_diff > 0.1:
        print(f"  Result: SUCCESS - {coherence_diff:.3f} difference detected")
    else:
        print(f"  Result: FAILED - only {coherence_diff:.3f} difference")


def main():
    """Run all semantic validation tests"""
    print("\n" + "=" * 80)
    print("CRITICAL VALIDATION: SEMANTIC TRANSFORMER")
    print("=" * 80)
    print()
    print("Testing whether semantic-geometric transformer with 3D cosine coupling")
    print("NOW successfully correlates eigenstate formation with understanding.")
    print()
    print("This addresses the original challenge that discrete tokenizer FAILED:")
    print("  Can the system distinguish 'the cat sat' from 'cat the sat'?")
    print()

    # Run tests
    test_grammatical_vs_ungrammatical_semantic()
    test_semantic_coherence_semantic()
    test_the_core_question_semantic()

    print("\n" + "=" * 80)
    print("FINAL CONCLUSION")
    print("=" * 80)
    print()
    print("If semantic transformer shows SIGNIFICANT differences between:")
    print("  - Grammatical vs ungrammatical")
    print("  - Coherent vs incoherent")
    print("  - 'The cat sat' vs 'Cat the sat'")
    print()
    print("Then we have validated:")
    print("  ✓ Semantic encoding solves the understanding problem")
    print("  ✓ 3D cosine coupling creates proper metric signatures")
    print("  ✓ Eigenstate formation NOW correlates with understanding")
    print()
    print("This would prove the framework WORKS when properly implemented.")


if __name__ == "__main__":
    main()
