#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Critical Test: Does Eigenstate Detection Correspond to Understanding?

The Challenge:
We detect eigenstates in token sequences. But does this correspond to
actual understanding/meaning, or just pattern convergence?

Test Cases:
1. Grammatical vs Ungrammatical
   - "The cat sat" (meaningful) vs "Cat the sat" (meaningless)
   - Should eigenstate formation differ?

2. Semantic Coherence vs Incoherence
   - "Light travels fast" (coherent) vs "Light eats blue" (incoherent)
   - Does eigenstate detect semantic sense?

3. Conceptual Consistency vs Contradiction
   - "Photons are light" (consistent) vs "Photons are heavy" (contradictory)
   - Can eigenstate detect contradiction?

4. True vs False Statements
   - "Water is wet" (true) vs "Water is dry" (false)
   - Does truth value affect eigenstate?

This is the REAL test of whether we're measuring understanding or just stability.
"""

import sys
# Add project root to path (works in any environment)
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.eigen_discrete_tokenizer import process_sentence_discrete
from src.eigen_recursive_ai import RecursiveEigenAI
from examples.measure_ai_understanding import UnderstandingMetrics


def test_grammatical_vs_ungrammatical():
    """
    Test 1: Does eigenstate detect grammatical structure?

    If eigenstate = understanding, grammatical sentences should show
    different behavior than ungrammatical scrambles.
    """
    print("=" * 80)
    print("TEST 1: GRAMMATICAL vs UNGRAMMATICAL")
    print("=" * 80)
    print()
    print("Question: Does eigenstate formation correlate with grammatical structure?")
    print()

    test_cases = [
        # Grammatical
        (["the", "cat", "sat"], "grammatical", "Simple valid sentence"),
        (["light", "travels", "fast"], "grammatical", "Valid sentence"),
        (["photons", "are", "particles"], "grammatical", "Valid statement"),

        # Ungrammatical (scrambled)
        (["cat", "the", "sat"], "ungrammatical", "Scrambled words"),
        (["travels", "light", "fast"], "ungrammatical", "Wrong order"),
        (["are", "photons", "particles"], "ungrammatical", "Scrambled"),

        # Nonsense (valid grammar, no meaning)
        (["the", "light", "sat"], "nonsense", "Grammatical but nonsensical"),
        (["photons", "travel", "blue"], "nonsense", "Grammar OK, meaning broken"),
    ]

    results = []

    for words, category, description in test_cases:
        text = " ".join(words)

        # Test discrete tokenization
        result = process_sentence_discrete(words, verbose=False)

        # Compute understanding metrics
        metrics = UnderstandingMetrics.compute_eigenstate_score(words)

        print(f"{category.upper():15s} | '{text}'")
        print(f"  Description: {description}")
        print(f"  Eigenstate: {'✓' if result['period'] else '✗'} ", end="")
        if result['period']:
            print(f"(period-{result['period']})")
        else:
            print()
        print(f"  Understanding score: {metrics['understanding_score']:.3f}")
        print(f"  Stability: {metrics['stability']:.3f}")
        print()

        results.append({
            'text': text,
            'category': category,
            'eigenstate': result['period'] is not None,
            'score': metrics['understanding_score'],
            'stability': metrics['stability']
        })

    # Analysis
    print("=" * 80)
    print("ANALYSIS:")
    print("=" * 80)
    print()

    grammatical = [r for r in results if r['category'] == 'grammatical']
    ungrammatical = [r for r in results if r['category'] == 'ungrammatical']
    nonsense = [r for r in results if r['category'] == 'nonsense']

    gram_avg_score = np.mean([r['score'] for r in grammatical])
    ungram_avg_score = np.mean([r['score'] for r in ungrammatical])
    nonsense_avg_score = np.mean([r['score'] for r in nonsense])

    print(f"Grammatical:    avg score = {gram_avg_score:.3f}")
    print(f"Ungrammatical:  avg score = {ungram_avg_score:.3f}")
    print(f"Nonsense:       avg score = {nonsense_avg_score:.3f}")
    print()

    # Check if there's meaningful difference
    if abs(gram_avg_score - ungram_avg_score) > 0.1:
        print("✓ SIGNIFICANT DIFFERENCE detected between grammatical and ungrammatical")
        print("  → Eigenstate MAY correlate with grammatical structure")
    else:
        print("✗ NO SIGNIFICANT DIFFERENCE between grammatical and ungrammatical")
        print("  → Eigenstate does NOT detect grammar (only token patterns)")

    print()
    return results


def test_semantic_coherence():
    """
    Test 2: Does eigenstate detect semantic coherence?

    Coherent: concepts that fit together
    Incoherent: concepts that don't fit
    """
    print("\n\n" + "=" * 80)
    print("TEST 2: SEMANTIC COHERENCE")
    print("=" * 80)
    print()
    print("Question: Does eigenstate distinguish meaningful vs nonsensical?")
    print()

    test_cases = [
        # Coherent
        (["photons", "emit", "light"], True, "Physically coherent"),
        (["water", "is", "wet"], True, "Semantically coherent"),
        (["birds", "can", "fly"], True, "Generally true"),

        # Incoherent
        (["photons", "eat", "chairs"], False, "Physically nonsensical"),
        (["water", "is", "dry"], False, "Contradictory"),
        (["numbers", "smell", "blue"], False, "Category error"),
    ]

    results = []

    for words, is_coherent, description in test_cases:
        text = " ".join(words)

        metrics = UnderstandingMetrics.compute_eigenstate_score(words)
        result = process_sentence_discrete(words, verbose=False)

        label = "COHERENT  " if is_coherent else "INCOHERENT"

        print(f"{label} | '{text}'")
        print(f"  {description}")
        print(f"  Eigenstate: {'✓' if result['period'] else '✗'}")
        print(f"  Understanding score: {metrics['understanding_score']:.3f}")
        print()

        results.append({
            'text': text,
            'coherent': is_coherent,
            'eigenstate': result['period'] is not None,
            'score': metrics['understanding_score']
        })

    # Analysis
    print("=" * 80)
    print("ANALYSIS:")
    print("=" * 80)
    print()

    coherent = [r for r in results if r['coherent']]
    incoherent = [r for r in results if not r['coherent']]

    coherent_avg = np.mean([r['score'] for r in coherent])
    incoherent_avg = np.mean([r['score'] for r in incoherent])

    print(f"Coherent:    avg score = {coherent_avg:.3f}")
    print(f"Incoherent:  avg score = {incoherent_avg:.3f}")
    print()

    if abs(coherent_avg - incoherent_avg) > 0.1:
        print("✓ SIGNIFICANT DIFFERENCE detected")
        print("  → Eigenstate MAY correlate with semantic coherence")
    else:
        print("✗ NO SIGNIFICANT DIFFERENCE")
        print("  → Eigenstate does NOT detect semantic meaning")

    print()
    return results


def test_with_actual_embeddings():
    """
    Test 3: Use actual word meanings (not just token identity)

    Problem with current tests: tokens are hashed independently.
    "cat" and "sat" have no semantic relationship encoded.

    Let's test if the RECURSIVE AI (which builds context) does better.
    """
    print("\n\n" + "=" * 80)
    print("TEST 3: RECURSIVE AI WITH CONTEXT")
    print("=" * 80)
    print()
    print("Question: Does recursive AI detect meaning through accumulated context?")
    print()

    # Create two AIs
    ai_meaningful = RecursiveEigenAI(embedding_dim=64)
    ai_nonsense = RecursiveEigenAI(embedding_dim=64)

    # Train on meaningful sequence
    print("Training AI #1 on MEANINGFUL sequence:")
    meaningful_sequence = [
        "light is electromagnetic waves",
        "photons are particles of light",
        "light travels very fast",
        "photons have no mass"
    ]

    for text in meaningful_sequence:
        result = ai_meaningful.process(text, verbose=False)
        print(f"  '{text}' → M_norm={np.linalg.norm(result['M_context_new']):.3f}")

    print()

    # Train on nonsense sequence
    print("Training AI #2 on NONSENSE sequence:")
    nonsense_sequence = [
        "light is electromagnetic chairs",
        "photons are particles of sadness",
        "light travels very purple",
        "photons have no Tuesday"
    ]

    for text in nonsense_sequence:
        result = ai_nonsense.process(text, verbose=False)
        print(f"  '{text}' → M_norm={np.linalg.norm(result['M_context_new']):.3f}")

    print()
    print("=" * 80)
    print("ANALYSIS:")
    print("=" * 80)
    print()

    # Check if meaningful AI converged to eigenstate
    state_meaningful = ai_meaningful.get_state_summary()
    state_nonsense = ai_nonsense.get_state_summary()

    print(f"Meaningful AI:")
    print(f"  Eigenstate reached: {state_meaningful['eigenstate_reached']}")
    print(f"  Iterations: {state_meaningful['iteration']}")

    print()
    print(f"Nonsense AI:")
    print(f"  Eigenstate reached: {state_nonsense['eigenstate_reached']}")
    print(f"  Iterations: {state_nonsense['iteration']}")

    print()

    if state_meaningful['eigenstate_reached'] and not state_nonsense['eigenstate_reached']:
        print("✓ Meaningful converged, nonsense did not")
        print("  → Recursive AI MAY detect semantic coherence")
    elif not state_meaningful['eigenstate_reached'] and not state_nonsense['eigenstate_reached']:
        print("⋯ Neither converged (need more iterations?)")
    else:
        print("✗ Both converged or wrong pattern")
        print("  → Recursive AI does NOT distinguish meaning from nonsense")


def test_the_core_question():
    """
    The ACTUAL test: "The cat sat" vs "Cat the sat"

    This is what the user asked for specifically.
    """
    print("\n\n" + "=" * 80)
    print("THE CORE QUESTION")
    print("=" * 80)
    print()
    print("User's challenge: Can your system distinguish these?")
    print()

    grammatical = ["the", "cat", "sat"]
    scrambled = ["cat", "the", "sat"]

    print("TEST A: 'The cat sat' (grammatical)")
    result_gram = process_sentence_discrete(grammatical, verbose=False)
    metrics_gram = UnderstandingMetrics.compute_eigenstate_score(grammatical)

    print(f"  Eigenstate: {'✓' if result_gram['period'] else '✗'}")
    print(f"  Period: {result_gram['period']}")
    print(f"  Understanding score: {metrics_gram['understanding_score']:.3f}")
    print(f"  Stability: {metrics_gram['stability']:.3f}")
    print()

    print("TEST B: 'Cat the sat' (ungrammatical)")
    result_scram = process_sentence_discrete(scrambled, verbose=False)
    metrics_scram = UnderstandingMetrics.compute_eigenstate_score(scrambled)

    print(f"  Eigenstate: {'✓' if result_scram['period'] else '✗'}")
    print(f"  Period: {result_scram['period']}")
    print(f"  Understanding score: {metrics_scram['understanding_score']:.3f}")
    print(f"  Stability: {metrics_scram['stability']:.3f}")
    print()

    print("=" * 80)
    print("VERDICT:")
    print("=" * 80)
    print()

    if metrics_gram['understanding_score'] > metrics_scram['understanding_score'] + 0.1:
        print("✓ YES - Grammatical sentence scores higher")
        print("  System CAN distinguish meaningful from scrambled")
        print()
        print("  This suggests eigenstate detection correlates with actual understanding")
    elif abs(metrics_gram['understanding_score'] - metrics_scram['understanding_score']) < 0.05:
        print("✗ NO - Scores are essentially identical")
        print("  System CANNOT distinguish meaningful from scrambled")
        print()
        print("  This means we're detecting PATTERN STABILITY, not UNDERSTANDING")
        print()
        print("  CRITICAL IMPLICATION:")
        print("  - Eigenstate = geometric convergence ✓")
        print("  - Eigenstate = semantic understanding ✗")
        print()
        print("  We have a pattern detector, not an understanding detector.")
    else:
        print("⋯ UNCLEAR - Small difference, inconclusive")
        print("  Need better tests or more sensitive metrics")

    print()
    print("=" * 80)
    print("HONEST ASSESSMENT:")
    print("=" * 80)
    print()
    print("Current framework detects:")
    print("  ✓ Token repetition (\"photon photon\" → eigenstate)")
    print("  ✓ Discrete geometric stability")
    print("  ✓ XOR cascade periodicity")
    print()
    print("Does NOT YET detect:")
    print("  ? Grammatical structure")
    print("  ? Semantic coherence")
    print("  ? Conceptual meaning")
    print()
    print("Next steps to validate 'understanding' claim:")
    print("  1. Encode semantic relationships between words")
    print("  2. Test on grammar vs non-grammar at scale")
    print("  3. Compare with human judgments of 'understanding'")
    print("  4. Show eigenstate correlates with comprehension tasks")


def main():
    """Run all critical validation tests"""
    print("\n" + "=" * 80)
    print("CRITICAL VALIDATION: EIGENSTATE = UNDERSTANDING?")
    print("=" * 80)
    print()
    print("Testing whether eigenstate detection measures actual understanding")
    print("or just pattern convergence.")
    print()

    # Run tests
    test_grammatical_vs_ungrammatical()
    test_semantic_coherence()
    test_with_actual_embeddings()
    test_the_core_question()

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("The challenge was: Does eigenstate = understanding?")
    print()
    print("If the tests show NO DIFFERENCE between:")
    print("  - Grammatical vs ungrammatical")
    print("  - Coherent vs incoherent")
    print("  - 'The cat sat' vs 'Cat the sat'")
    print()
    print("Then we must be HONEST:")
    print("  We're detecting GEOMETRIC STABILITY, not SEMANTIC UNDERSTANDING")
    print()
    print("This doesn't invalidate the framework - it clarifies what it ACTUALLY does.")
    print()
    print("To make the 'understanding' claim, we need to show eigenstate formation")
    print("correlates with semantic/grammatical structure, not just token patterns.")


if __name__ == "__main__":
    main()
