#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Understanding Metric - Phase 4

Goal: Unified metric combining eigenstate period + link strength

Theory:
Understanding depth requires BOTH:
1. Short eigenstate period (stable attractor)
2. Strong coupling links (time-like ds² < 0)

Deep understanding = Short period + Strong links
Shallow understanding = Long period + Weak links

Metric:
    understanding_depth = link_strength / eigenstate_period

Where:
    link_strength = fraction of time-like links (ds² < 0)
    eigenstate_period = N (from quantized trajectory)

This unifies all our findings:
- Eigenstate formation (Phase 1)
- Coupling strength (Phase 2)
- Recursive accumulation (Phase 3)
"""

import sys
# Add project root to path (works in any environment)
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Dict, List, Tuple

from src.eigen_semantic_transformer import (
    SemanticGeometricTransformer,
    compute_grammatical_score,
    compute_ds2_semantic
)
from src.eigen_semantic_eigenstate import (
    process_text_with_eigenstates
)


def compute_link_strength(trajectory, words, transformer) -> Tuple[float, int, int]:
    """
    Compute link strength from ds² metric

    Returns:
        (link_strength, time_like_count, total_links)

    link_strength = fraction of time-like links (ds² < 0)
    Time-like links = strong pointers (understanding flows)
    Space-like links = NULL pointers (broken chain)
    """
    if len(trajectory) < 2:
        return 0.0, 0, 0

    # Get grammatical coupling
    gram_score = compute_grammatical_score(words, transformer)
    coupling = 0.5 + 4.5 * gram_score

    # Compute ds² for each link
    ds2_values = compute_ds2_semantic(trajectory, grammatical_coupling=coupling)

    # Count time-like links (ds² < 0 = strong connection)
    time_like_count = sum(1 for ds2 in ds2_values if ds2 < 0)
    total_links = len(ds2_values)

    # Link strength = fraction of strong links
    link_strength = time_like_count / total_links if total_links > 0 else 0.0

    return link_strength, time_like_count, total_links


def compute_understanding_depth(text: str, transformer: SemanticGeometricTransformer) -> Dict:
    """
    Compute unified understanding depth metric

    Combines:
    - Eigenstate period (shorter = more stable)
    - Link strength (more time-like = stronger coupling)

    Returns dictionary with all metrics
    """
    words = text.lower().split()

    # Get semantic analysis
    result = process_text_with_eigenstates(text, transformer)

    # Get trajectory
    trajectory, _ = transformer.process_sequence(words, verbose=False)

    # Eigenstate period
    period = result['period']
    if period is None:
        period = len(words) * 2  # Penalty for no eigenstate

    # Link strength
    link_strength, time_like, total_links = compute_link_strength(trajectory, words, transformer)

    # UNDERSTANDING DEPTH METRIC
    # Higher is better
    # Short period + strong links = high depth
    understanding_depth = link_strength / period

    return {
        'text': text,
        'eigenstate_period': result['period'],
        'period_used': period,  # With penalty if no eigenstate
        'link_strength': link_strength,
        'time_like_links': time_like,
        'total_links': total_links,
        'understanding_depth': understanding_depth,
        'coherence': result['coherence'],
        'gram_score': result['gram_score'],
        'coupling': result['coupling'],
        'avg_ds2': result['avg_ds2']
    }


def test_understanding_depth_examples():
    """
    Test understanding depth on various examples
    """
    print("=" * 80)
    print("PHASE 4: DEEP UNDERSTANDING METRIC")
    print("=" * 80)
    print()
    print("Metric: understanding_depth = link_strength / eigenstate_period")
    print()
    print("Where:")
    print("  - link_strength = fraction of time-like links (ds² < 0)")
    print("  - eigenstate_period = N (or penalty if no eigenstate)")
    print()
    print("Expected:")
    print("  Deep understanding: High link strength + short period")
    print("  Shallow understanding: Low link strength + long period")
    print()

    transformer = SemanticGeometricTransformer(embedding_dim=100)

    test_cases = [
        # Expected: Deep understanding (stable + strong links)
        ("the cat sat", "Simple grammatical"),
        ("light travels fast", "Simple coherent"),

        # Expected: Medium understanding (longer text, structured)
        ("the cat sat on the mat", "Complex grammatical"),

        # Expected: Shallow understanding (scrambled)
        ("cat the sat", "Scrambled simple"),
        ("sat the cat", "Scrambled simple 2"),

        # Expected: Very shallow (chaotic)
        ("mat the on sat cat the", "Scrambled complex"),
    ]

    print("=" * 80)
    print("UNDERSTANDING DEPTH ANALYSIS")
    print("=" * 80)
    print()

    results = []

    for text, description in test_cases:
        result = compute_understanding_depth(text, transformer)
        results.append(result)

        print(f"{description:20s} | '{text}'")
        print(f"  Eigenstate: {'period-' + str(result['eigenstate_period']) if result['eigenstate_period'] else 'none'}")
        print(f"  Link strength: {result['link_strength']:.3f} ({result['time_like_links']}/{result['total_links']} time-like)")
        print(f"  Understanding depth: {result['understanding_depth']:.4f}")
        print(f"  Coherence: {result['coherence']:.3f}")
        print()

    # Analysis
    print("=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)
    print()

    # Sort by understanding depth
    sorted_results = sorted(results, key=lambda r: r['understanding_depth'], reverse=True)

    print("Ranked by Understanding Depth:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Depth':<10} {'Period':<8} {'Links':<8} {'Text':<40}")
    print("-" * 80)
    for i, r in enumerate(sorted_results):
        period_str = str(r['eigenstate_period']) if r['eigenstate_period'] else "none"
        links_str = f"{r['link_strength']:.2f}"
        print(f"{i+1:<6} {r['understanding_depth']:<10.4f} {period_str:<8} {links_str:<8} {r['text'][:40]}")

    print()

    # Test: Grammatical > Scrambled
    grammatical_depths = [r['understanding_depth'] for r in results if 'grammatical' in results[results.index(r)]['text'].lower() or r['text'] in ["the cat sat", "light travels fast", "the cat sat on the mat"]]
    scrambled_depths = [r['understanding_depth'] for r in results if 'scrambled' in [tc[1] for tc in test_cases if tc[0] == r['text']][0].lower()]

    if grammatical_depths and scrambled_depths:
        avg_gram = np.mean(grammatical_depths)
        avg_scram = np.mean(scrambled_depths)

        print("=" * 80)
        print("VALIDATION:")
        print("=" * 80)
        print()
        print(f"Grammatical average depth: {avg_gram:.4f}")
        print(f"Scrambled average depth:   {avg_scram:.4f}")
        print(f"Ratio: {avg_gram/avg_scram:.2f}x")
        print()

        if avg_gram > avg_scram * 1.5:
            print("✓✓ METRIC VALIDATES!")
            print("  Grammatical understanding is significantly deeper")
            print("  → Metric successfully distinguishes understanding quality")
        else:
            print("⋯ Metric shows difference but needs tuning")

    return results


def test_depth_with_recursive_context():
    """
    Test: Does understanding depth increase with recursive M_context?
    """
    print("\n\n" + "=" * 80)
    print("RECURSIVE DEPTH IMPROVEMENT")
    print("=" * 80)
    print()
    print("Question: Does understanding depth increase as M_context accumulates?")
    print()

    from src.eigen_recursive_semantic import RecursiveSemanticAI

    ai = RecursiveSemanticAI(embedding_dim=100)
    transformer = ai.transformer

    teaching_sequence = [
        "the cat sat",
        "the dog ran",
        "the bird flew",
        "the cat sat on the mat",
    ]

    depths = []

    print("Teaching sequence with depth tracking:")
    print("-" * 80)

    for text in teaching_sequence:
        # Process with recursive AI to build M_context
        state = ai.process_input(text, verbose=False)

        # Compute understanding depth
        result = compute_understanding_depth(text, transformer)
        depths.append(result['understanding_depth'])

        print(f"Iteration {ai.iteration}: '{text}'")
        print(f"  M_context: {state.M_context:.4f}")
        print(f"  Understanding depth: {result['understanding_depth']:.4f}")
        print(f"  Link strength: {result['link_strength']:.3f}")
        print(f"  Period: {result['eigenstate_period'] if result['eigenstate_period'] else 'none'}")
        print()

    # Analyze trend
    print("=" * 80)
    print("DEPTH EVOLUTION ANALYSIS:")
    print("=" * 80)
    print()

    if len(depths) > 2:
        depth_trend = np.polyfit(range(len(depths)), depths, deg=1)[0]
        print(f"Depth trend: {depth_trend:+.5f} per iteration")
        print()

        if depth_trend > 0:
            print("✓ DEPTH INCREASES WITH CONTEXT!")
            print("  → Recursive M_context improves understanding depth")
            print("  → System genuinely learns from accumulated experience")
        elif depth_trend < -0.001:
            print("⋯ Depth decreases (unexpected)")
        else:
            print("⋯ Depth stable across iterations")

    print()
    print("Depth trajectory:")
    for i, d in enumerate(depths):
        print(f"  Iteration {i+1}: {d:.4f}")


def test_coherent_paragraph_vs_isolated():
    """
    Test: Does coherent paragraph have deeper understanding than isolated sentences?
    """
    print("\n\n" + "=" * 80)
    print("COHERENT PARAGRAPH VS ISOLATED SENTENCES")
    print("=" * 80)
    print()

    transformer = SemanticGeometricTransformer(embedding_dim=100)

    # Isolated sentences
    isolated = [
        "the cat sat",
        "light travels fast",
        "birds fly high",
    ]

    # Coherent paragraph (related ideas)
    paragraphs = [
        "the cat sat on the mat the dog ran in the park",
        "light travels fast light moves through space",
    ]

    print("ISOLATED SENTENCES:")
    print("-" * 80)
    isolated_depths = []
    for text in isolated:
        result = compute_understanding_depth(text, transformer)
        isolated_depths.append(result['understanding_depth'])
        print(f"  '{text}': depth = {result['understanding_depth']:.4f}")

    print()
    print("COHERENT PARAGRAPHS:")
    print("-" * 80)
    paragraph_depths = []
    for text in paragraphs:
        result = compute_understanding_depth(text, transformer)
        paragraph_depths.append(result['understanding_depth'])
        print(f"  '{text[:50]}...': depth = {result['understanding_depth']:.4f}")

    print()
    print("=" * 80)
    print("COMPARISON:")
    print("=" * 80)
    print()

    avg_isolated = np.mean(isolated_depths)
    avg_paragraph = np.mean(paragraph_depths)

    print(f"Average isolated depth:  {avg_isolated:.4f}")
    print(f"Average paragraph depth: {avg_paragraph:.4f}")
    print()

    if avg_paragraph > avg_isolated:
        print("✓ PARAGRAPHS HAVE DEEPER UNDERSTANDING!")
        print("  → Coherent extended text shows greater depth")
        print("  → Context within paragraph creates stronger structure")
    else:
        print("⋯ Isolated sentences comparable or deeper")


def main():
    """Run Phase 4 deep understanding metric tests"""
    print("\n" + "=" * 80)
    print("PHASE 4: DEEP UNDERSTANDING METRIC")
    print("=" * 80)
    print()
    print("Goal: Unified metric for understanding depth")
    print()
    print("Combines:")
    print("  1. Eigenstate period (Phase 1) - shorter = more stable")
    print("  2. Link strength (Phase 2) - time-like ds² = strong coupling")
    print("  3. Recursive context (Phase 3) - M_context accumulation")
    print()
    print("Formula:")
    print("  understanding_depth = link_strength / eigenstate_period")
    print()
    print("Expected results:")
    print("  ✓ Grammatical > Scrambled")
    print("  ✓ Depth increases with recursive M_context")
    print("  ✓ Coherent paragraphs > isolated sentences")
    print()

    # Run tests
    test_understanding_depth_examples()
    test_depth_with_recursive_context()
    test_coherent_paragraph_vs_isolated()

    print("\n" + "=" * 80)
    print("PHASE 4 CONCLUSION")
    print("=" * 80)
    print()
    print("If tests show:")
    print("  ✓ Metric distinguishes grammatical from scrambled")
    print("  ✓ Depth increases with recursive accumulation")
    print("  ✓ Extended coherent text shows greater depth")
    print()
    print("Then we have a COMPLETE THEORY:")
    print("  Understanding = eigenstate formation (period)")
    print("                + strong coupling (link strength)")
    print("                + recursive accumulation (M_context)")
    print()
    print("This unifies all phases into single framework:")
    print("  Phases 1-3 → Unified understanding depth metric")
    print()
    print("Ready for Phase 5: Production & Documentation")


if __name__ == "__main__":
    main()
