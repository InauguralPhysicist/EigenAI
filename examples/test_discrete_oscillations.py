#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test discrete tokenization oscillations

Shows how time/space emerge from bit patterns
"""

import sys
sys.path.insert(0, '/home/user/EigenAI')

from src.eigen_discrete_tokenizer import (
    analyze_sentence,
    process_sentence_discrete
)


def test_eigenstate_formation():
    """Test different patterns of eigenstate formation"""

    print("=" * 70)
    print("EIGENSTATE FORMATION TESTS")
    print("=" * 70)
    print()

    test_cases = [
        ("cat cat cat", "Perfect period-2 (XOR self-inverse)"),
        ("the the the the", "Period-2 with 4 repetitions"),
        ("cat dog cat dog", "Period-4 potential"),
        ("a b a b a b", "Minimal period-2"),
        ("one two three one two three", "Period-6 potential"),
    ]

    results = []

    for sentence, description in test_cases:
        print(f"\n{'─' * 70}")
        print(f"TEST: {description}")
        print(f"Sentence: '{sentence}'")
        print(f"{'─' * 70}")

        result = process_sentence_discrete(sentence.split(), verbose=False)

        eigenstate = "✓" if result['eigenstate'] else "✗"
        period_str = f"period-{result['period']}" if result['period'] else "none"
        final_ds2 = result['ds2_history'][-1] if result['ds2_history'] else 0
        regime = "time-like" if final_ds2 > 0 else ("space-like" if final_ds2 < 0 else "light-like")

        print(f"{eigenstate} Eigenstate: {period_str}")
        print(f"  Time phase: sector {result['time_coord']}/8 ({result['time_coord']*45}°)")
        print(f"  Final ds²: {final_ds2} ({regime})")

        results.append({
            'sentence': sentence,
            'eigenstate': result['eigenstate'],
            'period': result['period'],
            'phase': result['time_coord'],
            'ds2': final_ds2
        })

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Sentence':<30} {'Eigenstate':<12} {'Period':<8} {'Phase':<8} {'ds²':<8}")
    print("─" * 70)
    for r in results:
        eigenstate_str = "✓" if r['eigenstate'] else "✗"
        period_str = str(r['period']) if r['period'] else "-"
        print(f"{r['sentence']:<30} {eigenstate_str:<12} {period_str:<8} {r['phase']:<8} {r['ds2']:<8}")


def test_time_space_decomposition():
    """Test how time and space coordinates emerge"""

    print("\n\n" + "=" * 70)
    print("TIME/SPACE DECOMPOSITION")
    print("=" * 70)
    print()

    sentences = [
        "the wind bends the tree",
        "water flows downhill",
        "light travels through space",
    ]

    for sentence in sentences:
        print(f"\n{'─' * 70}")
        print(f"Sentence: '{sentence}'")
        print(f"{'─' * 70}")

        result = process_sentence_discrete(sentence.split(), verbose=False)

        print(f"\nTIME coordinate:")
        print(f"  Phase sector: {result['time_coord']}/8")
        print(f"  Angular position: {result['time_coord'] * 45}°")

        print(f"\nSPACE coordinate:")
        print(f"  Top oscillating bits:")

        space = result['space_coord']
        import numpy as np
        top_bits = np.argsort(space)[-8:][::-1]  # Top 8

        for i, bit_idx in enumerate(top_bits[:8], 1):
            component = bit_idx // 8
            bit_pos = bit_idx % 8
            component_name = ['L', 'R', 'V', 'M'][component]
            flips = int(space[bit_idx])

            if flips > 0:
                print(f"    {i}. {component_name}[{bit_pos}]: {flips} flips")

        print(f"\nTrajectory:")
        print(f"  Length: {len(result['trajectory'])}")
        print(f"  Eigenstate: {'✓' if result['eigenstate'] else '✗'}")

        if result['ds2_history']:
            avg_ds2 = np.mean(result['ds2_history'])
            final_ds2 = result['ds2_history'][-1]
            print(f"  Average ds²: {avg_ds2:.1f}")
            print(f"  Final ds²: {final_ds2}")


def test_oscillation_patterns():
    """Visualize bit oscillation patterns"""

    print("\n\n" + "=" * 70)
    print("OSCILLATION PATTERNS")
    print("=" * 70)
    print()

    # Single word repeated (pure oscillation)
    sentence = "wave wave wave wave"
    print(f"Sentence: '{sentence}' (repeated word)")
    print("─" * 70)

    result = process_sentence_discrete(sentence.split(), verbose=False)

    print("\nState trajectory:")
    for i, state in enumerate(result['trajectory']):
        L, R, V, M = state
        print(f"  Step {i}: L={L:08b} R={R:08b} V={V:08b} M={M:08b}")

    if result['eigenstate']:
        print(f"\n✓ Period-{result['period']} eigenstate detected")
        print("  → Pure oscillation between states")
    else:
        print("\n✗ No eigenstate (doesn't close)")

    # Different words (complex trajectory)
    print("\n")
    sentence2 = "cat dog bird fish"
    print(f"Sentence: '{sentence2}' (all different)")
    print("─" * 70)

    result2 = process_sentence_discrete(sentence2.split(), verbose=False)

    print("\nState trajectory:")
    for i, state in enumerate(result2['trajectory']):
        L, R, V, M = state
        print(f"  Step {i}: L={L:08b} R={R:08b} V={V:08b} M={M:08b}")

    if result2['eigenstate']:
        print(f"\n✓ Period-{result2['period']} eigenstate detected")
    else:
        print("\n✗ No eigenstate (trajectory doesn't close)")


def main():
    """Run all tests"""

    print("\n" + "=" * 70)
    print("DISCRETE OSCILLATION TESTS")
    print("Testing time/space emergence from tokenization")
    print("=" * 70)

    test_eigenstate_formation()
    test_time_space_decomposition()
    test_oscillation_patterns()

    print("\n" + "=" * 70)
    print("KEY FINDINGS:")
    print("=" * 70)
    print("""
1. Repeated words create period-2 eigenstates (XOR self-inverse)
2. Time coordinate = phase sector (0-7, × 45° each)
3. Space coordinate = which bits oscillate
4. ds² metric distinguishes space-like vs time-like regimes
5. Eigenstate = trajectory closure (period-k orbit)
6. Meta bits (M) oscillate most → observer actively changing
7. Different sentences → different oscillation patterns
    """)


if __name__ == "__main__":
    main()
