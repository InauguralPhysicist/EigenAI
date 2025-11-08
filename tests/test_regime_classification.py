#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test: Regime Classification (Space-like, Time-like, Light-like)

Tests whether the framework properly identifies and tracks different
understanding regimes based on ds² = S² - C²
"""

import numpy as np
import pytest
from src.eigen_discrete_tokenizer import (
    analyze_sentence,
    compute_change_stability,
    TOTAL_BITS
)


def classify_regime(ds2: int) -> str:
    """Classify regime based on ds²"""
    if ds2 > 0:
        return "time-like"
    elif ds2 < 0:
        return "space-like"
    else:
        return "light-like"


def test_regime_classification_exists():
    """Test that we can classify regimes from ds² values"""
    result = analyze_sentence("The cat sat", verbose=False)

    # We have ds2_history
    assert 'ds2_history' in result
    assert len(result['ds2_history']) > 0

    # We can classify each step
    regimes = [classify_regime(ds2) for ds2 in result['ds2_history']]

    print(f"\nRegime trajectory:")
    for i, (ds2, regime) in enumerate(zip(result['ds2_history'], regimes)):
        print(f"  Step {i}: ds²={ds2:4d} → {regime}")

    assert len(regimes) == len(result['ds2_history'])


def test_different_sentences_different_regimes():
    """Test if different types of sentences produce different regime patterns"""

    # Simple repetition (should stabilize quickly → time-like)
    repetition = "cat cat cat"

    # Varied meaningful (might have more transitions)
    meaningful = "The quick brown fox jumps"

    result_rep = analyze_sentence(repetition, verbose=False)
    result_mean = analyze_sentence(meaningful, verbose=False)

    regimes_rep = [classify_regime(ds2) for ds2 in result_rep['ds2_history']]
    regimes_mean = [classify_regime(ds2) for ds2 in result_mean['ds2_history']]

    print(f"\nRepetition regimes: {regimes_rep}")
    print(f"Meaningful regimes: {regimes_mean}")

    # Just document - don't assert specific behavior yet
    # This shows what the framework actually does


def test_regime_transitions():
    """Test tracking regime transitions (space-like → time-like)"""
    result = analyze_sentence("Water flows downhill naturally", verbose=False)

    regimes = [classify_regime(ds2) for ds2 in result['ds2_history']]

    # Find transitions
    transitions = []
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i-1]:
            transitions.append((i, regimes[i-1], regimes[i]))

    print(f"\nRegime trajectory: {' → '.join(regimes)}")
    print(f"Transitions: {len(transitions)}")
    for step, from_regime, to_regime in transitions:
        print(f"  Step {step}: {from_regime} → {to_regime}")

    # Document behavior


def test_time_like_means_stable():
    """
    Test hypothesis: Time-like (ds² > 0) means stable/understood

    Time-like: S > C (more stability than change)
    Should correlate with eigenstate detection
    """
    result = analyze_sentence("The cat sat on the mat", verbose=False)

    regimes = [classify_regime(ds2) for ds2 in result['ds2_history']]

    # Count final regime occurrences
    final_10 = regimes[-10:] if len(regimes) >= 10 else regimes
    time_like_count = sum(1 for r in final_10 if r == "time-like")

    print(f"\nFinal 10 steps: {final_10}")
    print(f"Time-like ratio: {time_like_count}/{len(final_10)}")

    # If eigenstate detected, expect more time-like in final steps
    if result['eigenstate']:
        print("✓ Eigenstate detected")
        print(f"  Hypothesis: Should see time-like dominance")
        print(f"  Observed: {time_like_count}/{len(final_10)} time-like")


def test_space_like_means_learning():
    """
    Test hypothesis: Space-like (ds² < 0) means learning/changing

    Space-like: C > S (more change than stability)
    Should appear during exploration
    """
    # Fresh input with new information
    result = analyze_sentence("Quantum entanglement defies locality", verbose=False)

    regimes = [classify_regime(ds2) for ds2 in result['ds2_history']]

    # Count early vs late
    early = regimes[:len(regimes)//2]
    late = regimes[len(regimes)//2:]

    space_early = sum(1 for r in early if r == "space-like")
    space_late = sum(1 for r in late if r == "space-like")

    print(f"\nEarly steps: {early}")
    print(f"Late steps: {late}")
    print(f"Space-like: early={space_early}/{len(early)}, late={space_late}/{len(late)}")

    # Hypothesis: More space-like during early learning
    # (Though current implementation may not show this)


def test_light_like_is_transitional():
    """
    Test hypothesis: Light-like (ds² ≈ 0) is transitional

    Light-like: C ≈ S (balanced change and stability)
    Should appear at phase transitions
    """
    result = analyze_sentence("Balance requires equilibrium always", verbose=False)

    regimes = [classify_regime(ds2) for ds2 in result['ds2_history']]
    ds2_values = result['ds2_history']

    # Find light-like moments (ds² = 0 or very small)
    light_like_indices = [i for i, ds2 in enumerate(ds2_values) if abs(ds2) < 10]

    print(f"\nRegimes: {regimes}")
    print(f"Light-like moments at indices: {light_like_indices}")

    if light_like_indices:
        print("✓ Found light-like transitions")
        for idx in light_like_indices:
            print(f"  Step {idx}: ds²={ds2_values[idx]}")


def test_regime_should_be_in_output():
    """
    MISSING FEATURE TEST: Regime classification should be in output

    Currently we have to compute it from ds2_history.
    Should be explicit in results.
    """
    result = analyze_sentence("Test sentence", verbose=False)

    # What we have:
    assert 'ds2_history' in result  # ✓

    # What we DON'T have but SHOULD:
    assert 'regime_history' not in result  # This should exist!

    # Manual computation works:
    regime_history = [classify_regime(ds2) for ds2 in result['ds2_history']]
    assert len(regime_history) > 0

    print("\n⚠ MISSING: 'regime_history' not in output")
    print("Should add to process_sentence_discrete() return value")
    print(f"Computed manually: {regime_history}")


def test_extreme_regimes():
    """Test extreme cases of each regime"""

    # Create states with known C, S values
    # Time-like: S >> C
    C_time = 1
    S_time = TOTAL_BITS - C_time  # Max S
    ds2_time = S_time**2 - C_time**2
    regime_time = classify_regime(ds2_time)

    # Space-like: C >> S
    C_space = TOTAL_BITS - 1  # Max C
    S_space = 1
    ds2_space = S_space**2 - C_space**2
    regime_space = classify_regime(ds2_space)

    # Light-like: C = S
    C_light = TOTAL_BITS // 2
    S_light = TOTAL_BITS // 2
    ds2_light = S_light**2 - C_light**2
    regime_light = classify_regime(ds2_light)

    print(f"\nExtreme regimes:")
    print(f"  Time-like: C={C_time}, S={S_time}, ds²={ds2_time:5d} → {regime_time}")
    print(f"  Space-like: C={C_space}, S={S_space}, ds²={ds2_space:5d} → {regime_space}")
    print(f"  Light-like: C={C_light}, S={S_light}, ds²={ds2_light:5d} → {regime_light}")

    assert regime_time == "time-like"
    assert regime_space == "space-like"
    assert regime_light == "light-like"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
