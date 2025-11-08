#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Falsification Tests for EigenAI

These tests attempt to BREAK the theory that eigenstate = understanding.

Philosophy:
  "True intellect is knowing you may be wrong."
  "The truth doesn't move, only our position relative to it does."

These tests help us adjust our position relative to the truth by trying
to prove the theory wrong. If we can't break it, we gain confidence.
If we CAN break it, we learn something valuable.

Run with: pytest tests/test_falsification.py -v
"""

import numpy as np
import pytest
from src.eigen_text_core import (
    SemanticTriad,
    extract_LRV_from_sentence,
    compute_M_geometric,
    compute_M_xor,
    measure_understanding_change,
    detect_eigenstate,
    understanding_loop
)


# =============================================================================
# FALSIFICATION TESTS - Try to break the "eigenstate = understanding" claim
# =============================================================================

def test_random_noise_should_not_converge():
    """
    FALSIFICATION TEST: Random gibberish should NOT produce stable eigenstates

    Claim: Eigenstate detection indicates genuine understanding
    Test: Random noise should not produce eigenstates

    If random gibberish converges to eigenstates, it might mean:
    - Eigenstates are mathematical artifacts, not understanding indicators
    - Our convergence threshold is too loose
    - The theory needs refinement

    If it DOESN'T converge (as predicted), that supports the theory.
    """
    random_gibberish = "xkqz mplv jfgh wert yuio asdf qwer"

    M_final, M_history, metrics = understanding_loop(
        random_gibberish,
        max_iterations=100,
        method='geometric',
        verbose=False
    )

    # Record what happens for analysis
    # We don't assert failure/success - we document the behavior
    print(f"\nRandom noise test:")
    print(f"  Converged: {metrics['converged']}")
    print(f"  Iterations: {metrics['iterations']}")
    print(f"  Eigenstate type: {metrics.get('eigenstate_type', 'none')}")

    # The test passes either way - we're gathering data
    # But record your prediction: random noise should NOT converge
    # If it does, investigate why!


def test_contradictory_statements():
    """
    FALSIFICATION TEST: Logical contradictions should behave differently

    Prediction: Contradictory statements might not reach stable eigenstate
    or might show different convergence patterns than coherent statements.

    This tests if the framework can distinguish coherence from incoherence.
    """
    contradiction = "This statement is both true and false simultaneously"
    coherent = "Water flows downhill due to gravity"

    M_contra, hist_contra, metrics_contra = understanding_loop(
        contradiction, max_iterations=100, verbose=False
    )

    M_coherent, hist_coherent, metrics_coherent = understanding_loop(
        coherent, max_iterations=100, verbose=False
    )

    print(f"\nContradiction test:")
    print(f"  Contradiction converged: {metrics_contra['converged']} "
          f"in {metrics_contra['iterations']} iterations")
    print(f"  Coherent converged: {metrics_coherent['converged']} "
          f"in {metrics_coherent['iterations']} iterations")

    # Document behavior - does contradiction show different pattern?
    # This is exploratory - we're learning what the framework does


def test_memorization_vs_understanding():
    """
    FALSIFICATION TEST: Can we distinguish rote repetition from meaning?

    Pure repetition is memorization, not understanding.
    If both produce identical eigenstate patterns, the metric might not
    distinguish understanding from pattern matching.
    """
    # Pure repetition (memorization)
    repetition = "apple apple apple apple apple"

    # Meaningful relationship (understanding?)
    meaningful = "The apple fell from the tree"

    M_rep, hist_rep, metrics_rep = understanding_loop(
        repetition, max_iterations=50, verbose=False
    )

    M_mean, hist_mean, metrics_mean = understanding_loop(
        meaningful, max_iterations=50, verbose=False
    )

    print(f"\nMemorization vs Understanding:")
    print(f"  Repetition: converged={metrics_rep['converged']}, "
          f"iterations={metrics_rep['iterations']}, "
          f"type={metrics_rep.get('eigenstate_type', 'none')}")
    print(f"  Meaningful: converged={metrics_mean['converged']}, "
          f"iterations={metrics_mean['iterations']}, "
          f"type={metrics_mean.get('eigenstate_type', 'none')}")

    # What distinguishes them? This is an open research question
    # Document observed differences


def test_grammatical_vs_ungrammatical():
    """
    FALSIFICATION TEST: Does grammar affect eigenstate detection?

    If ungrammatical sentences show same eigenstate pattern as grammatical,
    the framework might be measuring something other than understanding.
    """
    grammatical = "The cat sat on the mat"
    ungrammatical = "Cat the mat on sat the"  # Same words, scrambled

    M_gram, hist_gram, metrics_gram = understanding_loop(
        grammatical, max_iterations=50, verbose=False
    )

    M_ungram, hist_ungram, metrics_ungram = understanding_loop(
        ungrammatical, max_iterations=50, verbose=False
    )

    print(f"\nGrammar test:")
    print(f"  Grammatical: {metrics_gram}")
    print(f"  Ungrammatical: {metrics_ungram}")

    # Both might converge, but do they show different patterns?


# =============================================================================
# MATHEMATICAL PROPERTY TESTS - These MUST hold if theory is correct
# =============================================================================

def test_M_geometric_always_normalized():
    """
    MATHEMATICAL INVARIANT: M must always be a unit vector

    If this fails, the geometric interpretation breaks down.
    This is a HARD requirement, not negotiable.
    """
    np.random.seed(42)  # Reproducible

    for i in range(100):
        # Random vectors of varying size
        dim = np.random.randint(10, 200)
        L = np.random.randn(dim)
        R = np.random.randn(dim)
        V = np.random.randn(dim)

        M = compute_M_geometric(L, R, V)
        norm = np.linalg.norm(M)

        assert abs(norm - 1.0) < 1e-10, \
            f"Test {i}: M not normalized! norm={norm}"


def test_eigenstate_implies_trajectory_stability():
    """
    LOGICAL INVARIANT: If eigenstate detected, trajectory must be stable

    If we claim an eigenstate exists but the trajectory is still changing,
    our detection is broken.
    """
    # Test on multiple sentences
    test_cases = [
        "Water flows downhill",
        "Light travels fast",
        "The earth orbits the sun"
    ]

    for text in test_cases:
        M_final, M_history, metrics = understanding_loop(
            text, max_iterations=100, verbose=False
        )

        if metrics['converged']:
            # If eigenstate claimed, last several vectors should be very similar
            last_10 = M_history[-10:]

            for i in range(len(last_10) - 1):
                alignment = np.dot(last_10[i], last_10[i+1])

                assert alignment > 0.90, \
                    f"Eigenstate claimed for '{text}' but trajectory unstable! " \
                    f"Alignment at step {i}: {alignment}"


def test_convergence_is_monotonic_improvement():
    """
    PROPERTY TEST: Understanding should improve or stabilize, not oscillate

    If alignment with previous state decreases then increases randomly,
    we're not measuring convergence to understanding.
    """
    text = "The cat sat on the mat"

    M_final, M_history, metrics = understanding_loop(
        text, max_iterations=50, verbose=False
    )

    # Check alignment between consecutive states
    alignments = []
    for i in range(1, len(M_history)):
        alignment = np.dot(M_history[i-1], M_history[i])
        alignments.append(alignment)

    # After some initial variation, should trend toward stability
    # Last 10 should all be high (>0.8) if converged
    if metrics['converged']:
        last_10_alignments = alignments[-10:]
        for i, align in enumerate(last_10_alignments):
            assert align > 0.8, \
                f"Converged but unstable: alignment[{i}] = {align}"


# =============================================================================
# ERROR HANDLING TESTS - Verify validation works
# =============================================================================

def test_extract_LRV_rejects_empty_string():
    """Input validation: Empty strings should be rejected"""
    with pytest.raises(ValueError, match="cannot be empty"):
        extract_LRV_from_sentence("", embedding_dim=100)

    with pytest.raises(ValueError, match="cannot be empty"):
        extract_LRV_from_sentence("   ", embedding_dim=100)


def test_extract_LRV_rejects_invalid_types():
    """Input validation: Wrong types should be rejected"""
    with pytest.raises(TypeError, match="must be a string"):
        extract_LRV_from_sentence(123, embedding_dim=100)

    with pytest.raises(TypeError, match="must be an integer"):
        extract_LRV_from_sentence("test", embedding_dim=10.5)


def test_extract_LRV_rejects_invalid_dimensions():
    """Input validation: Invalid dimensions should be rejected"""
    with pytest.raises(ValueError, match="must be positive"):
        extract_LRV_from_sentence("test", embedding_dim=-5)

    with pytest.raises(ValueError, match="must be positive"):
        extract_LRV_from_sentence("test", embedding_dim=0)


def test_compute_M_geometric_rejects_mismatched_shapes():
    """Input validation: Mismatched vector shapes should be rejected"""
    L = np.array([1.0, 0.0])
    R = np.array([0.0, 1.0, 0.0])  # Wrong size!
    V = np.array([0.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="same shape"):
        compute_M_geometric(L, R, V)


def test_compute_M_geometric_rejects_zero_vectors():
    """Input validation: All-zero vectors should be rejected"""
    with pytest.raises(ValueError, match="zero or nearly zero"):
        compute_M_geometric(
            np.zeros(3),
            np.zeros(3),
            np.zeros(3)
        )


def test_compute_M_geometric_rejects_nan():
    """Input validation: NaN values should be rejected"""
    L = np.array([np.nan, 0.0, 0.0])
    R = np.array([0.0, 1.0, 0.0])
    V = np.array([0.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="NaN"):
        compute_M_geometric(L, R, V)


def test_compute_M_geometric_rejects_inf():
    """Input validation: Infinite values should be rejected"""
    L = np.array([np.inf, 0.0, 0.0])
    R = np.array([0.0, 1.0, 0.0])
    V = np.array([0.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="Inf"):
        compute_M_geometric(L, R, V)


# =============================================================================
# EDGE CASE TESTS - Boundary conditions
# =============================================================================

def test_single_word_sentence():
    """Edge case: Single word should work without error"""
    triad = extract_LRV_from_sentence("Hello", embedding_dim=100)

    assert isinstance(triad, SemanticTriad)
    assert triad.L.shape == (100,)
    # Should complete without error


def test_very_long_sentence():
    """Edge case: Very long text should not crash"""
    long_text = " ".join(["word"] * 1000)

    M, hist, metrics = understanding_loop(
        long_text,
        max_iterations=50,
        verbose=False
    )

    assert len(M) > 0
    assert 'converged' in metrics


def test_special_characters():
    """Edge case: Special characters should be handled"""
    special = "Hello! How are you? I'm fine. #test @user"

    M, hist, metrics = understanding_loop(
        special,
        max_iterations=50,
        verbose=False
    )

    assert len(M) > 0


def test_unicode_text():
    """Edge case: Unicode should work"""
    unicode_text = "Hello 世界 мир العالم"

    triad = extract_LRV_from_sentence(unicode_text, embedding_dim=100)
    assert isinstance(triad, SemanticTriad)


# =============================================================================
# INTEGRATION TESTS - Full pipeline
# =============================================================================

def test_full_pipeline_runs_end_to_end():
    """Integration: Complete pipeline should work without errors"""
    text = "Water flows downhill due to gravity"

    # Step 1: Extract triad
    triad = extract_LRV_from_sentence(text, embedding_dim=128)

    # Step 2: Compute M
    M = compute_M_geometric(triad.L, triad.R, triad.V)
    assert abs(np.linalg.norm(M) - 1.0) < 1e-6

    # Step 3: Run understanding loop
    M_final, history, metrics = understanding_loop(text, verbose=False)

    # Step 4: Verify results
    assert len(history) > 0
    assert M_final is not None
    assert 'converged' in metrics
    assert 'iterations' in metrics


def test_discrete_vs_continuous_consistency():
    """
    Integration: Do discrete (XOR) and continuous (geometric) agree?

    If the two methods consistently disagree, one might be flawed,
    or they're measuring different things.
    """
    test_sentences = [
        "The cat sat on the mat",
        "Water flows downhill",
        "Light travels through space",
    ]

    agreement_count = 0

    for text in test_sentences:
        M_geo, _, metrics_geo = understanding_loop(
            text, method='geometric', max_iterations=50, verbose=False
        )

        M_xor, _, metrics_xor = understanding_loop(
            text, method='xor', max_iterations=50, verbose=False
        )

        # Do both agree on convergence?
        both_converged = metrics_geo['converged'] and metrics_xor['converged']
        neither_converged = (not metrics_geo['converged'] and
                           not metrics_xor['converged'])

        if both_converged or neither_converged:
            agreement_count += 1
        else:
            print(f"\nMethods disagree on: '{text}'")
            print(f"  Geometric: converged={metrics_geo['converged']}, "
                  f"iterations={metrics_geo['iterations']}")
            print(f"  XOR: converged={metrics_xor['converged']}, "
                  f"iterations={metrics_xor['iterations']}")

    # Document agreement rate
    agreement_rate = agreement_count / len(test_sentences)
    print(f"\nMethod agreement rate: {agreement_rate:.1%}")

    # We don't assert - just document
    # High disagreement would be interesting to investigate


if __name__ == "__main__":
    # Run with verbose output to see the exploratory results
    pytest.main([__file__, "-v", "-s"])
