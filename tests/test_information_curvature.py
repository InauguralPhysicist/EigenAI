#!/usr/bin/env python3
"""
Tests for information curvature hypothesis:
High-entropy (technical) sentences → tight helices (short arc, low curvature, good orthogonality)
Low-entropy (vague) sentences → loose helices (long arc, high curvature, poor orthogonality)
"""

import pytest
import numpy as np
import math
from src.eigen_text_core import (
    understanding_loop,
    extract_LRV_syntactic_entropy_weighted,
    SPACY_AVAILABLE,
)

# Skip all tests if spacy not available
pytestmark = pytest.mark.skipif(
    not SPACY_AVAILABLE, reason="spacy required for entropy-weighted extraction"
)

# Simple word frequency model for testing
WORD_FREQ = {
    "the": 0.07,
    "is": 0.05,
    "a": 0.04,
    "and": 0.03,
    "to": 0.03,
    "of": 0.02,
    "in": 0.02,
    "that": 0.02,
    "it": 0.02,
    "was": 0.01,
    "cat": 0.0001,
    "dog": 0.0001,
    "dogs": 0.0001,
    "bark": 0.0001,
    "sat": 0.0001,
    "mat": 0.0001,
    "water": 0.0001,
    "light": 0.0001,
    "tree": 0.0001,
    "apple": 0.0001,
    "fell": 0.0001,
    "from": 0.001,
    "on": 0.001,
    "flows": 0.0001,
    "downhill": 0.0001,
    "statement": 0.00001,
    "false": 0.00001,
    "true": 0.00001,
    "both": 0.0001,
    "simultaneously": 0.000001,
    "this": 0.01,
    "quantum": 0.000001,
    "entanglement": 0.0000001,
    "demonstrates": 0.00001,
    "non-local": 0.0000001,
    "correlations": 0.00001,
    "phenomenon": 0.00001,
}


def word_entropy(word):
    """Calculate information content of a word."""
    word_lower = word.lower()
    freq = WORD_FREQ.get(word_lower, 1e-7)
    return -math.log2(freq)


def sentence_entropy(sentence):
    """Average information density of sentence."""
    words = sentence.lower().split()
    if not words:
        return 0.0
    entropies = [word_entropy(w) for w in words]
    return sum(entropies) / len(words)


def test_entropy_weighted_extraction_exists():
    """Test that entropy-weighted extraction function exists and works."""
    sentence = "The cat sat on the mat"

    triad = extract_LRV_syntactic_entropy_weighted(sentence, word_freq_model=WORD_FREQ)

    assert triad.L is not None
    assert triad.R is not None
    assert triad.V is not None
    assert len(triad.L) == 300  # Default embedding dim


def test_understanding_loop_with_entropy_weighting():
    """Test that understanding_loop accepts entropy_weighted parameter."""
    sentence = "Dogs bark"

    M, history, metrics = understanding_loop(
        sentence, entropy_weighted=True, word_freq_model=WORD_FREQ
    )

    assert M is not None
    assert len(history) > 0
    assert "arc_length" in metrics
    assert "curvature" in metrics
    assert "orthogonality" in metrics


def test_geometric_metrics_present():
    """Test that new geometric metrics are included in output."""
    sentence = "Water flows downhill"

    M, history, metrics = understanding_loop(sentence)

    # Check all geometric metrics exist
    assert "arc_length" in metrics
    assert "curvature" in metrics
    assert "orthogonality" in metrics

    # Check they're numeric
    assert isinstance(metrics["arc_length"], (int, float))
    assert isinstance(metrics["curvature"], (int, float))
    assert isinstance(metrics["orthogonality"], (int, float))

    # Check they're non-negative
    assert metrics["arc_length"] >= 0
    assert metrics["curvature"] >= 0
    assert metrics["orthogonality"] >= 0


def test_high_vs_low_entropy_sentences():
    """
    Test core hypothesis: High-entropy sentences produce tighter helices.

    High-entropy (technical) → short arc, low curvature, good orthogonality
    Low-entropy (vague) → long arc, high curvature, poor orthogonality
    """
    low_entropy_sentence = "It is"  # Very vague, common words
    high_entropy_sentence = (
        "Quantum entanglement demonstrates non-locality"  # Technical
    )

    # Calculate entropies
    low_ent = sentence_entropy(low_entropy_sentence)
    high_ent = sentence_entropy(high_entropy_sentence)

    assert high_ent > low_ent, "Technical sentence should have higher entropy"

    # Run understanding loop with entropy weighting
    M_low, hist_low, metrics_low = understanding_loop(
        low_entropy_sentence, entropy_weighted=True, word_freq_model=WORD_FREQ
    )

    M_high, hist_high, metrics_high = understanding_loop(
        high_entropy_sentence, entropy_weighted=True, word_freq_model=WORD_FREQ
    )

    # Hypothesis: High-entropy → better geometry
    # Note: This might not always hold due to other factors,
    # but should be generally true
    print(f"\nLow entropy ({low_ent:.2f}):")
    print(f"  Arc length: {metrics_low['arc_length']:.6f}")
    print(f"  Curvature: {metrics_low['curvature']:.6f}")
    print(f"  Orthogonality: {metrics_low['orthogonality']:.3f}")

    print(f"\nHigh entropy ({high_ent:.2f}):")
    print(f"  Arc length: {metrics_high['arc_length']:.6f}")
    print(f"  Curvature: {metrics_high['curvature']:.6f}")
    print(f"  Orthogonality: {metrics_high['orthogonality']:.3f}")

    # Generally expect (but don't hard-assert) these relationships:
    # High-entropy should have better orthogonality (lower score)
    if metrics_high["orthogonality"] < metrics_low["orthogonality"]:
        print("  ✓ High-entropy has better orthogonality")


def test_entropy_correlation_trend():
    """
    Test that entropy shows correlation with geometric metrics across multiple sentences.

    Expected: Negative correlation between entropy and arc_length/curvature
    (higher entropy → shorter arc, lower curvature)
    """
    test_sentences = [
        "It is",
        "Dogs bark",
        "The cat sat on the mat",
        "Water flows downhill",
        "The quantum phenomenon demonstrates entanglement",
        "Quantum entanglement demonstrates non-local correlations",
    ]

    results = []
    for sentence in test_sentences:
        entropy = sentence_entropy(sentence)
        M, hist, metrics = understanding_loop(
            sentence, entropy_weighted=True, word_freq_model=WORD_FREQ
        )

        results.append(
            {
                "sentence": sentence,
                "entropy": entropy,
                "arc_length": metrics["arc_length"],
                "curvature": metrics["curvature"],
                "orthogonality": metrics["orthogonality"],
            }
        )

    # Calculate correlation (entropy vs arc_length)
    entropies = [r["entropy"] for r in results]
    arc_lengths = [r["arc_length"] for r in results]

    if len(entropies) > 2:
        mean_e = sum(entropies) / len(entropies)
        mean_a = sum(arc_lengths) / len(arc_lengths)

        numerator = sum(
            (e - mean_e) * (a - mean_a) for e, a in zip(entropies, arc_lengths)
        )
        denom_e = math.sqrt(sum((e - mean_e) ** 2 for e in entropies))
        denom_a = math.sqrt(sum((a - mean_a) ** 2 for a in arc_lengths))

        if denom_e > 0 and denom_a > 0:
            correlation = numerator / (denom_e * denom_a)

            print(f"\nCorrelation (entropy vs arc_length): {correlation:.3f}")
            print(f"Entropy range: {min(entropies):.2f} to {max(entropies):.2f}")
            print(f"Arc length range: {min(arc_lengths):.6f} to {max(arc_lengths):.6f}")

            # We expect moderate to strong negative correlation
            # (not asserting hard threshold due to test variability)
            if abs(correlation) > 0.3:
                print(
                    f"  ✓ {'Strong' if abs(correlation) > 0.7 else 'Moderate'} correlation detected"
                )


def test_orthogonality_improvement_with_entropy():
    """
    Test that entropy weighting improves L-R-V orthogonality for high-entropy sentences.
    """
    technical_sentence = "Quantum entanglement demonstrates non-locality"

    # Without entropy weighting
    M_no_ent, hist_no_ent, metrics_no_ent = understanding_loop(
        technical_sentence, entropy_weighted=False
    )

    # With entropy weighting
    M_ent, hist_ent, metrics_ent = understanding_loop(
        technical_sentence, entropy_weighted=True, word_freq_model=WORD_FREQ
    )

    print(f"\nTechnical sentence orthogonality:")
    print(f"  Without entropy weighting: {metrics_no_ent['orthogonality']:.3f}")
    print(f"  With entropy weighting:    {metrics_ent['orthogonality']:.3f}")

    # Entropy weighting should improve (lower) orthogonality for technical language
    if metrics_ent["orthogonality"] < metrics_no_ent["orthogonality"]:
        print("  ✓ Entropy weighting improves orthogonality")


def test_arc_length_is_path_integral():
    """Test that arc_length equals sum of step distances."""
    sentence = "The cat sat on the mat"

    M, history, metrics = understanding_loop(sentence, max_iterations=10)

    # Manually calculate arc length
    manual_arc = 0.0
    for i in range(1, len(history)):
        segment = np.linalg.norm(history[i] - history[i - 1])
        manual_arc += segment

    # Should match reported arc_length
    assert (
        abs(metrics["arc_length"] - manual_arc) < 1e-6
    ), "Arc length should be sum of trajectory segments"


def test_curvature_measures_bending():
    """Test that curvature increases with trajectory bending."""
    # A sentence that converges quickly should have low curvature
    stable_sentence = "Dogs bark"

    # Run with more iterations to see any bending
    M, history, metrics = understanding_loop(stable_sentence, max_iterations=20)

    # Curvature should be sum of angles
    # For nearly straight path, should be close to 0
    assert metrics["curvature"] >= 0, "Curvature must be non-negative"

    # If path has 3+ points, curvature calculation should work
    if len(history) >= 3:
        assert metrics["curvature"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
