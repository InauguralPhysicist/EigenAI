#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Context Accumulation Layer

Tests the relative information impact implementation:
- Context density scaling
- Novelty detection
- Relative impact computation
- Phase transition detection
- Integration with EigenAI components
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.eigen_context_accumulator import ContextAccumulator, ContextEntry
from src.eigen_text_core import understanding_loop
from src.eigen_recursive_ai import RecursiveEigenAI


class TestContextEntry:
    """Test ContextEntry dataclass"""

    def test_create_context_entry(self):
        """Test creating a context entry"""
        vec = np.random.randn(100)
        entry = ContextEntry(
            vector=vec,
            timestamp=0,
            metadata={"text": "test"},
            impact=0.5
        )

        assert entry.timestamp == 0
        assert entry.impact == 0.5
        assert entry.metadata["text"] == "test"
        assert np.array_equal(entry.vector, vec)


class TestContextAccumulator:
    """Test ContextAccumulator class"""

    def test_initialization(self):
        """Test accumulator initialization"""
        acc = ContextAccumulator()

        assert acc.get_context_density() == 0
        assert acc.timestamp == 0
        assert len(acc.context_history) == 0

    def test_first_experience_high_impact(self):
        """Test that first experience has maximum impact"""
        acc = ContextAccumulator()
        vec = np.random.randn(100)
        vec = vec / np.linalg.norm(vec)

        impact = acc.compute_relative_impact(vec)

        # First experience should have maximum impact (1.0)
        assert impact == pytest.approx(1.0, rel=1e-3)

    def test_repeated_experience_low_impact(self):
        """Test that repeated experience has low impact"""
        acc = ContextAccumulator()
        vec = np.random.randn(100)
        vec = vec / np.linalg.norm(vec)

        # Add to context
        acc.add_context(vec, {"text": "first"})

        # Same vector again should have very low impact
        impact = acc.compute_relative_impact(vec)

        assert impact < 0.1  # Should be very low

    def test_context_density_reduces_impact(self):
        """Test that accumulated context reduces baseline impact"""
        acc = ContextAccumulator()

        # Create base vector
        base_vec = np.random.randn(100)
        base_vec = base_vec / np.linalg.norm(base_vec)

        # Add 100 similar contexts
        for i in range(100):
            similar_vec = base_vec + np.random.randn(100) * 0.1
            similar_vec = similar_vec / np.linalg.norm(similar_vec)
            acc.add_context(similar_vec, {"text": f"experience {i}"})

        # Now test impact of another similar vector
        test_vec = base_vec + np.random.randn(100) * 0.1
        test_vec = test_vec / np.linalg.norm(test_vec)

        impact = acc.compute_relative_impact(test_vec)

        # Impact should be low due to dense similar context
        assert impact < 0.2

    def test_novelty_detection(self):
        """Test novelty score computation"""
        acc = ContextAccumulator()

        # Add some contexts
        for i in range(10):
            vec = np.random.randn(100)
            vec = vec / np.linalg.norm(vec)
            acc.add_context(vec, {"text": f"context {i}"})

        # Test novel vector (orthogonal to all previous)
        novel_vec = np.random.randn(100)
        novel_vec = novel_vec / np.linalg.norm(novel_vec)

        novelty = acc.compute_novelty_score(novel_vec)

        # Should have high novelty (>0.5 typically)
        assert novelty > 0.3

        # Test familiar vector (same as first)
        familiar_vec = acc.context_history[0].vector

        novelty_familiar = acc.compute_novelty_score(familiar_vec)

        # Should have low novelty
        assert novelty_familiar < novelty

    def test_novel_despite_dense_context(self):
        """Test that truly novel input maintains high impact despite dense context"""
        acc = ContextAccumulator()

        # Create cluster of similar vectors
        base_vec = np.random.randn(100)
        base_vec = base_vec / np.linalg.norm(base_vec)

        for i in range(100):
            similar_vec = base_vec + np.random.randn(100) * 0.05
            similar_vec = similar_vec / np.linalg.norm(similar_vec)
            acc.add_context(similar_vec)

        # Novel vector orthogonal to cluster
        novel_vec = np.random.randn(100)
        novel_vec = novel_vec / np.linalg.norm(novel_vec)

        # Similar vector to cluster
        similar_vec = base_vec + np.random.randn(100) * 0.05
        similar_vec = similar_vec / np.linalg.norm(similar_vec)

        impact_novel = acc.compute_relative_impact(novel_vec)
        impact_similar = acc.compute_relative_impact(similar_vec)

        # Novel should have higher impact than similar
        assert impact_novel > impact_similar

    def test_find_similar_contexts(self):
        """Test finding similar historical contexts"""
        acc = ContextAccumulator()

        # Add contexts with known structure
        for i in range(10):
            vec = np.zeros(100)
            vec[i] = 1.0  # One-hot encoding
            acc.add_context(vec, {"text": f"vector {i}"})

        # Query with vector similar to #5
        query = np.zeros(100)
        query[5] = 1.0

        similar = acc.find_similar_contexts(query, top_k=3)

        # Should find vector #5 as most similar
        assert similar[0][0].metadata["text"] == "vector 5"
        assert similar[0][1] > 0.9  # High similarity

    def test_phase_transition_detection(self):
        """Test detecting paradigm shifts"""
        acc = ContextAccumulator()

        # Add dense familiar context
        base_vec = np.random.randn(100)
        base_vec = base_vec / np.linalg.norm(base_vec)

        for i in range(50):
            similar_vec = base_vec + np.random.randn(100) * 0.05
            similar_vec = similar_vec / np.linalg.norm(similar_vec)
            acc.add_context(similar_vec)

        # No phase transition yet
        is_trans, trans_type = acc.detect_phase_transition(window=10, impact_threshold=0.4)
        assert not is_trans

        # Add highly novel contexts (paradigm shift)
        for i in range(10):
            novel_vec = np.random.randn(100)
            novel_vec = novel_vec / np.linalg.norm(novel_vec)
            acc.add_context(novel_vec)

        # Should detect phase transition
        is_trans, trans_type = acc.detect_phase_transition(window=10, impact_threshold=0.1)
        assert is_trans
        assert trans_type in ["sustained_novelty", "punctuated_novelty"]

    def test_statistics(self):
        """Test statistics computation"""
        acc = ContextAccumulator()

        # Add some contexts
        for i in range(20):
            vec = np.random.randn(100)
            vec = vec / np.linalg.norm(vec)
            acc.add_context(vec, {"text": f"context {i}"})

        stats = acc.get_statistics()

        assert stats["context_density"] == 20
        assert stats["total_contexts_seen"] == 20
        assert "avg_impact" in stats
        assert "avg_novelty" in stats
        assert "impact_trend" in stats

    def test_max_history_size(self):
        """Test that max history size is enforced"""
        acc = ContextAccumulator(max_history_size=10)

        # Add 20 contexts
        for i in range(20):
            vec = np.random.randn(100)
            acc.add_context(vec)

        # Should only keep last 10
        assert acc.get_context_density() == 10
        assert acc.total_contexts_seen == 20

    def test_recency_weighting(self):
        """Test recency weighting for similarity"""
        acc = ContextAccumulator(use_recency_weighting=True, recency_decay=0.9)

        # Add old context
        old_vec = np.random.randn(100)
        old_vec = old_vec / np.linalg.norm(old_vec)
        acc.add_context(old_vec, {"text": "old"})

        # Add 50 unrelated contexts
        for i in range(50):
            vec = np.random.randn(100)
            acc.add_context(vec)

        # Add recent context identical to old
        acc.add_context(old_vec.copy(), {"text": "recent"})

        # Query with old_vec
        similar = acc.find_similar_contexts(old_vec, top_k=2, use_recency=True)

        # Recent should be ranked higher due to recency weighting
        assert similar[0][0].metadata["text"] == "recent"


class TestIntegrationWithEigenTextCore:
    """Test integration with eigen_text_core"""

    def test_understanding_loop_with_accumulator(self):
        """Test that understanding_loop works with context accumulator"""
        acc = ContextAccumulator()

        # First sentence - high impact
        M1, hist1, metrics1 = understanding_loop(
            "The wind bends the tree",
            max_iterations=20,
            context_accumulator=acc,
            verbose=False
        )

        assert "relative_impact" in metrics1
        assert "novelty_score" in metrics1
        assert metrics1["relative_impact"] == pytest.approx(1.0, rel=0.1)

        # Second similar sentence - lower impact
        M2, hist2, metrics2 = understanding_loop(
            "The breeze moves the branch",
            max_iterations=20,
            context_accumulator=acc,
            verbose=False
        )

        # Should have lower impact than first
        assert metrics2["relative_impact"] < metrics1["relative_impact"]

        # Context density should have increased
        assert acc.get_context_density() == 2

    def test_impact_modulates_learning_rate(self):
        """Test that relative impact modulates learning rate"""
        acc = ContextAccumulator()

        # First text - high impact, high learning rate
        M1, hist1, metrics1 = understanding_loop(
            "Quantum entanglement is nonlocal",
            max_iterations=20,
            learning_rate=0.1,
            context_accumulator=acc
        )

        # Add many similar contexts
        for i in range(50):
            understanding_loop(
                "Quantum physics is interesting",
                max_iterations=10,
                context_accumulator=acc,
                verbose=False
            )

        # Similar text - low impact, low learning rate
        M2, hist2, metrics2 = understanding_loop(
            "Quantum mechanics is fascinating",
            max_iterations=20,
            learning_rate=0.1,
            context_accumulator=acc
        )

        # Effective learning rate should be lower for familiar text
        assert metrics2["effective_learning_rate"] < metrics1["effective_learning_rate"]


class TestIntegrationWithRecursiveAI:
    """Test integration with RecursiveEigenAI"""

    def test_recursive_ai_with_accumulator(self):
        """Test RecursiveEigenAI with context accumulator"""
        acc = ContextAccumulator()
        ai = RecursiveEigenAI(embedding_dim=64, context_accumulator=acc)

        # Process first input
        result1 = ai.process("Cats are mammals", verbose=False)

        assert "relative_impact" in result1
        assert "novelty_score" in result1

        # First input should have high impact
        assert result1["relative_impact"] > 0.5

        # Process similar input
        result2 = ai.process("Dogs are mammals", verbose=False)

        # Should have lower impact than first
        assert result2["relative_impact"] < result1["relative_impact"]

    def test_self_modification_rate_modulation(self):
        """Test that self-modification rate is modulated by impact"""
        acc = ContextAccumulator()
        ai = RecursiveEigenAI(embedding_dim=64, context_accumulator=acc)

        # First input - high impact
        result1 = ai.process("Completely novel concept", verbose=False)
        rate1 = result1["extraction_rules"]["self_modification_rate"]

        # Add many similar inputs
        for i in range(20):
            ai.process("Similar familiar concept", verbose=False)

        # Familiar input - lower impact
        result2 = ai.process("Another familiar concept", verbose=False)
        rate2 = result2["extraction_rules"]["self_modification_rate"]

        # Self-modification rate should be lower for familiar input
        assert rate2 < rate1

    def test_state_summary_includes_context_stats(self):
        """Test that state summary includes context statistics"""
        acc = ContextAccumulator()
        ai = RecursiveEigenAI(embedding_dim=64, context_accumulator=acc)

        # Process some inputs
        ai.process("Input one", verbose=False)
        ai.process("Input two", verbose=False)

        summary = ai.get_state_summary()

        assert "context_stats" in summary
        assert "avg_impact" in summary
        assert "recent_impact" in summary
        assert summary["context_stats"]["context_density"] == 2


class TestRelativeImpactScaling:
    """Test the core insight: impact ∝ 1 / log(context_density)"""

    def test_impact_decreases_with_context_density(self):
        """Test that impact decreases as context accumulates"""
        acc = ContextAccumulator()

        base_vec = np.random.randn(100)
        base_vec = base_vec / np.linalg.norm(base_vec)

        impacts = []

        for i in range(100):
            # Similar vector
            vec = base_vec + np.random.randn(100) * 0.1
            vec = vec / np.linalg.norm(vec)

            impact = acc.compute_relative_impact(vec)
            impacts.append(impact)

            acc.add_context(vec)

        # Impacts should generally decrease
        early_avg = np.mean(impacts[:20])
        late_avg = np.mean(impacts[80:])

        assert late_avg < early_avg

    def test_impact_formula_correct(self):
        """Test that impact follows the formula: impact = novelty / log(density + 1)"""
        acc = ContextAccumulator()

        # Add some contexts
        for i in range(10):
            vec = np.random.randn(100)
            vec = vec / np.linalg.norm(vec)
            acc.add_context(vec)

        # Test vector
        test_vec = np.random.randn(100)
        test_vec = test_vec / np.linalg.norm(test_vec)

        # Compute manually
        novelty = acc.compute_novelty_score(test_vec)
        density = acc.get_context_density()
        expected_impact = novelty / (np.log(density + 1) + 1.0)

        # Compute via accumulator
        actual_impact = acc.compute_relative_impact(test_vec)

        # Should match
        assert actual_impact == pytest.approx(expected_impact, rel=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
