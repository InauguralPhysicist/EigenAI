#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Eigengate Reasoning Layer

Tests deterministic semantic analysis using Boolean logic gates.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.eigen_gate_reasoning import (
    eigengate_Q25,
    semantic_to_eigengate,
    analyze_balance,
    resolve_oscillation,
    classify_regime_eigengate,
    generate_truth_table,
    EigengateState
)
from src.eigen_discrete_tokenizer import tokenize_word, DiscreteToken


class TestEigengateQ25:
    """Test core eigengate Q25 computation"""

    def test_eigengate_balanced_asymmetric_AB(self):
        """Test balanced state via A-B asymmetry"""
        state = eigengate_Q25(A=1, B=0, D=0, C=0)
        assert state.Q == 1
        assert state.is_balanced()
        assert state.has_AB_asymmetry()
        assert not state.has_DC_symmetry()

    def test_eigengate_balanced_symmetric_DC(self):
        """Test balanced state via D-C symmetry"""
        state = eigengate_Q25(A=0, B=0, D=1, C=1)
        assert state.Q == 1
        assert state.is_balanced()
        assert not state.has_AB_asymmetry()
        assert state.has_DC_symmetry()

    def test_eigengate_balanced_both_conditions(self):
        """Test balanced state when both conditions satisfied"""
        state = eigengate_Q25(A=1, B=0, D=1, C=1)
        assert state.Q == 1
        assert state.is_balanced()
        assert state.has_AB_asymmetry()
        assert state.has_DC_symmetry()

    def test_eigengate_imbalanced(self):
        """Test imbalanced state"""
        # Symmetric A-B, Asymmetric D-C → imbalanced
        state = eigengate_Q25(A=0, B=0, D=0, C=1)
        assert state.Q == 0
        assert not state.is_balanced()
        assert not state.has_AB_asymmetry()
        assert not state.has_DC_symmetry()

    def test_eigengate_all_ones(self):
        """Test all inputs = 1"""
        state = eigengate_Q25(A=1, B=1, D=1, C=1)
        assert state.Q == 1  # D⊙C = 1 (symmetric)
        assert state.is_balanced()

    def test_eigengate_all_zeros(self):
        """Test all inputs = 0"""
        state = eigengate_Q25(A=0, B=0, D=0, C=0)
        assert state.Q == 1  # D⊙C = 1 (symmetric)
        assert state.is_balanced()

    def test_xor_computation(self):
        """Test XOR intermediate value"""
        state = eigengate_Q25(A=1, B=0, D=0, C=0)
        assert state.xor_AB == 1  # 1 XOR 0 = 1

        state = eigengate_Q25(A=1, B=1, D=0, C=0)
        assert state.xor_AB == 0  # 1 XOR 1 = 0

    def test_xnor_computation(self):
        """Test XNOR intermediate value"""
        state = eigengate_Q25(A=0, B=0, D=1, C=1)
        assert state.xnor_DC == 1  # 1 XNOR 1 = 1

        state = eigengate_Q25(A=0, B=0, D=1, C=0)
        assert state.xnor_DC == 0  # 1 XNOR 0 = 0


class TestSemanticToEigengate:
    """Test semantic state to binary eigengate mapping"""

    def test_all_above_threshold(self):
        """Test values above threshold map to 1"""
        A, B, D, C = semantic_to_eigengate(200, 150, 180, 200, threshold=128)
        assert A == 1
        assert B == 1
        assert D == 1
        assert C == 1

    def test_all_below_threshold(self):
        """Test values below threshold map to 0"""
        A, B, D, C = semantic_to_eigengate(100, 50, 80, 100, threshold=128)
        assert A == 0
        assert B == 0
        assert D == 0
        assert C == 0

    def test_mixed_threshold(self):
        """Test mixed values across threshold"""
        A, B, D, C = semantic_to_eigengate(200, 50, 180, 100, threshold=128)
        assert A == 1  # 200 >= 128
        assert B == 0  # 50 < 128
        assert D == 1  # 180 >= 128
        assert C == 0  # 100 < 128

    def test_boundary_values(self):
        """Test threshold boundary"""
        # Exactly at threshold
        A, B, D, C = semantic_to_eigengate(128, 128, 128, 128, threshold=128)
        assert A == 1  # 128 >= 128
        assert B == 1
        assert D == 1
        assert C == 1

        # Just below threshold
        A, B, D, C = semantic_to_eigengate(127, 127, 127, 127, threshold=128)
        assert A == 0  # 127 < 128
        assert B == 0
        assert D == 0
        assert C == 0


class TestAnalyzeBalance:
    """Test 5W1H analysis generation"""

    def test_balanced_analysis_structure(self):
        """Test analysis returns correct structure"""
        state = eigengate_Q25(A=1, B=0, D=1, C=0)
        analysis = analyze_balance(state)

        assert 'what' in analysis
        assert 'who' in analysis
        assert 'when' in analysis
        assert 'where' in analysis
        assert 'why' in analysis
        assert 'how' in analysis

    def test_balanced_what_statement(self):
        """Test 'what' for balanced state"""
        state = eigengate_Q25(A=1, B=0, D=1, C=0)
        analysis = analyze_balance(state)
        assert 'balanced' in analysis['what'].lower()

    def test_imbalanced_what_statement(self):
        """Test 'what' for imbalanced state"""
        state = eigengate_Q25(A=0, B=0, D=0, C=1)
        analysis = analyze_balance(state)
        assert 'imbalanced' in analysis['what'].lower()

    def test_asymmetric_who_statement(self):
        """Test 'who' for asymmetric A-B"""
        state = eigengate_Q25(A=1, B=0, D=0, C=0)
        analysis = analyze_balance(state)
        assert 'asymmetric' in analysis['who'].lower()

    def test_how_contains_values(self):
        """Test 'how' contains gate values"""
        state = eigengate_Q25(A=1, B=0, D=1, C=0)
        analysis = analyze_balance(state)
        assert 'XOR' in analysis['how']
        assert 'XNOR' in analysis['how']
        assert 'Q=' in analysis['how']


class TestResolveOscillation:
    """Test oscillation resolution via eigengate"""

    def test_resolved_state(self):
        """Test resolved (balanced) state"""
        resolved, analysis = resolve_oscillation(L=200, R=50, V=180, M=200)
        assert resolved is True
        assert 'balanced' in analysis['what'].lower()

    def test_unresolved_state(self):
        """Test unresolved (imbalanced) state"""
        # Symmetric low L-R, Asymmetric low-high V-M
        resolved, analysis = resolve_oscillation(L=50, R=50, V=50, M=200)
        assert resolved is False
        assert 'imbalanced' in analysis['what'].lower()

    def test_analysis_structure(self):
        """Test analysis has 5W1H structure"""
        resolved, analysis = resolve_oscillation(L=200, R=50, V=180, M=200)
        assert isinstance(analysis, dict)
        assert all(key in analysis for key in ['what', 'who', 'when', 'where', 'why', 'how'])


class TestClassifyRegime:
    """Test regime classification"""

    def test_light_like_regime(self):
        """Test light-like classification for balanced state"""
        state = eigengate_Q25(A=1, B=0, D=1, C=0)
        regime = classify_regime_eigengate(state)
        assert regime == "light-like"

    def test_regime_for_all_balanced(self):
        """Test all balanced states classify as light-like"""
        balanced_inputs = [
            (1, 0, 0, 0),  # XOR true
            (0, 1, 0, 0),  # XOR true
            (0, 0, 1, 1),  # XNOR true
            (0, 0, 0, 0),  # XNOR true
            (1, 1, 1, 1),  # XNOR true
            (1, 0, 1, 1),  # Both true
        ]

        for A, B, D, C in balanced_inputs:
            state = eigengate_Q25(A, B, D, C)
            regime = classify_regime_eigengate(state)
            assert regime == "light-like", f"Failed for A={A}, B={B}, D={D}, C={C}"


class TestTruthTable:
    """Test truth table generation"""

    def test_truth_table_completeness(self):
        """Test truth table has all 16 combinations"""
        table = generate_truth_table()
        assert len(table) == 16

    def test_truth_table_structure(self):
        """Test each row has required fields"""
        table = generate_truth_table()
        required_fields = ['decimal', 'A', 'B', 'D', 'C', 'Q', 'xor_AB', 'xnor_DC',
                          'balanced', 'regime', 'what', 'who', 'when', 'where', 'why', 'how']

        for row in table:
            for field in required_fields:
                assert field in row

    def test_truth_table_known_values(self):
        """Test specific known truth table entries"""
        table = generate_truth_table()

        # Find entry for A=1, B=0, D=1, C=0 (decimal 10)
        row = next(r for r in table if r['decimal'] == 10)
        assert row['A'] == 1
        assert row['B'] == 0
        assert row['D'] == 1
        assert row['C'] == 0
        assert row['Q'] == 1  # Should be balanced
        assert row['balanced'] is True

        # Find entry for A=0, B=0, D=0, C=1 (decimal 1)
        row = next(r for r in table if r['decimal'] == 1)
        assert row['A'] == 0
        assert row['B'] == 0
        assert row['D'] == 0
        assert row['C'] == 1
        assert row['Q'] == 0  # Should be imbalanced
        assert row['balanced'] is False

    def test_truth_table_balance_count(self):
        """Test correct number of balanced states"""
        table = generate_truth_table()
        balanced_count = sum(1 for row in table if row['balanced'])

        # Q = (A ⊕ B) ∨ (D ⊙ C) should have 12/16 balanced states
        # Only A=B and D≠C gives Q=0 (4 combinations)
        assert balanced_count == 12


class TestDiscreteTokenIntegration:
    """Test integration with DiscreteToken class"""

    def test_token_has_eigengate_methods(self):
        """Test DiscreteToken has eigengate methods"""
        token = DiscreteToken(L=200, R=50, V=180, M=200, word="test")

        assert hasattr(token, 'compute_eigengate_balance')
        assert hasattr(token, 'generate_5w1h_analysis')
        assert hasattr(token, 'resolve_oscillation_eigengate')

    def test_token_eigengate_balance(self):
        """Test token eigengate balance computation"""
        token = DiscreteToken(L=200, R=50, V=180, M=200, word="quantum")
        balanced, analysis = token.compute_eigengate_balance()

        assert isinstance(balanced, bool)
        assert isinstance(analysis, dict)
        assert all(key in analysis for key in ['what', 'who', 'when', 'where', 'why', 'how'])

    def test_token_5w1h_analysis(self):
        """Test token 5W1H generation"""
        token = DiscreteToken(L=200, R=50, V=180, M=200, word="physics")
        analysis = token.generate_5w1h_analysis()

        assert isinstance(analysis, dict)
        assert 'what' in analysis
        assert 'why' in analysis

    def test_token_oscillation_resolution(self):
        """Test token oscillation resolution"""
        token = DiscreteToken(L=200, R=50, V=180, M=200, word="eigenstate")
        regime = token.resolve_oscillation_eigengate()

        assert isinstance(regime, str)
        assert regime in ['light-like', 'time-like', 'space-like', 'oscillating']

    def test_tokenized_word_eigengate(self):
        """Test eigengate analysis of tokenized word"""
        token = tokenize_word("understanding")
        balanced, analysis = token.compute_eigengate_balance()

        assert isinstance(balanced, bool)
        assert 'what' in analysis


class TestEigengateExamples:
    """Test examples from specification"""

    def test_example_1_1_0_1_0(self):
        """Test example: A=1, B=0, D=1, C=0"""
        state = eigengate_Q25(A=1, B=0, D=1, C=0)
        assert state.Q == 1
        assert state.is_balanced()

        analysis = analyze_balance(state)
        assert 'balanced' in analysis['what'].lower()
        assert 'asymmetric' in analysis['who'].lower()

    def test_example_0_1_0_1(self):
        """Test example: A=0, B=1, D=0, C=1"""
        state = eigengate_Q25(A=0, B=1, D=0, C=1)
        assert state.Q == 1
        assert state.is_balanced()

    def test_example_1_1_1_1(self):
        """Test example: A=1, B=1, D=1, C=1"""
        state = eigengate_Q25(A=1, B=1, D=1, C=1)
        assert state.Q == 1
        assert state.is_balanced()

        analysis = analyze_balance(state)
        assert 'symmetric' in analysis['who'].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
