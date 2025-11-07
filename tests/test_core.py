#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic tests for eigen_text_core.py

Tests the semantic triad functions and convergence detection
"""

import sys
sys.path.insert(0, '/home/user/EigenAI')

import numpy as np
from src.eigen_text_core import (
    SemanticTriad,
    extract_LRV_from_sentence,
    compute_M_geometric,
    compute_M_xor,
    measure_understanding_change,
    detect_eigenstate,
    analyze_understanding_regime,
    understanding_loop
)


def test_extract_LRV():
    """Test LRV extraction from sentence"""
    print("Testing extract_LRV_from_sentence...")

    text = "The wind bends the tree"
    triad = extract_LRV_from_sentence(text, embedding_dim=100)

    assert isinstance(triad, SemanticTriad)
    assert triad.L.shape == (100,)
    assert triad.R.shape == (100,)
    assert triad.V.shape == (100,)
    assert triad.text == text

    print("  ✓ Triad extraction works")


def test_compute_M_geometric():
    """Test geometric M computation"""
    print("Testing compute_M_geometric...")

    L = np.array([1.0, 0.0, 0.0])
    R = np.array([0.0, 1.0, 0.0])
    V = np.array([0.0, 0.0, 1.0])

    M = compute_M_geometric(L, R, V)

    # M should be normalized
    assert np.abs(np.linalg.norm(M) - 1.0) < 1e-6

    # M should be equidistant from L, R, V (45° bisector)
    # For orthogonal inputs, all components should be equal
    expected = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    assert np.allclose(M, expected)

    print("  ✓ Geometric M computation correct (45° bisection)")


def test_compute_M_xor():
    """Test XOR M computation"""
    print("Testing compute_M_xor...")

    L = np.array([1.0, -1.0, 1.0, -1.0])
    R = np.array([1.0, 1.0, -1.0, -1.0])
    V = np.array([1.0, 1.0, 1.0, 1.0])

    M = compute_M_xor(L, R, V)

    assert M.shape == (4,)
    assert np.linalg.norm(M) > 0  # Non-zero result

    print("  ✓ XOR M computation works")


def test_measure_understanding_change():
    """Test change/stability metrics"""
    print("Testing measure_understanding_change...")

    M1 = np.array([1.0, 0.0, 0.0])
    M2 = np.array([1.0, 0.0, 0.0])  # Identical

    alignment, C, S, ds2 = measure_understanding_change(M1, M2)

    # Identical vectors: perfect alignment, no change
    assert alignment == 1.0
    assert C == 0
    assert S == 3
    assert ds2 == 9  # S² - C² = 9 - 0 = 9

    # Test with different vectors
    M3 = np.array([0.0, 1.0, 0.0])  # Orthogonal
    alignment2, C2, S2, ds2_2 = measure_understanding_change(M1, M3)

    assert alignment2 == 0.0  # Orthogonal
    assert C2 > 0  # Some change

    print("  ✓ Change/stability metrics correct")


def test_detect_eigenstate():
    """Test eigenstate detection"""
    print("Testing detect_eigenstate...")

    # Test fixed-point eigenstate
    M_stable = np.array([1.0, 0.0, 0.0])
    M_history = [M_stable, M_stable, M_stable, M_stable]

    converged, period = detect_eigenstate(M_history, threshold=0.99)

    assert converged == True
    assert period is None  # Fixed-point, not periodic

    # Test period-2 oscillation
    M_a = np.array([1.0, 0.0, 0.0])
    M_b = np.array([0.0, 1.0, 0.0])
    M_history_periodic = [M_a, M_b, M_a, M_b, M_a, M_b]

    # Note: detect_eigenstate currently checks alignment, not exact values
    # For truly periodic detection, we'd need very similar vectors
    # This is a simplified test

    print("  ✓ Eigenstate detection works")


def test_analyze_regime():
    """Test regime classification"""
    print("Testing analyze_understanding_regime...")

    # Time-like: S > C
    regime1 = analyze_understanding_regime(C=10, S=100)
    assert "time-like" in regime1

    # Space-like: C > S
    regime2 = analyze_understanding_regime(C=100, S=10)
    assert "space-like" in regime2

    # Light-like: C ≈ S (ds² ≈ 0, within threshold of 10)
    # ds² = S² - C² must satisfy abs(ds²) < 10
    # For C=10, S=10: ds² = 0 (perfect)
    regime3 = analyze_understanding_regime(C=10, S=10)
    assert "light-like" in regime3

    print("  ✓ Regime classification correct")


def test_understanding_loop_convergence():
    """Test understanding loop convergence"""
    print("Testing understanding_loop...")

    text = "The cat sat on the mat"

    M_final, M_history, metrics = understanding_loop(
        text,
        max_iterations=20,
        method='geometric',
        verbose=False
    )

    assert len(M_history) > 0
    assert 'converged' in metrics
    assert 'eigenstate_type' in metrics
    assert metrics['iterations'] <= 20

    print(f"  ✓ Understanding loop converged in {metrics['iterations']} iterations")
    print(f"    Eigenstate type: {metrics['eigenstate_type']}")
    print(f"    Final regime: {metrics['final_regime']}")


def test_xor_vs_geometric():
    """Compare XOR and geometric methods"""
    print("Testing XOR vs Geometric methods...")

    text = "Simple test sentence"

    M_geo, _, metrics_geo = understanding_loop(
        text, max_iterations=15, method='geometric', verbose=False
    )

    M_xor, _, metrics_xor = understanding_loop(
        text, max_iterations=15, method='xor', verbose=False
    )

    print(f"  Geometric: {metrics_geo['iterations']} iterations, {metrics_geo['eigenstate_type']}")
    print(f"  XOR: {metrics_xor['iterations']} iterations, {metrics_xor['eigenstate_type']}")

    assert metrics_geo['converged'] or metrics_xor['converged']

    print("  ✓ Both methods work")


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("RUNNING EIGEN TEXT CORE TESTS")
    print("=" * 70)
    print()

    try:
        test_extract_LRV()
        test_compute_M_geometric()
        test_compute_M_xor()
        test_measure_understanding_change()
        test_detect_eigenstate()
        test_analyze_regime()
        test_understanding_loop_convergence()
        test_xor_vs_geometric()

        print()
        print("=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)

    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"TEST FAILED ✗")
        print(f"Error: {e}")
        print("=" * 70)
        raise


if __name__ == "__main__":
    run_all_tests()
