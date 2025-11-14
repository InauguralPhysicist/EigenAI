#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Eigengate logic circuit eigenstate detection

Verifies:
- Truth table correctness
- Feedback behavior (oscillation vs. convergence)
- Connection to eigenstate framework
- Regime classification (light-like, time-like, space-like)
"""

import sys
import os

# Add project root and src to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import directly from module file (avoids numpy dependency from __init__)
from eigen_logic_gate import (
    eigengate,
    eigengate_with_components,
    simulate_eigengate_feedback,
    XOR,
    XNOR,
    OR,
    verify_truth_table,
    connect_to_eigenstate_framework,
)


def test_basic_gates():
    """Test basic XOR, XNOR, OR gates"""
    # XOR tests
    assert XOR(0, 0) == 0, "XOR(0,0) should be 0"
    assert XOR(0, 1) == 1, "XOR(0,1) should be 1"
    assert XOR(1, 0) == 1, "XOR(1,0) should be 1"
    assert XOR(1, 1) == 0, "XOR(1,1) should be 0"

    # XNOR tests
    assert XNOR(0, 0) == 1, "XNOR(0,0) should be 1"
    assert XNOR(0, 1) == 0, "XNOR(0,1) should be 0"
    assert XNOR(1, 0) == 0, "XNOR(1,0) should be 0"
    assert XNOR(1, 1) == 1, "XNOR(1,1) should be 1"

    # OR tests
    assert OR(0, 0) == 0, "OR(0,0) should be 0"
    assert OR(0, 1) == 1, "OR(0,1) should be 1"
    assert OR(1, 0) == 1, "OR(1,0) should be 1"
    assert OR(1, 1) == 1, "OR(1,1) should be 1"

    print("✓ Basic gate tests passed")


def test_eigengate_truth_table():
    """Test Eigengate against complete truth table"""
    truth_table = {
        (0, 0, 0, 0): 1,
        (0, 0, 0, 1): 0,
        (0, 0, 1, 0): 0,
        (0, 0, 1, 1): 1,
        (0, 1, 0, 0): 1,
        (0, 1, 0, 1): 1,
        (0, 1, 1, 0): 1,
        (0, 1, 1, 1): 1,
        (1, 0, 0, 0): 1,
        (1, 0, 0, 1): 1,
        (1, 0, 1, 0): 1,
        (1, 0, 1, 1): 1,
        (1, 1, 0, 0): 1,
        (1, 1, 0, 1): 0,
        (1, 1, 1, 0): 0,
        (1, 1, 1, 1): 1,
    }

    errors = 0
    for (A, B, D, C), expected in truth_table.items():
        actual = eigengate(A, B, D, C)
        if actual != expected:
            print(f"✗ FAIL: A={A} B={B} D={D} C={C}: expected {expected}, got {actual}")
            errors += 1

    assert errors == 0, f"Truth table verification failed with {errors} errors"
    print("✓ Eigengate truth table tests passed (16/16 cases)")


def test_feedback_oscillation():
    """Test feedback behavior: oscillation case"""
    # A=0, B=0, D=0 should oscillate
    trajectory, period = simulate_eigengate_feedback(0, 0, 0, initial_C=0, max_steps=10, verbose=False)

    assert period == 2, f"Expected period-2 oscillation, got period {period}"
    assert trajectory == [1, 0, 1, 0, 1, 0, 1, 0, 1, 0], f"Unexpected trajectory: {trajectory}"

    print("✓ Feedback oscillation test passed (period-2)")


def test_feedback_stable_convergence():
    """Test feedback behavior: stable convergence"""
    # A=1, B=0, D=1, C=0 should converge to 1
    trajectory1, period1 = simulate_eigengate_feedback(1, 0, 1, initial_C=0, max_steps=10, verbose=False)

    assert period1 == 1, f"Expected stable convergence (period-1), got period {period1}"
    assert trajectory1[-1] == 1, f"Expected final Q25=1, got {trajectory1[-1]}"

    # A=1, B=1, D=1, C=0 should converge to 0
    trajectory2, period2 = simulate_eigengate_feedback(1, 1, 1, initial_C=0, max_steps=10, verbose=False)

    assert period2 == 1, f"Expected stable convergence (period-1), got period {period2}"
    assert trajectory2[-1] == 0, f"Expected final Q25=0, got {trajectory2[-1]}"

    print("✓ Feedback stable convergence tests passed")


def test_specification_examples():
    """Test specific examples from user specification"""
    # Example 1: A=1, B=0, D=1, starting C=0 → stable at 1
    traj1, period1 = simulate_eigengate_feedback(1, 0, 1, 0, max_steps=10, verbose=False)
    assert period1 == 1, "Example 1 should stabilize"
    assert traj1[-1] == 1, "Example 1 should converge to 1"

    # Example 2: A=0, B=0, D=0 → oscillates 1,0,1,0,...
    traj2, period2 = simulate_eigengate_feedback(0, 0, 0, 0, max_steps=10, verbose=False)
    assert period2 == 2, "Example 2 should oscillate with period-2"
    assert traj2 == [1, 0, 1, 0, 1, 0, 1, 0, 1, 0], "Example 2 should alternate 1,0"

    # Example 3: A=1, B=1, D=1 → stable at 0
    traj3, period3 = simulate_eigengate_feedback(1, 1, 1, 0, max_steps=10, verbose=False)
    assert period3 == 1, "Example 3 should stabilize"
    assert traj3[-1] == 0, "Example 3 should converge to 0"

    # Example 4: A=0, B=1, D=0 → stable at 1
    traj4, period4 = simulate_eigengate_feedback(0, 1, 0, 0, max_steps=10, verbose=False)
    assert period4 == 1, "Example 4 should stabilize"
    assert traj4[-1] == 1, "Example 4 should converge to 1"

    print("✓ Specification example tests passed (4/4 cases)")


def test_regime_classification():
    """Test regime classification (light-like, time-like, space-like)"""
    # Time-like: both XOR and XNOR true (A≠B AND D==C)
    result_time = eigengate_with_components(1, 0, 1, 1)
    assert result_time['XOR_AB'] == 1, "Should have XOR=1"
    assert result_time['XNOR_DC'] == 1, "Should have XNOR=1"
    assert result_time['Q25'] == 1, "Should output Q25=1"

    # Space-like: both XOR and XNOR false (A==B AND D≠C)
    result_space = eigengate_with_components(1, 1, 1, 0)
    assert result_space['XOR_AB'] == 0, "Should have XOR=0"
    assert result_space['XNOR_DC'] == 0, "Should have XNOR=0"
    assert result_space['Q25'] == 0, "Should output Q25=0"

    # Light-like: mixed (one true, one false)
    result_light1 = eigengate_with_components(1, 0, 1, 0)
    assert (result_light1['XOR_AB'] == 1 and result_light1['XNOR_DC'] == 0), "Should be mixed"
    assert result_light1['Q25'] == 1, "Should output Q25=1"

    result_light2 = eigengate_with_components(0, 0, 1, 1)
    assert (result_light2['XOR_AB'] == 0 and result_light2['XNOR_DC'] == 1), "Should be mixed"
    assert result_light2['Q25'] == 1, "Should output Q25=1"

    print("✓ Regime classification tests passed")


def test_eigenstate_framework_connection():
    """Test connection to (L,R,V,M) eigenstate framework"""
    # Test mapping
    connection = connect_to_eigenstate_framework(1, 0, 1, 0)

    assert connection['eigengate_output'] == 1, "Should output Q25=1"
    assert 'L' in connection['mapping_to_LRVM'], "Should map to L"
    assert 'R' in connection['mapping_to_LRVM'], "Should map to R"
    assert 'V' in connection['mapping_to_LRVM'], "Should map to V"
    assert 'M' in connection['mapping_to_LRVM'], "Should map to M"
    assert connection['eigenstate_indicator'] == True, "Should indicate eigenstate possible"

    print("✓ Eigenstate framework connection test passed")


def test_all_configurations_coverage():
    """Test all 8 fixed configurations (A,B,D) with feedback"""
    configs_tested = 0
    stable_count = 0
    oscillating_count = 0

    for A in [0, 1]:
        for B in [0, 1]:
            for D in [0, 1]:
                trajectory, period = simulate_eigengate_feedback(A, B, D, 0, max_steps=10, verbose=False)
                configs_tested += 1

                if period == 1:
                    stable_count += 1
                elif period == 2:
                    oscillating_count += 1

    assert configs_tested == 8, f"Should test 8 configurations, tested {configs_tested}"
    assert stable_count + oscillating_count == 8, "All configs should be stable or oscillating"

    print(f"✓ All configurations coverage: {configs_tested} tested ({stable_count} stable, {oscillating_count} oscillating)")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("EIGENGATE TESTS")
    print("=" * 70)
    print()

    test_basic_gates()
    test_eigengate_truth_table()
    test_feedback_oscillation()
    test_feedback_stable_convergence()
    test_specification_examples()
    test_regime_classification()
    test_eigenstate_framework_connection()
    test_all_configurations_coverage()

    print()
    print("=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
