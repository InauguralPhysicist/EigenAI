#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eigen: Logic Gate Eigenstate Detection (Eigengate)

Implements the fundamental balance detection circuit:
    Q25 = (A ⊕ B) ∨ (D ⊙ C)

Where:
- A ⊕ B: XOR detects asymmetry (difference) between A and B
- D ⊙ C: XNOR detects symmetry (equivalence) between D and C
- Q25: OR combines to signal overall balance

This circuit serves as a "light-like" measurement that resolves
system oscillations by detecting the eigenvalue of balance.

Key Properties:
- Q25 = 1: System balanced (A≠B OR D==C OR both)
- Q25 = 0: System imbalanced (A==B AND D≠C)
- With feedback: Creates oscillations or converges to stable state
- Without feedback (measurement): Resolves to deterministic output

Connection to Physics:
- Light-like: Q25 acts as null measurement (boundary condition)
- Time-like: Sequential oscillations in feedback loop
- Space-like: Non-causal opposing gate outputs before resolution

This pattern underlies all eigenstate detection in the framework:
text, EM fields, gravity, quantum mechanics.
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class LogicState:
    """
    Logic circuit state (A, B, D, C, Q25)

    Attributes
    ----------
    A, B, D, C : int (0 or 1)
        Input bits
    Q25 : int (0 or 1)
        Output: balance indicator
    """
    A: int
    B: int
    D: int
    C: int
    Q25: int = 0

    def __repr__(self):
        return f"Logic(A={self.A} B={self.B} D={self.D} C={self.C} Q25={self.Q25})"


def XOR(a: int, b: int) -> int:
    """
    XOR gate: Detects asymmetry (difference)

    Returns 1 when inputs differ, 0 when same

    Truth table:
    A B | XOR
    ----+----
    0 0 |  0
    0 1 |  1
    1 0 |  1
    1 1 |  0
    """
    return a ^ b


def XNOR(d: int, c: int) -> int:
    """
    XNOR gate: Detects symmetry (equivalence)

    Returns 1 when inputs match, 0 when different
    XNOR = NOT(XOR)

    Truth table:
    D C | XNOR
    ----+-----
    0 0 |  1
    0 1 |  0
    1 0 |  0
    1 1 |  1
    """
    return 1 - (d ^ c)


def OR(x: int, y: int) -> int:
    """
    OR gate: Combines conditions

    Returns 1 if either input is 1

    Truth table:
    X Y | OR
    ----+---
    0 0 | 0
    0 1 | 1
    1 0 | 1
    1 1 | 1
    """
    return 1 if (x or y) else 0


def eigengate(A: int, B: int, D: int, C: int) -> int:
    """
    Eigengate circuit: Q25 = (A ⊕ B) ∨ (D ⊙ C)

    Detects system balance through pairwise symmetry checks:
    - XOR on (A,B): Detects dissimilarity (odd parity)
    - XNOR on (D,C): Detects similarity (even parity)
    - OR combines: System balanced if EITHER condition holds

    Parameters
    ----------
    A, B, D, C : int (0 or 1)
        Input bits

    Returns
    -------
    Q25 : int (0 or 1)
        1 = balanced, 0 = imbalanced

    Examples
    --------
    >>> eigengate(1, 0, 1, 0)  # A≠B (1), D≠C (0) → 1 OR 0 = 1
    1
    >>> eigengate(1, 1, 1, 0)  # A==B (0), D≠C (0) → 0 OR 0 = 0
    0
    >>> eigengate(0, 0, 1, 1)  # A==B (0), D==C (1) → 0 OR 1 = 1
    1

    Notes
    -----
    Canonical form using only NOT and AND:
    Q25 = (A∧¬B) ∨ (¬A∧B) ∨ (D∧C) ∨ (¬D∧¬C)
    """
    xor_AB = XOR(A, B)      # Detects asymmetry
    xnor_DC = XNOR(D, C)    # Detects symmetry
    Q25 = OR(xor_AB, xnor_DC)  # Combines balance conditions

    return Q25


def eigengate_with_components(A: int, B: int, D: int, C: int) -> Dict:
    """
    Eigengate with intermediate gate outputs

    Returns
    -------
    result : dict
        {
            'A': input A,
            'B': input B,
            'D': input D,
            'C': input C,
            'XOR_AB': A ⊕ B,
            'XNOR_DC': D ⊙ C,
            'Q25': final output,
            'balanced': bool
        }
    """
    xor_AB = XOR(A, B)
    xnor_DC = XNOR(D, C)
    Q25 = OR(xor_AB, xnor_DC)

    return {
        'A': A,
        'B': B,
        'D': D,
        'C': C,
        'XOR_AB': xor_AB,
        'XNOR_DC': xnor_DC,
        'Q25': Q25,
        'balanced': Q25 == 1
    }


def simulate_eigengate_feedback(
    A: int,
    B: int,
    D: int,
    initial_C: int = 0,
    max_steps: int = 10,
    verbose: bool = False
) -> Tuple[List[int], Optional[int]]:
    """
    Simulate Eigengate with feedback: Q25 → C

    This creates a feedback loop where the output is fed back to input C,
    allowing the system to oscillate or converge to a stable state.

    Parameters
    ----------
    A, B, D : int (0 or 1)
        Fixed inputs
    initial_C : int
        Initial value for C (default 0)
    max_steps : int
        Maximum simulation steps
    verbose : bool
        Print step-by-step evolution

    Returns
    -------
    trajectory : list of int
        Sequence of Q25 values
    period : int or None
        If oscillation detected, returns period
        If stable, returns 1
        If no pattern, returns None

    Notes
    -----
    Behavior modes:
    - Stable (converges): Q25 → constant
    - Oscillating: Q25 alternates in cycle
    - Chaotic: No clear pattern

    Connection to eigenstate theory:
    - Stable state = eigenstate (period-1)
    - Oscillation = periodic orbit (period-2, period-4, etc.)
    - Q25 measurement "collapses" feedback to stable value
    """
    C = initial_C
    trajectory = []

    if verbose:
        print(f"Eigengate Feedback Simulation")
        print(f"Fixed: A={A}, B={B}, D={D}")
        print(f"Initial C={C}")
        print(f"{'Step':<6} {'C':<6} {'XOR(A,B)':<10} {'XNOR(D,C)':<12} {'Q25':<6} {'Regime'}")
        print("─" * 60)

    for step in range(max_steps):
        # Compute Q25
        xor_AB = XOR(A, B)
        xnor_DC = XNOR(D, C)
        Q25 = OR(xor_AB, xnor_DC)

        trajectory.append(Q25)

        # Classify regime
        if xor_AB and xnor_DC:
            regime = "time-like"  # Both conditions true (settled)
        elif not xor_AB and not xnor_DC:
            regime = "space-like"  # Both false (unstable)
        else:
            regime = "light-like"  # Mixed (transition)

        if verbose:
            print(f"{step:<6} {C:<6} {xor_AB:<10} {xnor_DC:<12} {Q25:<6} {regime}")

        # Feedback: Q25 → C for next iteration
        C = Q25

    # Detect period
    period = detect_logic_cycle(trajectory)

    if verbose:
        print()
        if period == 1:
            print(f"✓ Converged to stable state: Q25={trajectory[-1]}")
        elif period:
            print(f"✓ Oscillating with period-{period}")
            print(f"  Cycle: {trajectory[-period:]}")
        else:
            print(f"✗ No stable pattern detected")

    return trajectory, period


def detect_logic_cycle(trajectory: List[int], threshold: int = 0) -> Optional[int]:
    """
    Detect periodic cycle in logic trajectory

    Returns period if found, None otherwise
    """
    if len(trajectory) < 2:
        return None

    # Check for convergence (period-1)
    if len(trajectory) >= 3:
        if trajectory[-1] == trajectory[-2] == trajectory[-3]:
            return 1  # Stable eigenstate

    # Check for period-2, period-4, period-8
    for period in [2, 4, 8]:
        if len(trajectory) < 2 * period:
            continue

        is_periodic = True
        for offset in range(period):
            idx_curr = len(trajectory) - 1 - offset
            idx_prev = idx_curr - period

            if idx_prev < 0:
                is_periodic = False
                break

            if trajectory[idx_curr] != trajectory[idx_prev]:
                is_periodic = False
                break

        if is_periodic:
            return period

    return None


def verify_truth_table() -> bool:
    """
    Verify Eigengate against complete truth table

    Returns True if all 16 cases match expected outputs
    """
    # Truth table from specification
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

    all_correct = True
    mismatches = []

    for (A, B, D, C), expected_Q25 in truth_table.items():
        actual_Q25 = eigengate(A, B, D, C)

        if actual_Q25 != expected_Q25:
            all_correct = False
            mismatches.append((A, B, D, C, expected_Q25, actual_Q25))

    if mismatches:
        print("❌ Truth table mismatches found:")
        for A, B, D, C, expected, actual in mismatches:
            print(f"  A={A} B={B} D={C} C={C}: expected {expected}, got {actual}")

    return all_correct


def test_all_feedback_configurations():
    """
    Test all 8 fixed input configurations (A, B, D) with feedback
    """
    print("\n" + "=" * 70)
    print("EIGENGATE FEEDBACK ANALYSIS - ALL CONFIGURATIONS")
    print("=" * 70)
    print()

    results = []

    for A in [0, 1]:
        for B in [0, 1]:
            for D in [0, 1]:
                print(f"\nConfiguration: A={A}, B={B}, D={D}")
                print("─" * 40)

                trajectory, period = simulate_eigengate_feedback(
                    A, B, D, initial_C=0, max_steps=10, verbose=False
                )

                # Determine behavior
                if period == 1:
                    behavior = f"Stable (Q25={trajectory[-1]})"
                elif period == 2:
                    behavior = f"Oscillating (period-2: {trajectory[-2:]})"
                elif period:
                    behavior = f"Periodic (period-{period})"
                else:
                    behavior = "No clear pattern"

                print(f"  Trajectory: {trajectory}")
                print(f"  Behavior: {behavior}")

                results.append({
                    'A': A, 'B': B, 'D': D,
                    'trajectory': trajectory,
                    'period': period,
                    'behavior': behavior
                })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Eigengate Feedback Behavior")
    print("=" * 70)
    print(f"{'A':<3} {'B':<3} {'D':<3} {'Period':<10} {'Behavior':<30}")
    print("─" * 70)

    for r in results:
        period_str = str(r['period']) if r['period'] else "none"
        print(f"{r['A']:<3} {r['B']:<3} {r['D']:<3} {period_str:<10} {r['behavior']:<30}")

    return results


def connect_to_eigenstate_framework(A: int, B: int, D: int, C: int) -> Dict:
    """
    Connect Eigengate to broader eigenstate framework

    Shows how Q25 relates to (L, R, V, M) pattern and metric regimes

    Returns
    -------
    analysis : dict
        {
            'eigengate_output': Q25,
            'mapping_to_LRVM': interpretation,
            'regime_classification': light-like/time-like/space-like,
            'eigenstate_indicator': bool
        }
    """
    result = eigengate_with_components(A, B, D, C)

    # Map to (L, R, V, M) framework
    # A, B represent lexical/relational duality
    # D, C represent value/context duality
    # Q25 acts as Meta-observer

    mapping = {
        'L': A,  # Lexical (subject)
        'R': B,  # Relational (verb)
        'V': D,  # Value (object)
        'Context': C,  # Context/observer
        'M': result['Q25']  # Meta-understanding
    }

    # Regime classification
    xor_AB = result['XOR_AB']
    xnor_DC = result['XNOR_DC']

    if xor_AB and xnor_DC:
        regime = "time-like (stable, both conditions met)"
    elif not xor_AB and not xnor_DC:
        regime = "space-like (unstable, conflict)"
    else:
        regime = "light-like (transition, partial balance)"

    # Eigenstate indicator
    # Q25=1 with feedback convergence indicates eigenstate
    eigenstate = result['balanced']

    return {
        'eigengate_output': result['Q25'],
        'intermediate_gates': {
            'XOR_AB': xor_AB,
            'XNOR_DC': xnor_DC
        },
        'mapping_to_LRVM': mapping,
        'regime_classification': regime,
        'eigenstate_indicator': eigenstate,
        'interpretation': (
            "Balanced - eigenstate possible" if eigenstate
            else "Imbalanced - no eigenstate"
        )
    }


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EIGENGATE: FUNDAMENTAL BALANCE DETECTION CIRCUIT")
    print("=" * 70)
    print()
    print("Circuit: Q25 = (A ⊕ B) ∨ (D ⊙ C)")
    print("  - XOR detects asymmetry (A≠B)")
    print("  - XNOR detects symmetry (D==C)")
    print("  - OR combines to signal balance")
    print()

    # Verify truth table
    print("─" * 70)
    print("TRUTH TABLE VERIFICATION")
    print("─" * 70)

    if verify_truth_table():
        print("✓ All 16 cases match expected truth table")
    else:
        print("✗ Truth table mismatch detected")

    print()

    # Test specific examples from user description
    print("─" * 70)
    print("TEST CASES FROM SPECIFICATION")
    print("─" * 70)
    print()

    # Example 1: A=1, B=0, D=1, C=0 (from diagram)
    print("Example 1: A=1, B=0, D=1 (starting C=0)")
    traj1, period1 = simulate_eigengate_feedback(1, 0, 1, 0, verbose=True)

    print("\n")

    # Example 2: A=0, B=0, D=0 (oscillation case)
    print("Example 2: A=0, B=0, D=0 (should oscillate)")
    traj2, period2 = simulate_eigengate_feedback(0, 0, 0, 0, verbose=True)

    print("\n")

    # Example 3: A=1, B=1, D=1 (stable case)
    print("Example 3: A=1, B=1, D=1 (should stabilize)")
    traj3, period3 = simulate_eigengate_feedback(1, 1, 1, 0, verbose=True)

    # Test all configurations
    test_all_feedback_configurations()

    # Connection to framework
    print("\n" + "=" * 70)
    print("CONNECTION TO EIGENSTATE FRAMEWORK")
    print("=" * 70)
    print()

    example_connection = connect_to_eigenstate_framework(1, 0, 1, 0)
    print(f"Eigengate output (Q25): {example_connection['eigengate_output']}")
    print(f"Regime: {example_connection['regime_classification']}")
    print(f"Mapping to (L,R,V,M): {example_connection['mapping_to_LRVM']}")
    print(f"Interpretation: {example_connection['interpretation']}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print("""
1. Eigengate = fundamental balance detector via XOR ∨ XNOR
2. Q25 measurement acts as "light-like" resolution
3. Feedback loop creates time-like (oscillation) behavior
4. Stable convergence = eigenstate detected
5. Maps to (L,R,V,M) framework: Q25 ≈ Meta-observer
6. Universal pattern across text, EM, gravity, quantum
7. XOR detects asymmetry, XNOR detects symmetry
8. OR combination resolves to balanced/imbalanced state
    """)
