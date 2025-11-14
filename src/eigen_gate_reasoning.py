#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eigengate Reasoning Layer - Deterministic Semantic Analysis

Implements Boolean logic gates for deterministic semantic reasoning, mapping
the Eigengate circuit Q25 = (A ⊕ B) ∨ (D ⊙ C) to semantic state evaluation.

Key Concepts:
- Q (light-like): Measurement that resolves time-like/space-like oscillations
- Eigengate balance detection: Symmetry/asymmetry evaluation
- 5W1H extraction: Who, What, When, Where, Why, How from semantic states
- Deterministic reasoning without probabilistic collapse

Theoretical Foundation:
    Q = (A ⊕ B) ∨ (D ⊙ C)

Where:
- A, B: Primary semantic pair (e.g., L and R)
- D, C: Secondary semantic pair (e.g., V and M)
- ⊕: XOR (asymmetry detection)
- ⊙: XNOR (symmetry detection)
- ∨: OR (balance integration)

Q = 1: System is balanced (asymmetry in A-B OR symmetry in D-C)
Q = 0: System is imbalanced (symmetry in A-B AND asymmetry in D-C)

Regime Interpretation:
- Light-like (Q): Resolves conflicts, nullifies oscillations
- Time-like: Stable, sequential (ds² > 0)
- Space-like: Exploring, semantic (ds² < 0)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class EigengateState:
    """
    Eigengate evaluation result with 5W1H reasoning

    Attributes
    ----------
    Q : int (0 or 1)
        Eigengate output (1 = balanced, 0 = imbalanced)
    A : int (0 or 1)
        Primary input A
    B : int (0 or 1)
        Primary input B
    D : int (0 or 1)
        Secondary input D
    C : int (0 or 1)
        Secondary input C
    xor_AB : int (0 or 1)
        XOR result for A and B (asymmetry detection)
    xnor_DC : int (0 or 1)
        XNOR result for D and C (symmetry detection)
    """
    Q: int
    A: int
    B: int
    D: int
    C: int
    xor_AB: int
    xnor_DC: int

    def is_balanced(self) -> bool:
        """Check if system is balanced (Q = 1)"""
        return self.Q == 1

    def has_AB_asymmetry(self) -> bool:
        """Check if A-B pair is asymmetric"""
        return self.xor_AB == 1

    def has_DC_symmetry(self) -> bool:
        """Check if D-C pair is symmetric"""
        return self.xnor_DC == 1


def eigengate_Q25(A: int, B: int, D: int, C: int) -> EigengateState:
    """
    Compute Eigengate Q25 = (A ⊕ B) ∨ (D ⊙ C)

    Parameters
    ----------
    A : int (0 or 1)
        Primary input A
    B : int (0 or 1)
        Primary input B
    D : int (0 or 1)
        Secondary input D
    C : int (0 or 1)
        Secondary input C

    Returns
    -------
    EigengateState
        Complete evaluation with Q output and intermediate values

    Examples
    --------
    >>> state = eigengate_Q25(A=1, B=0, D=1, C=0)
    >>> state.Q
    1
    >>> state.is_balanced()
    True
    >>> state.has_AB_asymmetry()
    True
    """
    # XOR: A ⊕ B (asymmetry detection)
    xor_AB = A ^ B

    # XNOR: D ⊙ C (symmetry detection)
    # XNOR(D, C) = NOT(D XOR C) = 1 if D == C, else 0
    xnor_DC = 1 - (D ^ C)

    # OR: Combine for balance
    Q = xor_AB | xnor_DC

    return EigengateState(
        Q=Q,
        A=A,
        B=B,
        D=D,
        C=C,
        xor_AB=xor_AB,
        xnor_DC=xnor_DC
    )


def semantic_to_eigengate(L: int, R: int, V: int, M: int,
                          threshold: int = 128) -> Tuple[int, int, int, int]:
    """
    Map semantic state (L, R, V, M) to Eigengate inputs (A, B, D, C)

    Converts continuous semantic values to binary inputs for eigengate logic.

    Parameters
    ----------
    L : int (0-255)
        Lexical component
    R : int (0-255)
        Relational component
    V : int (0-255)
        Value component
    M : int (0-255)
        Meta component
    threshold : int, optional
        Binarization threshold (default: 128)

    Returns
    -------
    Tuple[int, int, int, int]
        (A, B, D, C) binary inputs

    Mapping Strategy:
    - A = binarize(L): Lexical presence
    - B = binarize(R): Relational presence
    - D = binarize(V): Value presence
    - C = binarize(M): Meta presence
    """
    A = 1 if L >= threshold else 0
    B = 1 if R >= threshold else 0
    D = 1 if V >= threshold else 0
    C = 1 if M >= threshold else 0

    return A, B, D, C


def analyze_balance(state: EigengateState) -> Dict[str, str]:
    """
    Generate 5W1H analysis from Eigengate state

    Parameters
    ----------
    state : EigengateState
        Eigengate evaluation result

    Returns
    -------
    Dict[str, str]
        5W1H analysis with keys: what, who, when, where, why, how

    Examples
    --------
    >>> state = eigengate_Q25(A=1, B=0, D=1, C=0)
    >>> analysis = analyze_balance(state)
    >>> print(analysis['what'])
    'System exhibits balanced state'
    """
    # What: System state
    if state.is_balanced():
        what = "System exhibits balanced state"
    else:
        what = "System displays imbalanced state"

    # Who: Entity involvement (A-B pair)
    if state.has_AB_asymmetry():
        who = f"Asymmetric entities (A={state.A}, B={state.B}) drive balance"
    else:
        who = f"Symmetric entities (A={state.A}, B={state.B}) indicate equivalence"

    # When: Temporal resolution
    if state.is_balanced():
        when = "Resolution occurs immediately (light-like measurement)"
    else:
        when = "Oscillation persists (time-like/space-like conflict)"

    # Where: Localization
    components = []
    if state.has_AB_asymmetry():
        components.append("A-B asymmetry")
    if state.has_DC_symmetry():
        components.append("D-C symmetry")

    if components:
        where = f"Balance localized in: {', '.join(components)}"
    else:
        where = "Global imbalance across all pairs"

    # Why: Rationale
    if state.is_balanced():
        if state.has_AB_asymmetry() and state.has_DC_symmetry():
            why = "Both asymmetry (A-B) and symmetry (D-C) conditions satisfied"
        elif state.has_AB_asymmetry():
            why = "Asymmetry in A-B satisfies equilibrium condition"
        else:  # must have DC symmetry
            why = "Symmetry in D-C satisfies equilibrium condition"
    else:
        why = "Symmetry in A-B AND asymmetry in D-C creates imbalance"

    # How: Mechanism
    how = (f"XOR(A={state.A}, B={state.B})={state.xor_AB}, "
           f"XNOR(D={state.D}, C={state.C})={state.xnor_DC}, "
           f"Q={state.Q}")

    return {
        "what": what,
        "who": who,
        "when": when,
        "where": where,
        "why": why,
        "how": how
    }


def resolve_oscillation(L: int, R: int, V: int, M: int,
                       threshold: int = 128) -> Tuple[bool, Dict[str, str]]:
    """
    Use Eigengate Q as light-like resolver for semantic oscillations

    Parameters
    ----------
    L, R, V, M : int (0-255)
        Semantic state components
    threshold : int, optional
        Binarization threshold

    Returns
    -------
    Tuple[bool, Dict[str, str]]
        (is_resolved, analysis_5w1h)

    Interpretation:
    - True: Oscillation resolved (Q=1, light-like measurement)
    - False: Oscillation continues (Q=0, time/space conflict)

    Examples
    --------
    >>> resolved, analysis = resolve_oscillation(L=200, R=50, V=180, M=200)
    >>> resolved
    True
    >>> print(analysis['what'])
    'System exhibits balanced state'
    """
    # Map to binary inputs
    A, B, D, C = semantic_to_eigengate(L, R, V, M, threshold)

    # Compute eigengate
    state = eigengate_Q25(A, B, D, C)

    # Generate analysis
    analysis = analyze_balance(state)

    return state.is_balanced(), analysis


def classify_regime_eigengate(state: EigengateState) -> str:
    """
    Classify regime using Eigengate framework

    Parameters
    ----------
    state : EigengateState
        Eigengate evaluation result

    Returns
    -------
    str
        Regime classification: 'light-like', 'time-like', 'space-like', 'oscillating'

    Mapping:
    - Q=1: Light-like (resolved, balanced)
    - Q=0 with AB asymmetry: Time-like (sequential, causal)
    - Q=0 with DC asymmetry: Space-like (exploring, non-causal)
    - Q=0 with both: Oscillating (unresolved conflict)
    """
    if state.Q == 1:
        return "light-like"  # Resolved, balanced

    # Q = 0: imbalanced
    if not state.has_AB_asymmetry() and not state.has_DC_symmetry():
        # Symmetric A-B, Asymmetric D-C
        if state.A == state.B == 1:
            return "time-like"  # Both high: sequential, stable
        else:
            return "space-like"  # Both low or mixed: exploring

    return "oscillating"  # Should not happen if Q computed correctly


def generate_truth_table() -> List[Dict]:
    """
    Generate complete truth table for Q25 = (A ⊕ B) ∨ (D ⊙ C)

    Returns
    -------
    List[Dict]
        All 16 input combinations with Q output and analysis
    """
    table = []

    for decimal in range(16):
        # Binary representation: ABDC
        A = (decimal >> 3) & 1
        B = (decimal >> 2) & 1
        D = (decimal >> 1) & 1
        C = decimal & 1

        state = eigengate_Q25(A, B, D, C)
        analysis = analyze_balance(state)

        table.append({
            "decimal": decimal,
            "A": A,
            "B": B,
            "D": D,
            "C": C,
            "Q": state.Q,
            "xor_AB": state.xor_AB,
            "xnor_DC": state.xnor_DC,
            "balanced": state.is_balanced(),
            "regime": classify_regime_eigengate(state),
            **analysis
        })

    return table


def print_truth_table():
    """Print formatted truth table for Q25"""
    table = generate_truth_table()

    print("=" * 80)
    print("Eigengate Q25 Truth Table: Q = (A ⊕ B) ∨ (D ⊙ C)")
    print("=" * 80)
    print(f"{'Dec':<4} {'A':<2} {'B':<2} {'D':<2} {'C':<2} {'Q':<2} {'XOR':<4} {'XNOR':<5} {'Balance':<10} {'Regime':<12}")
    print("-" * 80)

    for row in table:
        print(f"{row['decimal']:<4} {row['A']:<2} {row['B']:<2} {row['D']:<2} {row['C']:<2} "
              f"{row['Q']:<2} {row['xor_AB']:<4} {row['xnor_DC']:<5} "
              f"{'Yes' if row['balanced'] else 'No':<10} {row['regime']:<12}")

    print("=" * 80)
    print("\nBalance Statistics:")
    balanced_count = sum(1 for row in table if row['balanced'])
    print(f"  Balanced states: {balanced_count}/16 ({balanced_count/16*100:.1f}%)")
    print(f"  Imbalanced states: {16-balanced_count}/16 ({(16-balanced_count)/16*100:.1f}%)")

    print("\nRegime Distribution:")
    regimes = {}
    for row in table:
        regime = row['regime']
        regimes[regime] = regimes.get(regime, 0) + 1

    for regime, count in sorted(regimes.items()):
        print(f"  {regime}: {count}/16 ({count/16*100:.1f}%)")


# Example usage and testing
if __name__ == "__main__":
    print("Eigengate Reasoning Layer - Examples\n")

    # Example 1: Direct eigengate computation
    print("Example 1: Direct Eigengate Q25")
    print("-" * 40)
    state = eigengate_Q25(A=1, B=0, D=1, C=0)
    print(f"Inputs: A=1, B=0, D=1, C=0")
    print(f"Output Q: {state.Q}")
    print(f"Balanced: {state.is_balanced()}")
    print(f"A-B Asymmetry: {state.has_AB_asymmetry()}")
    print(f"D-C Symmetry: {state.has_DC_symmetry()}")
    print()

    # Example 2: 5W1H analysis
    print("Example 2: 5W1H Analysis")
    print("-" * 40)
    analysis = analyze_balance(state)
    for key, value in analysis.items():
        print(f"{key.upper()}: {value}")
    print()

    # Example 3: Semantic state resolution
    print("Example 3: Semantic State Resolution")
    print("-" * 40)
    L, R, V, M = 200, 50, 180, 200
    print(f"Semantic state: L={L}, R={R}, V={V}, M={M}")
    resolved, analysis = resolve_oscillation(L, R, V, M)
    print(f"Resolved: {resolved}")
    print(f"WHAT: {analysis['what']}")
    print(f"WHY: {analysis['why']}")
    print()

    # Example 4: Truth table
    print("Example 4: Complete Truth Table")
    print_truth_table()
