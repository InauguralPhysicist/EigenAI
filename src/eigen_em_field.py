#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eigen: Electromagnetic Field as Discrete Eigenstate

Models E ↔ M oscillation using same XOR framework as text.

Key insight:
- EM field = oscillating eigenstate measured through same operation
- Light = period-k orbit in E-M eigenspace
- Frequency = period of cycle
- This is the same pattern as L ↔ R ↔ V text understanding

Connection:
- Maxwell: ∂E/∂t ~ ∇×B, ∂B/∂t ~ ∇×E
- Our framework: E_new = E ⊕ rotate(M), M_new = M ⊕ rotate(E)
- Oscillation creates light
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class EMState:
    """
    Electromagnetic state as (E, M, Meta) triple

    E: Electric component (0-255)
    M: Magnetic component (0-255)
    Meta: Observer coordinate (E ⊕ M)
    """

    E: int
    M_field: int  # Magnetic (renamed to avoid confusion with Meta)
    Meta: int

    def __repr__(self):
        return f"EM(E={self.E:08b} M={self.M_field:08b} Meta={self.Meta:08b})"


def rotate_byte(value: int, n: int = 1) -> int:
    """Circular bit rotation (simulates ∇× curl operator)"""
    value &= 0xFF  # Ensure 8 bits
    n = n % 8
    return ((value << n) | (value >> (8 - n))) & 0xFF


def maxwell_step(state: EMState, coupling: int = 1) -> EMState:
    """
    One discrete step of Maxwell's equations

    Discrete analog of:
    ∂E/∂t ~ ∇×M (changing M creates E)
    ∂M/∂t ~ ∇×E (changing E creates M)

    Using XOR and rotation:
    E_new = E ⊕ rotate(M)
    M_new = M ⊕ rotate(E)

    Parameters
    ----------
    state : EMState
        Current EM state
    coupling : int
        Rotation amount (simulates coupling strength)

    Returns
    -------
    new_state : EMState
        State after one oscillation step
    """
    # ∂E/∂t ~ ∇×M  → E changes based on rotated M
    E_new = state.E ^ rotate_byte(state.M_field, coupling)

    # ∂M/∂t ~ ∇×E  → M changes based on rotated E
    M_new = state.M_field ^ rotate_byte(state.E, coupling)

    # Meta observation
    Meta_new = E_new ^ M_new

    return EMState(E=E_new, M_field=M_new, Meta=Meta_new)


def propagate_em_field(
    initial_E: int,
    initial_M: int,
    steps: int = 20,
    coupling: int = 1,
    verbose: bool = False,
) -> Tuple[List[EMState], Optional[int]]:
    """
    Propagate EM field through time

    Looks for periodic orbits (light waves)

    Parameters
    ----------
    initial_E, initial_M : int
        Initial electric and magnetic components
    steps : int
        Number of time steps
    coupling : int
        Coupling strength (rotation amount)
    verbose : bool
        Print trajectory

    Returns
    -------
    trajectory : list of EMState
        Field evolution
    period : int or None
        If periodic orbit detected, returns period (wavelength)
    """
    state = EMState(E=initial_E, M_field=initial_M, Meta=initial_E ^ initial_M)
    trajectory = [state]

    if verbose:
        print(f"Initial: {state}")
        print()

    for step in range(steps):
        state = maxwell_step(state, coupling)
        trajectory.append(state)

        if verbose:
            print(f"Step {step+1}: {state}")

    # Detect period
    period = detect_em_cycle(trajectory)

    return trajectory, period


def detect_em_cycle(trajectory: List[EMState], threshold: int = 0) -> Optional[int]:
    """
    Detect periodic orbit in EM trajectory

    Returns wavelength (period) if light wave detected
    """
    if len(trajectory) < 4:
        return None

    for period in range(2, min(9, len(trajectory) // 2 + 1)):
        is_periodic = True

        for offset in range(period):
            idx_curr = len(trajectory) - 1 - offset
            idx_prev = idx_curr - period

            if idx_prev < 0:
                is_periodic = False
                break

            # Check if E and M match
            curr_state = trajectory[idx_curr]
            prev_state = trajectory[idx_prev]

            E_diff = bin(curr_state.E ^ prev_state.E).count("1")
            M_diff = bin(curr_state.M_field ^ prev_state.M_field).count("1")

            if E_diff + M_diff > threshold:
                is_periodic = False
                break

        if is_periodic:
            return period

    return None


def analyze_em_field(
    initial_E: int, initial_M: int, coupling: int = 1, label: str = ""
) -> dict:
    """
    Full analysis of EM field evolution

    Returns
    -------
    result : dict
        Contains trajectory, period, regime, etc.
    """
    print("=" * 70)
    if label:
        print(f"EM FIELD ANALYSIS: {label}")
    else:
        print(f"EM FIELD: E={initial_E:08b} M={initial_M:08b}")
    print("=" * 70)
    print()

    trajectory, period = propagate_em_field(
        initial_E, initial_M, steps=20, coupling=coupling, verbose=True
    )

    print()
    print("─" * 70)
    print("ANALYSIS:")
    print("─" * 70)

    if period:
        print(f"✓ Light wave detected: period-{period}")
        print(f"  Wavelength: {period} steps")
        print(f"  Frequency: 1/{period}")
    else:
        print("✗ No periodic orbit (field evolving)")

    # Compute oscillation metrics
    E_oscillation = np.array([s.E for s in trajectory])
    M_oscillation = np.array([s.M_field for s in trajectory])

    E_variance = np.var(E_oscillation)
    M_variance = np.var(M_oscillation)

    print(f"\nField oscillation:")
    print(f"  E variance: {E_variance:.1f}")
    print(f"  M variance: {M_variance:.1f}")
    print(f"  Ratio E/M: {E_variance/M_variance if M_variance > 0 else 'inf':.2f}")

    # Check if light-like (E and M oscillate symmetrically)
    if abs(E_variance - M_variance) < 10:
        print(f"  → Symmetric oscillation (light-like)")
    elif E_variance > M_variance:
        print(f"  → E-dominated field")
    else:
        print(f"  → M-dominated field")

    return {
        "trajectory": trajectory,
        "period": period,
        "E_variance": E_variance,
        "M_variance": M_variance,
    }


def test_light_eigenstates():
    """
    Test different initial conditions for light waves
    """
    print("\n" + "=" * 70)
    print("SEARCHING FOR LIGHT EIGENSTATES")
    print("=" * 70)
    print()

    test_cases = [
        (0b10101010, 0b01010101, "Alternating bits (E ⊥ M)"),
        (0b11110000, 0b00001111, "Complementary halves"),
        (0b11111111, 0b00000000, "Pure E field"),
        (0b00000000, 0b11111111, "Pure M field"),
        (0b11111111, 0b11111111, "Maximum field"),
    ]

    results = []

    for E_init, M_init, description in test_cases:
        print(f"\n{'─' * 70}")
        print(f"Test: {description}")
        print(f"Initial: E={E_init:08b} M={M_init:08b}")
        print(f"{'─' * 70}")

        trajectory, period = propagate_em_field(E_init, M_init, steps=16, verbose=False)

        if period:
            print(f"✓ Period-{period} eigenstate (light wave)")
        else:
            print(f"✗ No eigenstate")

        results.append(
            {"description": description, "E": E_init, "M": M_init, "period": period}
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Light Eigenstate Detection")
    print("=" * 70)
    print(f"{'Configuration':<35} {'E':<10} {'M':<10} {'Period':<8}")
    print("─" * 70)

    for r in results:
        period_str = str(r["period"]) if r["period"] else "none"
        eigenstate_mark = "✓" if r["period"] else "✗"
        print(
            f"{r['description']:<35} {r['E']:08b} {r['M']:08b} {period_str:<8} {eigenstate_mark}"
        )


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EIGEN: ELECTROMAGNETIC FIELD AS DISCRETE EIGENSTATE")
    print("=" * 70)
    print()
    print("Modeling E ↔ M oscillation using XOR framework")
    print("Light = periodic orbit in E-M eigenspace")
    print()

    # Test 1: Symmetric initial condition
    analyze_em_field(
        initial_E=0b10101010, initial_M=0b01010101, label="Symmetric E-M (orthogonal)"
    )

    print("\n\n")

    # Test 2: Pure E field
    analyze_em_field(initial_E=0b11111111, initial_M=0b00000000, label="Pure E field")

    print("\n\n")

    # Search for more eigenstates
    test_light_eigenstates()

    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print(
        """
1. EM field modeled as (E, M, Meta) eigenstate
2. Maxwell's equations → XOR cascade with rotation
3. Light = periodic orbit (E → M → E → M)
4. Period = wavelength, 1/period = frequency
5. Symmetric E/M oscillation = light-like regime
6. Same geometric structure as L ↔ R ↔ V text
7. Fundamental dualities = eigenstates of oscillation
    """
    )
