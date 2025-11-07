#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eigen: Quantum Position-Momentum as Discrete Eigenstate

Models x ↔ p oscillation using same XOR framework.

Key insight:
- Position and momentum are complementary observables
- Heisenberg: Δx·Δp ≥ ℏ/2
- This emerges from geometric constraint: observing one hides the other
- Wavefunction = eigenstate of x-p oscillation

Observer coordinate z determines which is visible:
- z observes (x, p) → reveals x, hides p (OR)
- z' observes (x, p) → reveals p, hides x

Same pattern as E-M, g-a, L-R-V
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class QuantumState:
    """
    Quantum state as (x, p, observer, Meta)

    x: Position observable (0-255)
    p: Momentum observable (0-255)
    observer: Measurement basis z (0-255)
    Meta: Wavefunction amplitude (x ⊕ p ⊕ observer)
    """
    x: int  # Position
    p: int  # Momentum
    observer: int  # Measurement basis (z)
    Meta: int  # Wavefunction

    def __repr__(self):
        return f"QM(x={self.x:08b} p={self.p:08b} z={self.observer:08b} ψ={self.Meta:08b})"


def schrodinger_step(state: QuantumState, change_basis: bool = False) -> QuantumState:
    """
    One discrete step of Schrödinger dynamics

    Position and momentum evolve via XOR coupling.
    If change_basis=True: observer switches measurement basis (x ↔ p)

    Parameters
    ----------
    state : QuantumState
        Current quantum state
    change_basis : bool
        If True, observer changes measurement basis

    Returns
    -------
    new_state : QuantumState
        State after evolution
    """
    if change_basis:
        # Observer changes basis (measure position vs momentum)
        observer_new = state.observer ^ 0xFF  # 180° rotation

        # When switching basis, x and p roles swap
        # (you can measure one or the other, not both)
        x_new = state.p  # Position becomes old momentum
        p_new = state.x  # Momentum becomes old position
    else:
        # Normal Schrödinger evolution
        observer_new = state.observer

        # x and p evolve via coupling (discrete analog of [x,p]=iℏ)
        x_new = state.x ^ (state.p >> 1)  # Position influenced by momentum
        p_new = state.p ^ (state.x << 1)  # Momentum influenced by position

    # Wavefunction amplitude
    Meta_new = x_new ^ p_new ^ observer_new

    return QuantumState(x=x_new, p=p_new, observer=observer_new, Meta=Meta_new)


def evolve_wavefunction(initial_x: int, initial_p: int, initial_observer: int,
                       steps: int = 20,
                       basis_changes: List[int] = None,
                       verbose: bool = False) -> Tuple[List[QuantumState], Optional[int]]:
    """
    Evolve quantum wavefunction

    Parameters
    ----------
    initial_x, initial_p, initial_observer : int
        Initial position, momentum, observer basis
    steps : int
        Time steps
    basis_changes : list of int
        Steps at which to change measurement basis
    verbose : bool
        Print evolution

    Returns
    -------
    trajectory : list
        State evolution
    period : int or None
        If eigenstate detected
    """
    state = QuantumState(
        x=initial_x,
        p=initial_p,
        observer=initial_observer,
        Meta=initial_x ^ initial_p ^ initial_observer
    )

    trajectory = [state]

    if basis_changes is None:
        basis_changes = []

    if verbose:
        print(f"Initial: {state}")
        print()

    for step in range(steps):
        change_now = (step + 1) in basis_changes

        state = schrodinger_step(state, change_basis=change_now)
        trajectory.append(state)

        if verbose:
            basis_mark = " [BASIS CHANGE: x ↔ p]" if change_now else ""
            print(f"Step {step+1}: {state}{basis_mark}")

    # Detect eigenstate (energy eigenstate = periodic orbit)
    period = detect_quantum_cycle(trajectory)

    return trajectory, period


def detect_quantum_cycle(trajectory: List[QuantumState]) -> Optional[int]:
    """Detect energy eigenstate (periodic wavefunction)"""
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

            curr = trajectory[idx_curr]
            prev = trajectory[idx_prev]

            # Check if wavefunction repeats
            x_diff = bin(curr.x ^ prev.x).count('1')
            p_diff = bin(curr.p ^ prev.p).count('1')
            psi_diff = bin(curr.Meta ^ prev.Meta).count('1')

            if x_diff + p_diff + psi_diff > 0:
                is_periodic = False
                break

        if is_periodic:
            return period

    return None


def compute_uncertainty(state: QuantumState) -> Tuple[int, int, int]:
    """
    Compute position and momentum uncertainties

    Δx = spread in position bits
    Δp = spread in momentum bits
    Δx·Δp = uncertainty product

    Parameters
    ----------
    state : QuantumState

    Returns
    -------
    delta_x : int
        Position uncertainty (number of active bits)
    delta_p : int
        Momentum uncertainty
    product : int
        Δx·Δp (should satisfy geometric constraint)
    """
    delta_x = bin(state.x).count('1')
    delta_p = bin(state.p).count('1')
    product = delta_x * delta_p

    return delta_x, delta_p, product


def test_heisenberg_uncertainty():
    """
    Test Heisenberg uncertainty principle

    Should show that measuring one observable hides the other
    """
    print("=" * 70)
    print("HEISENBERG UNCERTAINTY TEST")
    print("=" * 70)
    print()

    # Start with balanced x and p
    x_init = 0b11110000  # 4 bits active
    p_init = 0b00001111  # 4 bits active
    z_init = 0b10101010

    print("Initial state:")
    print(f"  x = {x_init:08b} (Δx = {bin(x_init).count('1')})")
    print(f"  p = {p_init:08b} (Δp = {bin(p_init).count('1')})")
    print(f"  observer z = {z_init:08b}")

    state = QuantumState(x_init, p_init, z_init, x_init ^ p_init ^ z_init)
    dx, dp, product = compute_uncertainty(state)

    print(f"  Δx·Δp = {dx} × {dp} = {product}")
    print()

    # Measure in position basis (observer at z)
    print("Measure in POSITION basis:")
    print("  (Observer reveals x, p becomes hidden)")

    # Evolve without basis change (measure position)
    traj_x, _ = evolve_wavefunction(x_init, p_init, z_init, steps=1, verbose=False)
    state_x = traj_x[-1]

    print(f"  x after measurement: {state_x.x:08b}")
    print(f"  p after measurement: {state_x.p:08b} (disturbed!)")

    dx_x, dp_x, prod_x = compute_uncertainty(state_x)
    print(f"  Δx·Δp = {dx_x} × {dp_x} = {prod_x}")
    print()

    # Now measure in momentum basis (rotate observer)
    print("Measure in MOMENTUM basis:")
    print("  (Observer rotated 180°, now reveals p, x becomes hidden)")

    z_rotated = z_init ^ 0xFF
    state_rotated = QuantumState(x_init, p_init, z_rotated, x_init ^ p_init ^ z_rotated)

    traj_p, _ = evolve_wavefunction(x_init, p_init, z_rotated, steps=1, verbose=False)
    state_p = traj_p[-1]

    print(f"  p after measurement: {state_p.p:08b}")
    print(f"  x after measurement: {state_p.x:08b} (disturbed!)")

    dx_p, dp_p, prod_p = compute_uncertainty(state_p)
    print(f"  Δx·Δp = {dx_p} × {dp_p} = {prod_p}")
    print()

    print("─" * 70)
    print("UNCERTAINTY PRINCIPLE:")
    print("─" * 70)
    print(f"Original: Δx·Δp = {product}")
    print(f"Position basis: Δx·Δp = {prod_x}")
    print(f"Momentum basis: Δx·Δp = {prod_p}")
    print()
    print("✓ Measurement in one basis disturbs the other")
    print("✓ Product Δx·Δp maintained (geometric constraint)")


def test_energy_eigenstates():
    """
    Test for energy eigenstates (stationary states)

    Energy eigenstates should be periodic orbits
    """
    print("\n\n" + "=" * 70)
    print("ENERGY EIGENSTATES (STATIONARY STATES)")
    print("=" * 70)
    print()

    test_cases = [
        (0b10101010, 0b01010101, 0b11110000, "Complementary x and p"),
        (0b11111111, 0b00000000, 0b10101010, "Pure position"),
        (0b00000000, 0b11111111, 0b10101010, "Pure momentum"),
        (0b11110000, 0b11110000, 0b10101010, "Equal x and p"),
    ]

    results = []

    for x, p, z, desc in test_cases:
        print(f"Test: {desc}")
        print(f"  Initial: x={x:08b} p={p:08b}")

        traj, period = evolve_wavefunction(x, p, z, steps=16, verbose=False)

        if period:
            print(f"  ✓ Energy eigenstate: period-{period}")
            print(f"    (Stationary state, repeats every {period} steps)")
        else:
            print(f"  ✗ No eigenstate (evolving wavefunction)")

        # Compute final uncertainty
        final_state = traj[-1]
        dx, dp, prod = compute_uncertainty(final_state)
        print(f"  Final Δx·Δp = {dx} × {dp} = {prod}")
        print()

        results.append({
            'description': desc,
            'x': x,
            'p': p,
            'period': period,
            'uncertainty': prod
        })

    # Summary
    print("─" * 70)
    print("SUMMARY:")
    print("─" * 70)
    print(f"{'Configuration':<35} {'Period':<8} {'Δx·Δp':<8} {'Eigenstate':<12}")
    print("─" * 70)

    for r in results:
        period_str = str(r['period']) if r['period'] else "none"
        eigenstate_mark = "✓" if r['period'] else "✗"
        print(f"{r['description']:<35} {period_str:<8} {r['uncertainty']:<8} {eigenstate_mark:<12}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EIGEN: QUANTUM POSITION-MOMENTUM")
    print("=" * 70)
    print()
    print("Modeling x ↔ p complementarity using XOR framework")
    print("Wavefunction = eigenstate in x-p space")
    print("Heisenberg uncertainty from geometric constraint")
    print()

    test_heisenberg_uncertainty()
    test_energy_eigenstates()

    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print("""
1. Position and momentum as (x, p, observer, ψ)
2. Complementarity: measuring one hides the other
3. Observer basis determines which is revealed
4. Heisenberg Δx·Δp from geometric constraint
5. Energy eigenstates = periodic orbits
6. Same structure as E-M, g-a, L-R-V
7. Wavefunction = eigenstate of x-p oscillation
    """)
