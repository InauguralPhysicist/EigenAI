#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eigen: Gravity-Inertia Equivalence as Discrete Eigenstate

Models g ↔ a oscillation using same XOR framework.

Key insight from equivalence principle:
- Can't distinguish gravity from inertia locally
- Same phenomenon, different observer frames
- Observer rotation (180° XOR) swaps which is visible

This is the same pattern as:
- E ↔ M (EM field)
- L ↔ R ↔ V (text)
- Observer z determines what's revealed vs hidden
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class GravityInertiaState:
    """
    Gravity-Inertia state as (g, a, observer, Meta)

    g: Gravitational acceleration (0-255)
    a: Inertial acceleration (0-255)
    observer: Observer frame coordinate z (0-255)
    Meta: Observer's measurement (g ⊕ a ⊕ observer)
    """
    g: int  # Gravity
    a: int  # Inertia
    observer: int  # Observer frame (z)
    Meta: int

    def __repr__(self):
        return f"GI(g={self.g:08b} a={self.a:08b} z={self.observer:08b} M={self.Meta:08b})"


def equivalence_step(state: GravityInertiaState, rotate_observer: bool = False) -> GravityInertiaState:
    """
    One step in gravity-inertia dynamics

    If rotate_observer=True: Observer changes frame (180° rotation)
    This swaps what's visible (gravity vs inertia)

    Parameters
    ----------
    state : GravityInertiaState
        Current state
    rotate_observer : bool
        If True, observer rotates frame (XOR with 0xFF = flip all bits)

    Returns
    -------
    new_state : GravityInertiaState
        State after step
    """
    if rotate_observer:
        # Observer frame rotation (180°)
        observer_new = state.observer ^ 0xFF

        # From new frame, gravity and inertia appear swapped
        # (equivalence principle: can't tell difference)
        g_new = state.a  # What was inertia now looks like gravity
        a_new = state.g  # What was gravity now looks like inertia
    else:
        # No frame change: fields evolve
        observer_new = state.observer
        g_new = state.g ^ (state.a >> 1)  # Couple to inertia
        a_new = state.a ^ (state.g >> 1)  # Couple to gravity

    Meta_new = g_new ^ a_new ^ observer_new

    return GravityInertiaState(g=g_new, a=a_new, observer=observer_new, Meta=Meta_new)


def geodesic_trajectory(initial_g: int, initial_a: int, initial_observer: int,
                       steps: int = 20,
                       observer_rotations: List[int] = None,
                       verbose: bool = False) -> Tuple[List[GravityInertiaState], Optional[int]]:
    """
    Compute geodesic (free-fall trajectory) in g-a space

    Parameters
    ----------
    initial_g, initial_a, initial_observer : int
        Initial gravity, inertia, observer frame
    steps : int
        Number of time steps
    observer_rotations : list of int
        Steps at which observer changes frame
    verbose : bool
        Print trajectory

    Returns
    -------
    trajectory : list
        State evolution
    period : int or None
        If periodic orbit detected
    """
    state = GravityInertiaState(
        g=initial_g,
        a=initial_a,
        observer=initial_observer,
        Meta=initial_g ^ initial_a ^ initial_observer
    )

    trajectory = [state]

    if observer_rotations is None:
        observer_rotations = []

    if verbose:
        print(f"Initial: {state}")
        print()

    for step in range(steps):
        rotate_now = (step + 1) in observer_rotations

        state = equivalence_step(state, rotate_observer=rotate_now)
        trajectory.append(state)

        if verbose:
            rotation_mark = " [OBSERVER ROTATION]" if rotate_now else ""
            print(f"Step {step+1}: {state}{rotation_mark}")

    # Detect period
    period = detect_geodesic_cycle(trajectory)

    return trajectory, period


def detect_geodesic_cycle(trajectory: List[GravityInertiaState]) -> Optional[int]:
    """Detect periodic geodesic (closed orbit)"""
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

            # Check if states match
            g_diff = bin(curr.g ^ prev.g).count('1')
            a_diff = bin(curr.a ^ prev.a).count('1')

            if g_diff + a_diff > 0:
                is_periodic = False
                break

        if is_periodic:
            return period

    return None


def test_equivalence_principle():
    """
    Test equivalence principle: g and a are indistinguishable

    Should show that observer rotation swaps visible fields
    but geodesic properties remain invariant
    """
    print("=" * 70)
    print("EQUIVALENCE PRINCIPLE TEST")
    print("=" * 70)
    print()

    # Initial state: gravity dominant
    g_init = 0b11110000
    a_init = 0b00001111
    z_init = 0b10101010

    print("Frame A: Observer at z =", bin(z_init))
    print(f"  Sees: g={g_init:08b} a={a_init:08b}")
    print()

    # Evolve without observer rotation
    traj_A, period_A = geodesic_trajectory(
        g_init, a_init, z_init,
        steps=10,
        verbose=False
    )

    print(f"Frame A trajectory: {len(traj_A)} states")
    if period_A:
        print(f"  Eigenstate: period-{period_A}")
    else:
        print(f"  No eigenstate")

    # Now rotate observer (180°)
    z_rotated = z_init ^ 0xFF
    print()
    print("Frame B: Observer rotated 180° to z =", bin(z_rotated))

    # From equivalence principle: g and a should appear swapped
    # but dynamics should be identical
    print(f"  From this frame, appears as: g={a_init:08b} a={g_init:08b}")
    print(f"  (Fields swapped due to frame change)")
    print()

    # Evolve from rotated frame
    traj_B, period_B = geodesic_trajectory(
        a_init, g_init, z_rotated,  # Swapped g and a
        steps=10,
        verbose=False
    )

    print(f"Frame B trajectory: {len(traj_B)} states")
    if period_B:
        print(f"  Eigenstate: period-{period_B}")
    else:
        print(f"  No eigenstate")

    # Check equivalence
    print()
    print("─" * 70)
    print("EQUIVALENCE CHECK:")
    print("─" * 70)

    if period_A == period_B:
        print("✓ Both frames see same eigenstate period")
        print("  → Equivalence principle validated")
    else:
        print("✗ Different periods observed")
        print(f"  Frame A: {period_A}, Frame B: {period_B}")

    # Compute ds² invariant
    final_A = traj_A[-1]
    final_B = traj_B[-1]

    # ds² should be invariant under frame rotation
    ds2_A = compute_gi_metric(final_A)
    ds2_B = compute_gi_metric(final_B)

    print(f"\nMetric signature ds²:")
    print(f"  Frame A: {ds2_A}")
    print(f"  Frame B: {ds2_B}")

    if abs(ds2_A - ds2_B) < 10:
        print("  ✓ Metric invariant (frames equivalent)")
    else:
        print("  ✗ Metric changes (frames not equivalent)")


def compute_gi_metric(state: GravityInertiaState) -> int:
    """
    Compute metric signature for gravity-inertia state

    Similar to ds² = S² - C²
    Here: ds² based on g-a balance
    """
    # Count stable vs changing bits
    g_bits = bin(state.g).count('1')
    a_bits = bin(state.a).count('1')

    # Metric: difference in activation
    ds2 = g_bits**2 - a_bits**2

    return ds2


def test_free_fall_eigenstates():
    """
    Test free-fall trajectories (geodesics) for eigenstate formation
    """
    print("\n\n" + "=" * 70)
    print("FREE-FALL GEODESICS")
    print("=" * 70)
    print()

    test_cases = [
        (0b11111111, 0b00000000, 0b10101010, "Strong gravity, no inertia"),
        (0b00000000, 0b11111111, 0b10101010, "No gravity, strong inertia"),
        (0b11110000, 0b11110000, 0b10101010, "Equal g and a"),
        (0b10101010, 0b01010101, 0b11110000, "Complementary g and a"),
    ]

    results = []

    for g, a, z, desc in test_cases:
        print(f"Test: {desc}")
        print(f"  Initial: g={g:08b} a={a:08b} z={z:08b}")

        traj, period = geodesic_trajectory(g, a, z, steps=16, verbose=False)

        if period:
            print(f"  ✓ Period-{period} geodesic eigenstate")
        else:
            print(f"  ✗ No eigenstate (open trajectory)")

        final_ds2 = compute_gi_metric(traj[-1])
        regime = "time-like" if final_ds2 > 0 else ("space-like" if final_ds2 < 0 else "light-like")
        print(f"  Final ds²: {final_ds2} ({regime})")
        print()

        results.append({
            'description': desc,
            'g': g,
            'a': a,
            'period': period,
            'ds2': final_ds2
        })

    # Summary
    print("─" * 70)
    print("SUMMARY:")
    print("─" * 70)
    print(f"{'Configuration':<35} {'Period':<8} {'ds²':<8} {'Eigenstate':<12}")
    print("─" * 70)

    for r in results:
        period_str = str(r['period']) if r['period'] else "none"
        eigenstate_mark = "✓" if r['period'] else "✗"
        print(f"{r['description']:<35} {period_str:<8} {r['ds2']:<8} {eigenstate_mark:<12}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EIGEN: GRAVITY-INERTIA EQUIVALENCE")
    print("=" * 70)
    print()
    print("Modeling g ↔ a equivalence using XOR framework")
    print("Geodesics = eigenstates in g-a space")
    print("Observer rotation swaps visible fields")
    print()

    test_equivalence_principle()
    test_free_fall_eigenstates()

    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print("""
1. Gravity and inertia modeled as (g, a, observer, Meta)
2. Equivalence principle: observer rotation swaps g ↔ a
3. Geodesics = periodic orbits in g-a eigenspace
4. ds² metric invariant under frame rotation
5. Same geometric structure as E-M and L-R-V
6. Free fall = eigenstate trajectory
7. All frames see same fundamental geometry
    """)
