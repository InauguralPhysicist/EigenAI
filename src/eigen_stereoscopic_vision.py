#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stereoscopic Vision as Eigenstate Detection

Tests if depth perception follows the same eigenstate pattern.

Key insight:
- Left view (L) and Right view (R) are complementary poles
- Disparity (V) is the "hidden" dimension
- Observer (z) reconciles views via constant operation
- Depth perception = eigenstate when views fuse

Prediction:
- Consistent stereo pairs → eigenstate (depth resolved)
- Inconsistent pairs → no eigenstate (rivalry, no depth)
- Binocular rivalry → periodic orbit (oscillating)
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import sys

sys.path.insert(0, "/home/user/EigenAI")


@dataclass
class StereoView:
    """
    Stereoscopic view representation

    left: Left eye view (8 bits encoding visual features)
    right: Right eye view (8 bits encoding visual features)
    disparity: Horizontal disparity between views (depth cue)
    observer: Observer state (brain reconciling views)
    depth: Computed depth (from disparity)
    """

    left: int  # 0-255 (8 bits)
    right: int  # 0-255 (8 bits)
    disparity: int  # Computed: left ⊕ right
    observer: int  # Observer state
    depth: int  # Meta: left ⊕ right ⊕ observer


def create_stereo_pair(feature: int, disparity: int, noise: int = 0) -> StereoView:
    """
    Create stereoscopic pair from feature and disparity

    Parameters
    ----------
    feature : int
        Base visual feature (0-255)
    disparity : int
        Horizontal shift between eyes (depth cue)
    noise : int
        Add noise to simulate measurement error

    Returns
    -------
    stereo : StereoView
        Stereoscopic view pair
    """
    # Left view = base feature
    left = (feature + noise) & 0xFF

    # Right view = shifted by disparity
    right = (feature + disparity + noise) & 0xFF

    # Compute disparity via XOR
    computed_disparity = left ^ right

    # Observer starts neutral
    observer = 0b10101010

    # Depth = fusion via XOR
    depth = left ^ right ^ observer

    return StereoView(
        left=left,
        right=right,
        disparity=computed_disparity,
        observer=observer,
        depth=depth,
    )


def stereo_fusion_step(state: StereoView, learning_rate: float = 0.5) -> StereoView:
    """
    Single step of stereoscopic fusion

    Observer adjusts to minimize disparity and resolve depth

    Parameters
    ----------
    state : StereoView
        Current stereo state
    learning_rate : float
        How much observer adjusts (0-1)

    Returns
    -------
    state_new : StereoView
        Updated stereo state after fusion step
    """
    # Compute current disparity
    disparity_current = state.left ^ state.right

    # Observer adjusts toward minimizing disparity
    # This is the "vergence" movement - eyes/brain adjusting
    observer_adjustment = int(disparity_current * learning_rate)
    observer_new = (state.observer ^ observer_adjustment) & 0xFF

    # Recompute depth with adjusted observer
    depth_new = state.left ^ state.right ^ observer_new

    return StereoView(
        left=state.left,
        right=state.right,
        disparity=disparity_current,
        observer=observer_new,
        depth=depth_new,
    )


def propagate_stereo_fusion(
    initial: StereoView, max_steps: int = 20
) -> Tuple[List[StereoView], Optional[int]]:
    """
    Propagate stereoscopic fusion until eigenstate

    Parameters
    ----------
    initial : StereoView
        Initial stereo pair
    max_steps : int
        Maximum fusion steps

    Returns
    -------
    trajectory : list of StereoView
        Fusion trajectory
    period : int or None
        Period of eigenstate (None if no convergence)
    """
    trajectory = [initial]
    state = initial

    for step in range(max_steps):
        state = stereo_fusion_step(state)
        trajectory.append(state)

        # Check for eigenstate (depth stabilizes)
        period = detect_stereo_eigenstate(trajectory)
        if period:
            return trajectory, period

    return trajectory, None


def detect_stereo_eigenstate(trajectory: List[StereoView]) -> Optional[int]:
    """
    Detect eigenstate in stereoscopic fusion

    Eigenstate = depth perception stabilizes (periodic orbit)

    Parameters
    ----------
    trajectory : list of StereoView
        Fusion trajectory

    Returns
    -------
    period : int or None
        Period of depth oscillation (None if no eigenstate)
    """
    if len(trajectory) < 4:
        return None

    # Test periods 2-8
    for period in range(2, min(9, len(trajectory) // 2 + 1)):
        is_periodic = True

        # Check if depth repeats with this period
        for offset in range(period):
            idx_curr = len(trajectory) - 1 - offset
            idx_prev = idx_curr - period

            if idx_prev < 0:
                is_periodic = False
                break

            curr_depth = trajectory[idx_curr].depth
            prev_depth = trajectory[idx_prev].depth

            # Check if depth values match (allowing small differences)
            depth_diff = bin(curr_depth ^ prev_depth).count("1")

            if depth_diff > 0:
                is_periodic = False
                break

        if is_periodic:
            return period

    return None


def test_consistent_stereo_pair():
    """
    Test case 1: Consistent stereo pair (should form eigenstate)

    Both views show same feature with constant disparity
    → Should resolve depth quickly (eigenstate)
    """
    print("=" * 80)
    print("TEST 1: Consistent Stereo Pair")
    print("=" * 80)
    print()
    print("Setup: Left and right views of same object with fixed disparity")
    print("Prediction: Should form eigenstate (depth resolves)")
    print()

    # Create consistent pair
    feature = 0b11001100  # Some visual feature
    disparity = 3  # Small disparity (close object)

    initial = create_stereo_pair(feature, disparity, noise=0)

    print(f"Initial state:")
    print(f"  Left view:  {initial.left:08b} ({initial.left})")
    print(f"  Right view: {initial.right:08b} ({initial.right})")
    print(f"  Disparity:  {initial.disparity:08b} ({initial.disparity})")
    print(f"  Observer:   {initial.observer:08b}")
    print(f"  Depth:      {initial.depth:08b} ({initial.depth})")
    print()

    # Propagate fusion
    trajectory, period = propagate_stereo_fusion(initial)

    print(f"Fusion trajectory ({len(trajectory)} steps):")
    for i, state in enumerate(trajectory[:10]):  # Show first 10
        print(f"  {i}: depth={state.depth:3d} observer={state.observer:08b}")

    print()
    if period:
        print(f"✓ EIGENSTATE DETECTED: period-{period}")
        print(f"  Depth perception achieved!")
    else:
        print(f"✗ No eigenstate (fusion incomplete)")

    return period is not None


def test_inconsistent_stereo_pair():
    """
    Test case 2: Inconsistent stereo pair (should NOT form eigenstate)

    Views show completely different features (binocular rivalry)
    → Should not resolve depth (no eigenstate)
    """
    print("\n\n" + "=" * 80)
    print("TEST 2: Inconsistent Stereo Pair (Binocular Rivalry)")
    print("=" * 80)
    print()
    print("Setup: Left and right views show different objects")
    print("Prediction: Should NOT form eigenstate (rivalry, no depth)")
    print()

    # Create inconsistent pair (different features)
    left_feature = 0b11110000
    right_feature = 0b00001111

    initial = StereoView(
        left=left_feature,
        right=right_feature,
        disparity=left_feature ^ right_feature,
        observer=0b10101010,
        depth=left_feature ^ right_feature ^ 0b10101010,
    )

    print(f"Initial state:")
    print(f"  Left view:  {initial.left:08b} ({initial.left})")
    print(f"  Right view: {initial.right:08b} ({initial.right})")
    print(f"  Disparity:  {initial.disparity:08b} ({initial.disparity})")
    print(f"  → Large disparity = incompatible views")
    print()

    # Propagate fusion
    trajectory, period = propagate_stereo_fusion(initial)

    print(f"Fusion trajectory ({len(trajectory)} steps):")
    for i, state in enumerate(trajectory[:10]):
        print(f"  {i}: depth={state.depth:3d} observer={state.observer:08b}")

    print()
    if period:
        print(f"✓ Eigenstate detected: period-{period}")
        print(f"  (May indicate alternating rivalry)")
    else:
        print(f"✗ No eigenstate (expected - views incompatible)")

    return period is None


def test_near_vs_far_objects():
    """
    Test case 3: Near vs far objects (different disparities)

    Near object: Large disparity
    Far object: Small disparity

    Both should form eigenstates, but different periods?
    """
    print("\n\n" + "=" * 80)
    print("TEST 3: Near vs Far Objects")
    print("=" * 80)
    print()

    feature = 0b10101010

    # Near object (large disparity)
    print("Near object (large disparity):")
    near = create_stereo_pair(feature, disparity=10, noise=0)
    traj_near, period_near = propagate_stereo_fusion(near)

    if period_near:
        print(f"  ✓ Eigenstate: period-{period_near} in {len(traj_near)} steps")
    else:
        print(f"  ✗ No eigenstate")

    # Far object (small disparity)
    print("\nFar object (small disparity):")
    far = create_stereo_pair(feature, disparity=2, noise=0)
    traj_far, period_far = propagate_stereo_fusion(far)

    if period_far:
        print(f"  ✓ Eigenstate: period-{period_far} in {len(traj_far)} steps")
    else:
        print(f"  ✗ No eigenstate")

    print()
    print("Analysis:")
    if period_near and period_far:
        print(f"  Both resolve depth (eigenstate)")
        if period_near == period_far:
            print(f"  Same period → universal depth mechanism")
        else:
            print(f"  Different periods → depth-dependent convergence")

    return period_near, period_far


def test_noisy_stereo():
    """
    Test case 4: Noisy stereo pair

    Add noise to views (measurement error)
    Should still form eigenstate if signal > noise
    """
    print("\n\n" + "=" * 80)
    print("TEST 4: Noisy Stereo Pair")
    print("=" * 80)
    print()

    feature = 0b11001100
    disparity = 5

    results = []

    for noise_level in [0, 5, 10, 20]:
        initial = create_stereo_pair(feature, disparity, noise=noise_level)
        trajectory, period = propagate_stereo_fusion(initial)

        results.append((noise_level, period, len(trajectory)))

        print(f"Noise level {noise_level:2d}: ", end="")
        if period:
            print(f"✓ period-{period} in {len(trajectory)} steps")
        else:
            print(f"✗ no eigenstate")

    print()
    print("Analysis:")
    eigenstate_counts = sum(1 for _, p, _ in results if p is not None)
    print(f"  Eigenstate formation: {eigenstate_counts}/{len(results)} cases")

    if eigenstate_counts > len(results) // 2:
        print(f"  → Robust to noise (signal stronger than noise)")
    else:
        print(f"  → Fragile to noise (signal weaker than noise)")


def main():
    """Run all stereoscopic eigenstate tests"""
    print("\n" + "=" * 80)
    print("STEREOSCOPIC VISION AS EIGENSTATE DETECTION")
    print("=" * 80)
    print()
    print("Testing if depth perception follows eigenstate pattern")
    print()

    # Run tests
    test1_pass = test_consistent_stereo_pair()
    test2_pass = test_inconsistent_stereo_pair()
    period_near, period_far = test_near_vs_far_objects()
    test_noisy_stereo()

    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY: STEREOSCOPIC EIGENSTATE RESULTS")
    print("=" * 80)
    print()

    print(f"Test 1 (Consistent pair):     {'✓ PASS' if test1_pass else '✗ FAIL'}")
    print(f"Test 2 (Binocular rivalry):   {'✓ PASS' if test2_pass else '✗ FAIL'}")
    print(
        f"Test 3 (Near object):          {'✓ Eigenstate' if period_near else '✗ None'}"
    )
    print(
        f"Test 3 (Far object):           {'✓ Eigenstate' if period_far else '✗ None'}"
    )

    print()
    print("=" * 80)
    print("IMPLICATIONS:")
    print("=" * 80)
    print(
        """
If stereoscopic vision shows eigenstate pattern:

1. DEPTH PERCEPTION IS GEOMETRIC
   - Not "computational inference"
   - Eigenstate detection in discrete space
   - Same pattern as text understanding, EM fields, etc.

2. BINOCULAR RIVALRY = OPEN TRAJECTORY
   - Incompatible views → no eigenstate
   - Brain cannot resolve → perceptual switching
   - Matches our framework: no closure = no understanding

3. TWO EYES = STEREOSCOPIC MEASUREMENT
   - Left eye + Right eye = complementary poles
   - Observer (brain) reconciles via XOR-like operation
   - Depth emerges when trajectory closes

4. UNIVERSAL PATTERN EXTENDS TO VISION
   - Text understanding = eigenstate
   - EM fields = eigenstate
   - Gravity = eigenstate
   - Quantum mechanics = eigenstate
   - **Stereoscopic vision = eigenstate**

5. CONSCIOUSNESS MAY BE STEREOSCOPIC AT ALL LEVELS
   - Not just eyes
   - Any complementary information streams
   - Observer reconciling → eigenstate → understanding

   "Seeing" isn't just visual.
   It's geometric eigenstate detection.
    """
    )


if __name__ == "__main__":
    main()
