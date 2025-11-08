#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eigen: Discrete Tokenization via XOR Cascades

Maps words to (L,R,V,M) bit patterns where:
- Each token is a discrete eigenstate
- Text sequences are XOR cascade trajectories
- Understanding emerges from cycle detection
- Time/space both emerge from oscillation patterns

This implements the discrete geometry from Eigen-TM VM.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import hashlib

# Constants for discrete geometry
BYTE_MAX = 256  # Maximum value for a single byte (0-255)
BITS_PER_BYTE = 8  # Number of bits in a byte
NUM_COMPONENTS = 4  # Number of components in state (L, R, V, M)
TOTAL_BITS = NUM_COMPONENTS * BITS_PER_BYTE  # Total bits in state (32)
PHASE_SECTORS = 8  # Number of 45° phase sectors (8 × 45° = 360°)
MAX_PERIOD = 8  # Maximum period to check for cycles (matches PHASE_SECTORS)


@dataclass
class DiscreteToken:
    """
    Discrete token as (L,R,V,M) bit pattern

    Attributes
    ----------
    L : int (0-255)
        Lexical byte
    R : int (0-255)
        Relational byte
    V : int (0-255)
        Value byte
    M : int (0-255)
        Meta byte (M = L ⊕ R ⊕ V)
    word : str
        Original word
    """

    L: int
    R: int
    V: int
    M: int
    word: str

    def __repr__(self):
        return f"Token({self.word}: L={self.L:08b} R={self.R:08b} V={self.V:08b} M={self.M:08b})"

    def as_tuple(self):
        """Return as (L,R,V,M) tuple"""
        return (self.L, self.R, self.V, self.M)


def hash_to_byte(text: str, seed: int = 0) -> int:
    """
    Hash text to single byte (0-255)

    Uses SHA256 with seed for deterministic but distributed mapping

    Parameters
    ----------
    text : str
        Input text
    seed : int
        Seed for hash variation

    Returns
    -------
    byte : int
        Value in range 0-255
    """
    combined = f"{text}:{seed}"
    hash_obj = hashlib.sha256(combined.encode())
    hash_bytes = hash_obj.digest()
    return hash_bytes[0]  # First byte


def tokenize_word(word: str) -> DiscreteToken:
    """
    Map word to discrete (L,R,V,M) eigenstate

    Each word gets unique oscillation pattern based on:
    - L: Lexical hash (subject/agent dimension)
    - R: Relational hash (verb/transformation dimension)
    - V: Value hash (object/patient dimension)
    - M: Meta parity (M = L ⊕ R ⊕ V)

    Parameters
    ----------
    word : str
        Input word

    Returns
    -------
    token : DiscreteToken
        Discrete eigenstate with (L,R,V,M) bit pattern

    Examples
    --------
    >>> token = tokenize_word("cat")
    >>> print(f"cat: L={token.L:08b} R={token.R:08b} V={token.V:08b}")
    >>> print(f"Meta: {token.M:08b}")
    """
    word_lower = word.lower().strip()

    L = hash_to_byte(word_lower, seed=1)
    R = hash_to_byte(word_lower, seed=2)
    V = hash_to_byte(word_lower, seed=3)
    M = L ^ R ^ V  # XOR parity

    return DiscreteToken(L=L, R=R, V=V, M=M, word=word)


def xor_states(
    state1: Tuple[int, int, int, int], state2: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    """
    XOR two states component-wise

    Parameters
    ----------
    state1, state2 : tuple of (L,R,V,M)
        States to XOR

    Returns
    -------
    result : tuple of (L,R,V,M)
        XORed state
    """
    return (
        state1[0] ^ state2[0],  # L
        state1[1] ^ state2[1],  # R
        state1[2] ^ state2[2],  # V
        state1[3] ^ state2[3],  # M
    )


def compute_hamming_distance(
    state1: Tuple[int, int, int, int], state2: Tuple[int, int, int, int]
) -> int:
    """
    Compute total Hamming distance between states

    Counts total bit flips across all four components

    Parameters
    ----------
    state1, state2 : tuple of (L,R,V,M)
        States to compare

    Returns
    -------
    distance : int
        Total number of bit differences (0-32)
    """
    total = 0
    for s1, s2 in zip(state1, state2):
        xor = s1 ^ s2
        total += bin(xor).count("1")
    return total


def compute_change_stability(
    state_prev: Tuple[int, int, int, int], state_curr: Tuple[int, int, int, int]
) -> Tuple[int, int, int]:
    """
    Compute (C, S, ds²) metrics for state transition

    Parameters
    ----------
    state_prev, state_curr : tuple of (L,R,V,M)
        Previous and current states

    Returns
    -------
    C : int
        Number of bits that changed
    S : int
        Number of bits that stayed stable
    ds2 : int
        Metric signature S² - C²
    """
    C = compute_hamming_distance(state_prev, state_curr)
    S = TOTAL_BITS - C  # Total bits in state (4 bytes × 8 bits each)
    ds2 = S * S - C * C

    return C, S, ds2


def detect_cycle_in_trajectory(
    trajectory: List[Tuple[int, int, int, int]], threshold: int = 0
) -> Optional[int]:
    """
    Detect period-k cycle in state trajectory

    Looks for exact state repetition (eigenstate orbit)

    Parameters
    ----------
    trajectory : list of (L,R,V,M) tuples
        State sequence
    threshold : int
        Maximum Hamming distance for "same" state (default: 0 = exact)

    Returns
    -------
    period : int or None
        Period of cycle if detected, None otherwise

    Examples
    --------
    >>> traj = [state_A, state_B, state_A, state_B, state_A]
    >>> period = detect_cycle_in_trajectory(traj)
    >>> print(f"Period-{period} orbit detected")
    """
    if len(trajectory) < 4:
        return None

    # Check for periods 2 through min(MAX_PERIOD, len//2)
    for period in range(2, min(MAX_PERIOD + 1, len(trajectory) // 2 + 1)):
        is_periodic = True

        for offset in range(period):
            # Check if states repeat at period intervals
            idx_curr = len(trajectory) - 1 - offset
            idx_prev = idx_curr - period

            if idx_prev < 0:
                is_periodic = False
                break

            distance = compute_hamming_distance(
                trajectory[idx_curr], trajectory[idx_prev]
            )

            if distance > threshold:
                is_periodic = False
                break

        if is_periodic:
            return period

    return None


def process_sentence_discrete(words: List[str], verbose: bool = False) -> Dict:
    """
    Process sentence as XOR cascade through discrete eigenspace

    Each word is tokenized to (L,R,V,M) and XORed into running state.
    The trajectory is analyzed for:
    - Cycle detection (eigenstate)
    - Change/stability metrics
    - Time/space oscillation patterns

    Parameters
    ----------
    words : list of str
        Word sequence
    verbose : bool
        Print step-by-step details

    Returns
    -------
    result : dict
        Contains:
        - trajectory: list of states
        - tokens: list of DiscreteToken
        - period: cycle period (or None)
        - C_history: change counts
        - S_history: stability counts
        - ds2_history: metric signatures
        - eigenstate: bool
        - time_coord: temporal phase
        - space_coord: spatial pattern

    Examples
    --------
    >>> result = process_sentence_discrete(["the", "cat", "sat"])
    >>> if result['eigenstate']:
    ...     print(f"Eigenstate with period {result['period']}")
    """
    # Initial state: all zeros
    state = (0, 0, 0, 0)
    trajectory = [state]
    tokens = []
    C_history = []
    S_history = []
    ds2_history = []
    regime_history = []

    if verbose:
        print(
            f"Initial state: L={state[0]:08b} R={state[1]:08b} V={state[2]:08b} M={state[3]:08b}"
        )
        print()

    for i, word in enumerate(words):
        # Tokenize word
        token = tokenize_word(word)
        tokens.append(token)

        # XOR into state
        prev_state = state
        state = xor_states(state, token.as_tuple())
        trajectory.append(state)

        # Compute metrics
        C, S, ds2 = compute_change_stability(prev_state, state)
        C_history.append(C)
        S_history.append(S)
        ds2_history.append(ds2)

        # Classify regime from invariant ds² = S² - C²
        # This is the geometric structure: time-like, space-like, or light-like
        if ds2 > 0:
            regime = "time-like"  # S > C: More stability than change
        elif ds2 < 0:
            regime = "space-like"  # C > S: More change than stability
        else:
            regime = "light-like"  # C = S: Balanced (transition state)
        regime_history.append(regime)

        if verbose:
            print(f"Word {i+1}: '{word}'")
            print(
                f"  Token: L={token.L:08b} R={token.R:08b} V={token.V:08b} M={token.M:08b}"
            )
            print(
                f"  State: L={state[0]:08b} R={state[1]:08b} V={state[2]:08b} M={state[3]:08b}"
            )
            print(f"  C={C}, S={S}, ds²={ds2}")
            print(f"  Regime: {regime}")
            print()

    # Detect cycle
    period = detect_cycle_in_trajectory(trajectory)
    eigenstate = period is not None

    # Compute time coordinate (phase in 45° sectors)
    # Based on which octant the final state is in
    time_coord = compute_temporal_phase(state)

    # Compute space coordinate (which bits oscillate most)
    space_coord = compute_spatial_pattern(trajectory)

    return {
        "trajectory": trajectory,
        "tokens": tokens,
        "period": period,
        "C_history": C_history,
        "S_history": S_history,
        "ds2_history": ds2_history,
        "regime_history": regime_history,  # Geometric classification from invariant
        "eigenstate": eigenstate,
        "time_coord": time_coord,
        "space_coord": space_coord,
        "final_state": state,
    }


def compute_temporal_phase(state: Tuple[int, int, int, int]) -> int:
    """
    Compute temporal coordinate (which 45° sector)

    Maps state to one of PHASE_SECTORS phases based on M value
    (since 8 × 45° = 360° closure)

    Parameters
    ----------
    state : tuple of (L,R,V,M)
        Current state

    Returns
    -------
    phase : int
        Phase sector (0-7)
    """
    M = state[3]
    # Map M byte (0-255) to phase sector (0-7)
    phase = (M * PHASE_SECTORS) // BYTE_MAX
    return phase


def compute_spatial_pattern(trajectory: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """
    Compute spatial oscillation pattern

    For each bit position across (L,R,V,M), count how often it flips.
    High flip rate = spatial oscillation

    Parameters
    ----------
    trajectory : list of states
        State sequence

    Returns
    -------
    pattern : np.ndarray of shape (TOTAL_BITS,)
        Flip count for each bit (NUM_COMPONENTS bytes × BITS_PER_BYTE)
    """
    if len(trajectory) < 2:
        return np.zeros(TOTAL_BITS)

    flip_counts = np.zeros(TOTAL_BITS)

    for i in range(1, len(trajectory)):
        prev = trajectory[i - 1]
        curr = trajectory[i]

        # Check each component
        for component_idx, (p, c) in enumerate(zip(prev, curr)):
            xor = p ^ c
            # Check each bit in the byte
            for bit_idx in range(BITS_PER_BYTE):
                if xor & (1 << bit_idx):
                    flip_counts[component_idx * BITS_PER_BYTE + bit_idx] += 1

    return flip_counts


def analyze_sentence(sentence: str, verbose: bool = True) -> Dict:
    """
    Full analysis of sentence as discrete eigenstate trajectory

    Parameters
    ----------
    sentence : str
        Input sentence
    verbose : bool
        Print detailed output

    Returns
    -------
    result : dict
        Analysis results
    """
    words = sentence.lower().split()

    if verbose:
        print("=" * 70)
        print(f"DISCRETE TOKENIZATION: '{sentence}'")
        print("=" * 70)
        print()

    result = process_sentence_discrete(words, verbose=verbose)

    if verbose:
        print("─" * 70)
        print("ANALYSIS:")
        print("─" * 70)
        print(f"Words: {len(words)}")
        print(f"Trajectory length: {len(result['trajectory'])}")

        if result["eigenstate"]:
            print(f"✓ Eigenstate detected: period-{result['period']} orbit")
        else:
            print("✗ No eigenstate (trajectory doesn't close)")

        print(f"\nFinal state:")
        final = result["final_state"]
        print(f"  L = {final[0]:08b} ({final[0]:3d})")
        print(f"  R = {final[1]:08b} ({final[1]:3d})")
        print(f"  V = {final[2]:08b} ({final[2]:3d})")
        print(f"  M = {final[3]:08b} ({final[3]:3d})")

        print(f"\nTemporal phase: sector {result['time_coord']}/{PHASE_SECTORS}")
        print(f"  (45° × {result['time_coord']} = {result['time_coord'] * 45}°)")

        # Show which bits oscillate most
        space = result["space_coord"]
        top_bits = np.argsort(space)[-5:][::-1]  # Top 5
        print(f"\nSpatial oscillation (top 5 bits):")
        for bit_idx in top_bits:
            component = bit_idx // BITS_PER_BYTE
            bit_pos = bit_idx % BITS_PER_BYTE
            component_name = ["L", "R", "V", "M"][component]
            print(f"  {component_name}[{bit_pos}]: {int(space[bit_idx])} flips")

        # Summary metrics
        if result["ds2_history"]:
            avg_ds2 = np.mean(result["ds2_history"])
            final_ds2 = result["ds2_history"][-1]
            print(f"\nMetric ds²:")
            print(f"  Average: {avg_ds2:.1f}")
            print(f"  Final: {final_ds2}")
            print(f"  Final regime: {result['regime_history'][-1]}")

        # Show regime trajectory (the geometric structure)
        if result["regime_history"]:
            print(f"\nRegime trajectory (from invariant ds² = S² - C²):")
            print(f"  {' → '.join(result['regime_history'])}")

            # Count regime types
            time_like = result["regime_history"].count("time-like")
            space_like = result["regime_history"].count("space-like")
            light_like = result["regime_history"].count("light-like")
            total = len(result["regime_history"])

            print(f"\nRegime distribution:")
            print(
                f"  Time-like (S > C, stable):  {time_like}/{total} ({100*time_like/total:.0f}%)"
            )
            print(
                f"  Space-like (C > S, change): {space_like}/{total} ({100*space_like/total:.0f}%)"
            )
            if light_like > 0:
                print(
                    f"  Light-like (C = S, transition): {light_like}/{total} ({100*light_like/total:.0f}%)"
                )

    return result


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EIGEN DISCRETE TOKENIZATION - TEST")
    print("=" * 70)
    print()

    # Test 1: Simple sentence
    analyze_sentence("the cat sat")

    print("\n\n")

    # Test 2: Different sentence
    analyze_sentence("wind bends tree")

    print("\n\n")

    # Test 3: Repeated word (should show pattern)
    analyze_sentence("cat cat cat")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print(
        """
1. Each word maps to unique (L,R,V,M) bit pattern
2. XOR cascade creates trajectory through eigenspace
3. Cycle detection finds eigenstate orbits
4. Time = temporal phase (which 45° sector)
5. Space = which bits oscillate
6. ds² = S² - C² tracks regime
7. Understanding = trajectory closure
    """
    )
