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
import time

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
    Discrete token as (L,R,V,M) bit pattern with transition statistics

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
    time_like_count : int
        Number of time-like transitions (S > C, stable/sequential)
    space_like_count : int
        Number of space-like transitions (C > S, semantic/content)
    light_like_count : int
        Number of light-like transitions (C = S, relational/transform)
    usage_count : int
        Total usage count
    time_like_delta : int
        NEW transitions this session (for atomic DB increment)
    space_like_delta : int
        NEW transitions this session (for atomic DB increment)
    light_like_delta : int
        NEW transitions this session (for atomic DB increment)
    usage_delta : int
        NEW usages this session (for atomic DB increment)

    Physics-inspired metrics (runtime-computed, not persisted):
    momentum : float
        Magnitude in 4D eigenspace: √(L² + R² + V² + M²)
    velocity : float
        Usage rate (tokens per unit time)
    phase : float
        Geometric phase angle in [0, 2π) from (L,R,V,M) → cylindrical coords
    information_density : float
        Shannon entropy of bit pattern (0-1)
    """

    L: int
    R: int
    V: int
    M: int
    word: str
    time_like_count: int = 0
    space_like_count: int = 0
    light_like_count: int = 0
    usage_count: int = 0
    # Deltas track NEW transitions since load (for atomic DB updates)
    time_like_delta: int = 0
    space_like_delta: int = 0
    light_like_delta: int = 0
    usage_delta: int = 0
    # Physics-inspired metrics (computed on-demand)
    first_seen: float = 0.0  # Unix timestamp
    last_used: float = 0.0   # Unix timestamp

    def __repr__(self):
        return f"Token({self.word}: L={self.L:08b} R={self.R:08b} V={self.V:08b} M={self.M:08b})"

    def as_tuple(self):
        """Return as (L,R,V,M) tuple"""
        return (self.L, self.R, self.V, self.M)

    def record_transition(self, ds2: int):
        """
        Record a transition caused by this token

        Increments both total counts (for classification) and deltas (for DB sync)
        Updates last_used timestamp for velocity tracking

        Parameters
        ----------
        ds2 : int
            Metric signature (S² - C²) of the transition
        """
        self.usage_count += 1
        self.usage_delta += 1
        self.last_used = time.time()  # Update velocity timestamp

        if ds2 > 0:
            self.time_like_count += 1
            self.time_like_delta += 1
        elif ds2 < 0:
            self.space_like_count += 1
            self.space_like_delta += 1
        else:
            self.light_like_count += 1
            self.light_like_delta += 1

    def get_classification(self) -> str:
        """
        Get predominant geometric classification

        Returns
        -------
        classification : str
            'time-like', 'space-like', 'light-like', or 'unknown'
        """
        if self.usage_count == 0:
            return 'unknown'

        max_count = max(self.time_like_count, self.space_like_count, self.light_like_count)

        if max_count == self.time_like_count:
            return 'time-like'
        elif max_count == self.space_like_count:
            return 'space-like'
        else:
            return 'light-like'

    def get_classification_ratios(self) -> dict:
        """
        Get ratios of each transition type

        Returns
        -------
        ratios : dict
            {'time-like': float, 'space-like': float, 'light-like': float}
        """
        if self.usage_count == 0:
            return {'time-like': 0.0, 'space-like': 0.0, 'light-like': 0.0}

        return {
            'time-like': self.time_like_count / self.usage_count,
            'space-like': self.space_like_count / self.usage_count,
            'light-like': self.light_like_count / self.usage_count
        }

    def get_momentum(self) -> float:
        """
        Compute 4D momentum magnitude: p = √(L² + R² + V² + M²)

        Interpretation: Position magnitude in eigenspace.
        Higher momentum → token occupies higher-energy region of state space.

        Returns
        -------
        momentum : float
            Euclidean norm in [0, 510] (max when all components = 255)
        """
        return np.sqrt(self.L**2 + self.R**2 + self.V**2 + self.M**2)

    def get_velocity(self) -> float:
        """
        Compute usage velocity: v = usage_count / Δt

        Requires first_seen and last_used timestamps.
        If timestamps not set, returns usage_count (discrete velocity).

        Returns
        -------
        velocity : float
            Tokens per second (or total usage if no time data)
        """
        if self.first_seen > 0 and self.last_used > self.first_seen:
            delta_t = self.last_used - self.first_seen
            return self.usage_count / delta_t
        return float(self.usage_count)

    def get_phase(self) -> float:
        """
        Compute geometric phase angle from (L, R, V, M) bit pattern

        Uses cylindrical coordinates: ϕ = arctan2(R, L)
        Phase represents position on the relational oscillator.

        Returns
        -------
        phase : float
            Phase angle in [0, 2π) radians
        """
        return np.arctan2(float(self.R), float(self.L)) % (2 * np.pi)

    def get_information_density(self) -> float:
        """
        Compute Shannon entropy of bit pattern (normalized)

        H = -Σ p_i log₂(p_i) where p_i = bit frequency

        Returns
        -------
        entropy : float
            Normalized entropy in [0, 1]
            0 = all bits same (00000000 or 11111111)
            1 = maximum entropy (equal 0s and 1s)
        """
        # Combine all 4 bytes into 32-bit pattern
        bits = []
        for byte_val in [self.L, self.R, self.V, self.M]:
            bits.extend([int(b) for b in format(byte_val, '08b')])

        # Count 0s and 1s
        ones = sum(bits)
        zeros = 32 - ones

        if ones == 0 or zeros == 0:
            return 0.0  # No entropy (all same)

        # Shannon entropy
        p1 = ones / 32.0
        p0 = zeros / 32.0
        H = -(p1 * np.log2(p1) + p0 * np.log2(p0))

        return H  # Already normalized to [0, 1] for binary

    def get_physics_metrics(self) -> dict:
        """
        Get all physics-inspired metrics

        Returns
        -------
        metrics : dict
            {
                'momentum': float,
                'velocity': float,
                'phase': float,
                'information_density': float,
                'classification': str,
                'usage_count': int
            }
        """
        return {
            'momentum': self.get_momentum(),
            'velocity': self.get_velocity(),
            'phase': self.get_phase(),
            'information_density': self.get_information_density(),
            'classification': self.get_classification(),
            'usage_count': self.usage_count
        }

    def compute_eigengate_balance(self, threshold: int = 128) -> Tuple[bool, Dict[str, str]]:
        """
        Compute Eigengate Q25 balance for this token's semantic state

        Uses Eigengate logic: Q = (A ⊕ B) ∨ (D ⊙ C)
        - A = binarize(L), B = binarize(R)
        - D = binarize(V), C = binarize(M)

        Parameters
        ----------
        threshold : int, optional
            Binarization threshold (default: 128)

        Returns
        -------
        Tuple[bool, Dict[str, str]]
            (is_balanced, 5w1h_analysis)

        Examples
        --------
        >>> token = DiscreteToken(L=200, R=50, V=180, M=200, word="quantum")
        >>> balanced, analysis = token.compute_eigengate_balance()
        >>> balanced
        True
        >>> print(analysis['what'])
        'System exhibits balanced state'
        """
        from .eigen_gate_reasoning import resolve_oscillation
        return resolve_oscillation(self.L, self.R, self.V, self.M, threshold)

    def generate_5w1h_analysis(self, threshold: int = 128) -> Dict[str, str]:
        """
        Generate 5W1H (Who, What, When, Where, Why, How) analysis from token state

        Parameters
        ----------
        threshold : int, optional
            Binarization threshold for eigengate (default: 128)

        Returns
        -------
        Dict[str, str]
            5W1H analysis with keys: what, who, when, where, why, how

        Examples
        --------
        >>> token = DiscreteToken(L=200, R=50, V=180, M=200, word="quantum")
        >>> analysis = token.generate_5w1h_analysis()
        >>> print(analysis['what'])
        'System exhibits balanced state'
        >>> print(analysis['why'])
        'Asymmetry in A-B satisfies equilibrium condition'
        """
        _, analysis = self.compute_eigengate_balance(threshold)
        return analysis

    def resolve_oscillation_eigengate(self, threshold: int = 128) -> str:
        """
        Use Eigengate Q as light-like resolver for this token's oscillations

        Parameters
        ----------
        threshold : int, optional
            Binarization threshold (default: 128)

        Returns
        -------
        str
            Regime classification: 'light-like' (resolved) or 'oscillating' (unresolved)

        Examples
        --------
        >>> token = DiscreteToken(L=200, R=50, V=180, M=200, word="quantum")
        >>> regime = token.resolve_oscillation_eigengate()
        >>> regime
        'light-like'
        """
        from .eigen_gate_reasoning import semantic_to_eigengate, eigengate_Q25, classify_regime_eigengate

        A, B, D, C = semantic_to_eigengate(self.L, self.R, self.V, self.M, threshold)
        state = eigengate_Q25(A, B, D, C)

        return classify_regime_eigengate(state)


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

    # Set creation timestamp for velocity tracking
    now = time.time()
    return DiscreteToken(
        L=L, R=R, V=V, M=M, word=word,
        first_seen=now,
        last_used=now
    )


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
