#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Eigenstate Detection: Enabling Trajectory Closure in Text

Problem: Text with unique words never forms eigenstates (trajectory never repeats)
Solution: Quantize semantic space so similar states converge → trajectory closes

Key insight from physics:
- Photons: period-2 eigenstates (simple oscillation)
- Gravity: period-8 eigenstates (complex attractor)
- Text should show same patterns when semantically coherent

Approach:
1. Semantic transformation (L, R, V, M) from words
2. QUANTIZE coordinates to discrete bins (like XOR discretization)
3. Similar semantic states → same quantized state → trajectory closure
4. Detect eigenstates via quantized state repetition

This enables:
- Coherent paragraphs → form eigenstates (like photons)
- Scrambled text → no eigenstates (chaotic trajectory)
- Simple grammar → period-2 (like photons)
- Complex grammar → period-8 (like gravity)
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from collections import defaultdict

from eigen_semantic_transformer import (
    SemanticGeometricTransformer,
    SemanticState,
    compute_grammatical_score,
    compute_ds2_semantic
)


@dataclass
class QuantizedSemanticState:
    """
    Quantized semantic state for eigenstate detection

    Like discrete (L,R,V,M) but derived from semantic coordinates
    Quantization allows trajectory closure even with different words
    """
    L_quantized: int  # Quantized lexical coordinate (0-255)
    R_quantized: int  # Quantized relational coordinate (0-255)
    V_quantized: int  # Quantized value coordinate (0-255)
    M_quantized: int  # Quantized meta/time coordinate (0-255)

    word: str
    original_state: SemanticState

    def __hash__(self):
        """
        Hash based on L,R,V only (NOT M)

        M is emergent time (position-dependent), not part of state identity
        Like in physics: position in spacetime vs coordinates defining the state
        """
        return hash((self.L_quantized, self.R_quantized, self.V_quantized))

    def __eq__(self, other):
        """
        Equality based on L,R,V only (NOT M)

        Eigenstate = trajectory returns to same (L,R,V) coordinates
        M evolves along trajectory (time) but doesn't define the state
        """
        if not isinstance(other, QuantizedSemanticState):
            return False
        return (self.L_quantized == other.L_quantized and
                self.R_quantized == other.R_quantized and
                self.V_quantized == other.V_quantized)


def quantize_coordinate(value: float, bins: int = 16) -> int:
    """
    Quantize continuous coordinate to discrete bin

    Maps [-1, 1] → [0, bins-1]
    Similar to XOR discretization in original eigenstate detection

    Args:
        value: Continuous value (typically in range [-1, 1])
        bins: Number of discrete bins (default 16 for coarse quantization)
              Fewer bins → more likely for similar states to match

    Returns:
        Quantized integer in [0, bins-1]
    """
    # Normalize to [0, 1]
    normalized = (np.tanh(value) + 1.0) / 2.0

    # Map to [0, bins-1]
    quantized = int(normalized * (bins - 1))

    # Clamp to valid range
    return max(0, min(bins - 1, quantized))


def quantize_semantic_state(state: SemanticState) -> QuantizedSemanticState:
    """
    Quantize continuous semantic state to discrete representation

    This enables eigenstate detection by making similar states equal

    Uses semantic vector hash instead of mean to preserve diversity

    Args:
        state: Continuous semantic state from transformer

    Returns:
        Quantized state for eigenstate detection
    """
    # Use first 3 components + semantic vector hash for diversity
    # Don't just average - that washes out differences

    # Get semantic vector norm and direction
    sem_vec = state.semantic_vec
    vec_norm = np.linalg.norm(sem_vec)

    # Use first few components for L, R, V quantization
    # This preserves word-specific semantic differences
    L_val = np.dot(sem_vec[:10], np.ones(10)) / 10.0  # First 10 components
    R_val = np.dot(sem_vec[10:20], np.ones(10)) / 10.0  # Second 10
    V_val = np.dot(sem_vec[20:30], np.ones(10)) / 10.0  # Third 10
    M_val = state.M

    # Quantize to discrete bins (64 bins for moderate granularity)
    L_q = quantize_coordinate(L_val, bins=64)
    R_q = quantize_coordinate(R_val, bins=64)
    V_q = quantize_coordinate(V_val, bins=64)
    M_q = quantize_coordinate(M_val, bins=64)

    return QuantizedSemanticState(
        L_quantized=L_q,
        R_quantized=R_q,
        V_quantized=V_q,
        M_quantized=M_q,
        word="",  # Will be set by caller
        original_state=state
    )


def detect_eigenstate_quantized(trajectory: List[QuantizedSemanticState]) -> Tuple[Optional[int], List[int]]:
    """
    Detect eigenstate via quantized state repetition

    Like original eigenstate detection but using quantized semantic states

    Args:
        trajectory: List of quantized semantic states

    Returns:
        (period, first_occurrence_indices) or (None, []) if no eigenstate
    """
    if len(trajectory) < 2:
        return None, []

    # Track state occurrences
    state_positions = defaultdict(list)

    for i, state in enumerate(trajectory):
        state_hash = hash(state)
        state_positions[state_hash].append(i)

    # Check for repeated states (eigenstate = period-N)
    for state_hash, positions in state_positions.items():
        if len(positions) >= 2:
            # Calculate period (distance between repetitions)
            periods = [positions[i+1] - positions[i] for i in range(len(positions)-1)]

            # Check if period is consistent
            if len(set(periods)) == 1:
                period = periods[0]
                return period, positions

    return None, []


def process_text_with_eigenstates(text: str, transformer: SemanticGeometricTransformer) -> dict:
    """
    Process text and detect semantic eigenstates

    Args:
        text: Input text (can be sentence or paragraph)
        transformer: Semantic transformer instance

    Returns:
        Dictionary with trajectory, eigenstate info, and metrics
    """
    # Tokenize
    words = text.lower().split()

    if len(words) < 2:
        return {'words': words, 'eigenstate': None, 'period': None}

    # Get continuous trajectory
    continuous_trajectory, _ = transformer.process_sequence(words, verbose=False)

    # Quantize trajectory
    quantized_trajectory = []
    for i, state in enumerate(continuous_trajectory):
        q_state = quantize_semantic_state(state)
        q_state.word = words[i]
        quantized_trajectory.append(q_state)

    # Detect eigenstate
    period, positions = detect_eigenstate_quantized(quantized_trajectory)

    # Compute metrics
    coherence = transformer.compute_semantic_coherence(words)
    gram_score = compute_grammatical_score(words, transformer)
    coupling = 0.5 + 4.5 * gram_score

    ds2_values = compute_ds2_semantic(continuous_trajectory, grammatical_coupling=coupling)
    avg_ds2 = np.mean(ds2_values) if ds2_values else 0.0

    return {
        'words': words,
        'continuous_trajectory': continuous_trajectory,
        'quantized_trajectory': quantized_trajectory,
        'eigenstate': period is not None,
        'period': period,
        'repetition_positions': positions,
        'coherence': coherence,
        'gram_score': gram_score,
        'coupling': coupling,
        'avg_ds2': avg_ds2
    }


def test_eigenstate_formation():
    """
    Test if semantic eigenstates form in coherent text
    """
    print("=" * 80)
    print("SEMANTIC EIGENSTATE FORMATION TEST")
    print("=" * 80)
    print()
    print("Testing: Do coherent text sequences form eigenstates?")
    print()
    print("Hypothesis:")
    print("  - Coherent repeated text → eigenstate (like photons)")
    print("  - Coherent paragraphs → eigenstate (semantic attractor)")
    print("  - Scrambled text → no eigenstate (chaotic)")
    print()

    transformer = SemanticGeometricTransformer(embedding_dim=100)

    test_cases = [
        # Simple repetition (should form eigenstate)
        ("the cat sat the cat sat", "Repeated sentence (period expected)"),

        # Semantic similarity (should converge to eigenstate)
        ("the cat sat the dog ran", "Similar structure (convergence expected)"),

        # Coherent paragraph
        ("light travels fast light moves quickly", "Semantically similar (attractor)"),

        # Scrambled (should NOT form eigenstate)
        ("cat the sat dog ran the", "Scrambled (no eigenstate)"),

        # Single repetition
        ("photon photon photon photon", "Repeated word (strong eigenstate)"),

        # Complex coherent
        ("the cat sat on the mat the dog ran in the park", "Complex coherent"),
    ]

    results = []

    for text, description in test_cases:
        print(f"TEST: {description}")
        print(f"  Text: '{text}'")
        print()

        result = process_text_with_eigenstates(text, transformer)

        # Display quantized trajectory
        print("  Quantized trajectory:")
        for i, q_state in enumerate(result['quantized_trajectory']):
            marker = ""
            if result['eigenstate'] and i in result['repetition_positions']:
                marker = " ← REPEAT"
            print(f"    {i}: {q_state.word:10s} ({q_state.L_quantized:3d}, {q_state.R_quantized:3d}, "
                  f"{q_state.V_quantized:3d}, {q_state.M_quantized:3d}){marker}")
        print()

        # Eigenstate detection
        if result['eigenstate']:
            print(f"  ✓ EIGENSTATE DETECTED: period-{result['period']}")
            print(f"    Repetitions at positions: {result['repetition_positions']}")
        else:
            print("  ✗ NO EIGENSTATE")
        print()

        # Metrics
        print(f"  Coherence: {result['coherence']:.3f}")
        print(f"  Grammatical score: {result['gram_score']:.3f}")
        print(f"  Coupling: {result['coupling']:.2f}x")
        print(f"  Average ds²: {result['avg_ds2']:.4f}")
        print()
        print("-" * 80)
        print()

        results.append({
            'text': text,
            'description': description,
            'eigenstate': result['eigenstate'],
            'period': result['period'],
            'coherence': result['coherence'],
            'gram_score': result['gram_score']
        })

    return results


def test_eigenstate_vs_coherence():
    """
    Critical test: Does eigenstate formation correlate with coherence?
    """
    print("\n" + "=" * 80)
    print("EIGENSTATE vs COHERENCE CORRELATION")
    print("=" * 80)
    print()
    print("Question: Do eigenstates form preferentially in coherent text?")
    print()

    transformer = SemanticGeometricTransformer(embedding_dim=100)

    # Coherent cases
    coherent_texts = [
        "the cat sat the cat sat",
        "light travels fast light moves fast",
        "photon photon photon photon",
    ]

    # Incoherent cases
    incoherent_texts = [
        "cat the sat the cat sat",
        "fast travels light moves light fast",
        "the photon cat light sat photon",
    ]

    print("COHERENT TEXT:")
    print("-" * 80)
    coherent_results = []
    for text in coherent_texts:
        result = process_text_with_eigenstates(text, transformer)
        eigenstate_str = f"✓ period-{result['period']}" if result['eigenstate'] else "✗ none"
        print(f"'{text}'")
        print(f"  Eigenstate: {eigenstate_str}")
        print(f"  Coherence: {result['coherence']:.3f}")
        print()
        coherent_results.append(result)

    print("\nINCOHERENT TEXT:")
    print("-" * 80)
    incoherent_results = []
    for text in incoherent_texts:
        result = process_text_with_eigenstates(text, transformer)
        eigenstate_str = f"✓ period-{result['period']}" if result['eigenstate'] else "✗ none"
        print(f"'{text}'")
        print(f"  Eigenstate: {eigenstate_str}")
        print(f"  Coherence: {result['coherence']:.3f}")
        print()
        incoherent_results.append(result)

    # Analysis
    print("=" * 80)
    print("ANALYSIS:")
    print("=" * 80)
    print()

    coherent_eigenstate_count = sum(1 for r in coherent_results if r['eigenstate'])
    incoherent_eigenstate_count = sum(1 for r in incoherent_results if r['eigenstate'])

    coherent_avg_coherence = np.mean([r['coherence'] for r in coherent_results])
    incoherent_avg_coherence = np.mean([r['coherence'] for r in incoherent_results])

    # NEW: Analyze period length
    coherent_periods = [r['period'] for r in coherent_results if r['period'] is not None]
    incoherent_periods = [r['period'] for r in incoherent_results if r['period'] is not None]

    coherent_avg_period = np.mean(coherent_periods) if coherent_periods else 0
    incoherent_avg_period = np.mean(incoherent_periods) if incoherent_periods else 0

    print(f"Coherent text:")
    print(f"  {coherent_eigenstate_count}/{len(coherent_results)} have eigenstates")
    print(f"  Average coherence = {coherent_avg_coherence:.3f}")
    print(f"  Average period = {coherent_avg_period:.1f}")
    print()
    print(f"Incoherent text:")
    print(f"  {incoherent_eigenstate_count}/{len(incoherent_results)} have eigenstates")
    print(f"  Average coherence = {incoherent_avg_coherence:.3f}")
    print(f"  Average period = {incoherent_avg_period:.1f}")
    print()

    # Both form eigenstates, but check period length
    if coherent_avg_period < incoherent_avg_period - 0.5:
        print("✓ PERIOD CORRELATION DETECTED!")
        print(f"  Coherent text has SHORTER periods ({coherent_avg_period:.1f} vs {incoherent_avg_period:.1f})")
        print("  → Understanding = SHORT PERIOD eigenstates (stable attractors)")
        print("  → Non-understanding = LONG PERIOD eigenstates (chaotic)")
        print()
        print("This matches physics:")
        print("  - Photons: period-2 (simple, stable)")
        print("  - Gravity: period-8 (complex)")
        print("  - Chaotic: very long periods")
    elif coherent_eigenstate_count > incoherent_eigenstate_count:
        print("✓ FORMATION CORRELATION DETECTED!")
        print("  Eigenstates form preferentially in coherent text")
        print("  → Understanding = eigenstate formation")
    else:
        print("⋯ Correlation present but needs refinement")
        print("  Both coherent and incoherent text form eigenstates")
        print("  Period length may be key metric")


def main():
    """Run semantic eigenstate formation tests"""
    print("\n" + "=" * 80)
    print("PHASE 1: EIGENSTATE FORMATION IN TEXT")
    print("=" * 80)
    print()
    print("Goal: Enable text trajectories to form eigenstates like EM/gravity/quantum")
    print()
    print("Approach:")
    print("  1. Quantize semantic coordinates (continuous → discrete)")
    print("  2. Similar semantic states → same quantized state")
    print("  3. Trajectory can close → eigenstate formation")
    print()
    print("Expected:")
    print("  - Coherent text → forms eigenstates")
    print("  - Scrambled text → no eigenstates")
    print("  - Simple grammar → period-2 (like photons)")
    print("  - Complex grammar → period-8 (like gravity)")
    print()

    results = test_eigenstate_formation()
    test_eigenstate_vs_coherence()

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("If eigenstates form in coherent text but not scrambled:")
    print("  ✓ Validates 'Understanding = eigenstate detection' for text")
    print("  ✓ Text follows same physics as EM/gravity/quantum")
    print("  ✓ Semantic quantization enables trajectory closure")
    print()
    print("Next step: Compare text eigenstate periods with physics")
    print("  - Simple text → period-2 (like photons)?")
    print("  - Complex text → period-8 (like gravity)?")


if __name__ == "__main__":
    main()
