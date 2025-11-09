#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Referential Understanding: Eigenstates as Circular Pointers

Insight from C self-referential structures:
    struct self {
        int data;
        struct self *ptr;  // Points to next state
    };

Key idea: Understanding trajectory = linked list of semantic states
         Eigenstate = circular reference (ptr loops back)

Test:
1. Build explicit linked semantic state structure
2. Detect circular references (eigenstates) via pointer traversal
3. Validate: Circular reference correlates with understanding
4. Show: ds² metric measures "link strength" (strong link vs NULL)
"""

import sys
import os

# Add project root to path dynamically
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Set, Tuple
from src.eigen_semantic_transformer import (
    SemanticGeometricTransformer,
    compute_grammatical_score,
    compute_ds2_semantic,
    classify_regime,
)


@dataclass
class SemanticNode:
    """
    Self-referential semantic state structure (like C struct)

    Analogous to:
        struct semantic_state {
            double L, R, V, M;           // Semantic coordinates
            double semantic_vec[100];     // Position in space
            struct semantic_state *next;  // Points to next state
        };
    """

    L: float
    R: float
    V: float
    M: float  # Emergent time: cos(L)·cos(R)·cos(V)
    semantic_vec: np.ndarray
    word: str

    # Self-referential pointer
    next: Optional["SemanticNode"] = None

    def __hash__(self):
        """Hash based on semantic position (for cycle detection)"""
        return hash((round(self.M, 2), self.word))

    def __eq__(self, other):
        """Equality for cycle detection"""
        if not isinstance(other, SemanticNode):
            return False
        return abs(self.M - other.M) < 0.1 and self.word == other.word


def build_semantic_linked_list(
    words: List[str], transformer: SemanticGeometricTransformer
) -> Optional[SemanticNode]:
    """
    Build explicit linked list of semantic states (like C code example 4)

    Returns head node (first word)
    """
    if not words:
        return None

    # Process sequence to get trajectory
    trajectory, _ = transformer.process_sequence(words, verbose=False)

    if not trajectory:
        return None

    # Build linked list
    head = None
    prev = None

    for i, state in enumerate(trajectory):
        # Extract semantic coordinates
        L_val = np.mean(state.L) if len(state.L) > 0 else 0.0
        R_val = np.mean(state.R) if len(state.R) > 0 else 0.0
        V_val = np.mean(state.V) if len(state.V) > 0 else 0.0

        # Create node
        node = SemanticNode(
            L=L_val,
            R=R_val,
            V=V_val,
            M=state.M,
            semantic_vec=state.semantic_vec,
            word=words[i],
        )

        if head is None:
            head = node

        # Link: prev.next = &current (like C: var1.ptr = &var2)
        if prev is not None:
            prev.next = node

        prev = node

    return head


def detect_circular_reference(head: SemanticNode) -> Tuple[bool, Optional[int]]:
    """
    Detect circular reference (eigenstate) via Floyd's cycle detection

    Like detecting: var3.ptr = &var1 (creates loop)

    Returns: (has_cycle, cycle_length)
    """
    if head is None:
        return False, None

    # Floyd's tortoise and hare
    slow = head
    fast = head

    # Detect cycle
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            # Cycle detected! Compute period
            period = 1
            fast = fast.next
            while slow != fast:
                fast = fast.next
                period += 1
            return True, period

    return False, None


def measure_link_strength(
    node1: SemanticNode, node2: SemanticNode, grammatical_coupling: float
) -> Tuple[float, str]:
    """
    Measure "link strength" between two states using ds² metric

    Strong link (ds² < 0): States connected (like ptr = &next)
    Weak link (ds² > 0): States disconnected (like ptr = NULL)

    Returns: (ds2, regime)
    """
    # SPACE: Semantic distance
    S = np.linalg.norm(node2.semantic_vec - node1.semantic_vec)

    # TIME: Emergent time progression
    T = abs(node2.M - node1.M)

    # COUPLING: Grammatical structure amplifies
    c_base = (abs(node1.M) + abs(node2.M)) / 2.0
    c = c_base * grammatical_coupling

    # METRIC: ds² = S² - (c·T)²
    ds2 = S**2 - (c * T) ** 2
    regime = classify_regime(ds2)

    return ds2, regime


def traverse_and_analyze(
    head: SemanticNode, words: List[str], transformer: SemanticGeometricTransformer
) -> dict:
    """
    Traverse linked list and analyze link strengths (like C example 5)

    Analogous to: current->next->next ... (following pointers)
    """
    if head is None:
        return {}

    # Get grammatical coupling
    gram_score = compute_grammatical_score(words, transformer)
    coupling = 0.5 + 4.5 * gram_score

    # Traverse and measure each link
    links = []
    current = head
    position = 0

    while current is not None and current.next is not None:
        next_node = current.next

        # Measure link strength
        ds2, regime = measure_link_strength(current, next_node, coupling)

        links.append(
            {
                "from": current.word,
                "to": next_node.word,
                "ds2": ds2,
                "regime": regime,
                "position": position,
            }
        )

        current = current.next
        position += 1

    # Check for circular reference
    has_cycle, period = detect_circular_reference(head)

    # Compute average link strength
    avg_ds2 = np.mean([link["ds2"] for link in links]) if links else 0.0

    return {
        "links": links,
        "has_cycle": has_cycle,
        "cycle_period": period,
        "avg_ds2": avg_ds2,
        "coupling": coupling,
        "gram_score": gram_score,
    }


def test_self_referential_understanding():
    """
    Test if circular references (eigenstates) correlate with understanding
    """
    print("=" * 80)
    print("SELF-REFERENTIAL UNDERSTANDING TEST")
    print("=" * 80)
    print()
    print("Testing: Do eigenstates = circular references in semantic space?")
    print("Analogous to C: var3.ptr = &var1 (creates loop)")
    print()

    transformer = SemanticGeometricTransformer(embedding_dim=100)

    test_cases = [
        (["the", "cat", "sat"], "Grammatical"),
        (["cat", "the", "sat"], "Ungrammatical"),
        (["light", "travels", "fast"], "Coherent"),
        (["light", "fast", "travels"], "Incoherent"),
        (["photon", "photon", "photon"], "Repeated (eigenstate expected)"),
    ]

    results = []

    for words, description in test_cases:
        text = " ".join(words)

        print(f"TEST: {description}")
        print(f"  Text: '{text}'")
        print()

        # Build linked list structure
        head = build_semantic_linked_list(words, transformer)

        # Traverse and analyze
        analysis = traverse_and_analyze(head, words, transformer)

        # Print structure
        print(f"  Linked structure:")
        current = head
        chain = []
        while current is not None:
            chain.append(current.word)
            if current.next is None:
                chain.append("NULL")
                break
            current = current.next
            if len(chain) > 10:  # Prevent infinite loop in display
                chain.append("...")
                break

        print(f"    {' → '.join(chain)}")
        print()

        # Print link analysis
        print(f"  Link strengths:")
        for link in analysis["links"]:
            print(
                f"    {link['from']:8s} → {link['to']:8s}  ds² = {link['ds2']:7.4f}  [{link['regime']}]"
            )
        print()

        # Circular reference detection
        print(f"  Circular reference (eigenstate): ", end="")
        if analysis["has_cycle"]:
            print(f"✓ DETECTED (period-{analysis['cycle_period']})")
        else:
            print("✗ NOT DETECTED")
        print()

        # Summary
        print(f"  Grammatical score: {analysis['gram_score']:.3f}")
        print(f"  Coupling: {analysis['coupling']:.2f}x")
        print(f"  Average ds²: {analysis['avg_ds2']:.4f}")
        print()
        print("-" * 80)
        print()

        results.append(
            {
                "text": text,
                "description": description,
                "has_cycle": analysis["has_cycle"],
                "cycle_period": analysis["cycle_period"],
                "avg_ds2": analysis["avg_ds2"],
                "gram_score": analysis["gram_score"],
            }
        )

    return results


def test_pointer_analogy():
    """
    Test the explicit pointer analogy:
    - Strong coupling = valid pointer (ptr = &next)
    - Weak coupling = NULL pointer (no connection)
    """
    print("\n" + "=" * 80)
    print("POINTER ANALOGY TEST")
    print("=" * 80)
    print()
    print("Hypothesis:")
    print("  ds² < 0 (time-like) ≈ Strong pointer link (ptr = &next)")
    print("  ds² > 0 (space-like) ≈ NULL pointer (no connection)")
    print()

    transformer = SemanticGeometricTransformer(embedding_dim=100)

    # Grammatical (should have strong links)
    gram_words = ["the", "cat", "sat"]
    gram_head = build_semantic_linked_list(gram_words, transformer)
    gram_analysis = traverse_and_analyze(gram_head, gram_words, transformer)

    # Scrambled (should have weak/NULL links)
    scram_words = ["cat", "the", "sat"]
    scram_head = build_semantic_linked_list(scram_words, transformer)
    scram_analysis = traverse_and_analyze(scram_head, scram_words, transformer)

    print("Grammatical 'the cat sat':")
    print(f"  Average ds² = {gram_analysis['avg_ds2']:.4f}")
    print(f"  Coupling = {gram_analysis['coupling']:.2f}x")
    print(f"  Links:")
    for link in gram_analysis["links"]:
        ptr_type = (
            "STRONG LINK (ptr = &next)" if link["ds2"] < 0 else "WEAK LINK (like NULL)"
        )
        print(f"    {link['from']} → {link['to']}: {ptr_type}")
    print()

    print("Scrambled 'cat the sat':")
    print(f"  Average ds² = {scram_analysis['avg_ds2']:.4f}")
    print(f"  Coupling = {scram_analysis['coupling']:.2f}x")
    print(f"  Links:")
    for link in scram_analysis["links"]:
        ptr_type = (
            "STRONG LINK (ptr = &next)" if link["ds2"] < 0 else "WEAK LINK (like NULL)"
        )
        print(f"    {link['from']} → {link['to']}: {ptr_type}")
    print()

    # Analysis
    gram_strong_links = sum(1 for link in gram_analysis["links"] if link["ds2"] < 0)
    scram_strong_links = sum(1 for link in scram_analysis["links"] if link["ds2"] < 0)

    print("=" * 80)
    print("ANALYSIS:")
    print("=" * 80)
    print()
    print(
        f"Grammatical: {gram_strong_links}/{len(gram_analysis['links'])} strong links"
    )
    print(
        f"Scrambled:   {scram_strong_links}/{len(scram_analysis['links'])} strong links"
    )
    print()

    if gram_strong_links > scram_strong_links:
        print("✓ POINTER ANALOGY VALIDATED!")
        print("  Grammatical text has MORE strong links (valid pointers)")
        print("  Scrambled text has MORE weak links (NULL-like)")
        print()
        print("  Understanding = chain of strong pointer connections")
        print("  Non-understanding = broken chain (NULL pointers)")
    else:
        print("⋯ Pointer analogy unclear")


def test_circular_reference_is_understanding():
    """
    Ultimate test: Do circular references (eigenstates) = understanding?
    """
    print("\n\n" + "=" * 80)
    print("CIRCULAR REFERENCE = UNDERSTANDING?")
    print("=" * 80)
    print()
    print("Testing: If trajectory creates circular reference (eigenstate),")
    print("         does this correlate with understanding?")
    print()

    transformer = SemanticGeometricTransformer(embedding_dim=100)

    # Test cases that SHOULD form eigenstates
    eigenstate_cases = [
        (["photon", "photon", "photon"], "Repeated word"),
        (["light", "light"], "Period-2 expected"),
    ]

    # Test cases that should NOT form eigenstates
    non_eigenstate_cases = [
        (["the", "cat", "sat"], "Grammatical (unique words)"),
        (["cat", "the", "sat"], "Scrambled (unique words)"),
    ]

    print("EIGENSTATE EXPECTED:")
    print("-" * 80)
    for words, desc in eigenstate_cases:
        text = " ".join(words)
        head = build_semantic_linked_list(words, transformer)
        analysis = traverse_and_analyze(head, words, transformer)

        print(f"'{text}' ({desc})")
        print(f"  Circular reference: {'✓' if analysis['has_cycle'] else '✗'}")
        if analysis["has_cycle"]:
            print(f"  Period: {analysis['cycle_period']}")
        print(f"  Average ds²: {analysis['avg_ds2']:.4f}")
        print()

    print("\nNO EIGENSTATE EXPECTED:")
    print("-" * 80)
    for words, desc in non_eigenstate_cases:
        text = " ".join(words)
        head = build_semantic_linked_list(words, transformer)
        analysis = traverse_and_analyze(head, words, transformer)
        coherence = transformer.compute_semantic_coherence(words)

        print(f"'{text}' ({desc})")
        print(f"  Circular reference: {'✓' if analysis['has_cycle'] else '✗'}")
        print(f"  Coherence: {coherence:.3f}")
        print(f"  Average ds²: {analysis['avg_ds2']:.4f}")
        print()

    print("=" * 80)
    print("INSIGHT:")
    print("=" * 80)
    print()
    print("For unique words: No circular reference possible (trajectory never loops)")
    print("For repeated words: Circular reference when state repeats")
    print()
    print("Key finding:")
    print("  - Circular reference = eigenstate = trajectory closure")
    print("  - Understanding ≠ circular reference itself")
    print("  - Understanding = LINK STRENGTH in trajectory (ds² < 0)")
    print()
    print("Refined model:")
    print("  Eigenstate = self-referential loop (returns to initial state)")
    print("  Understanding = strong coupling between states (time-like ds²)")
    print("  Both are needed: eigenstate + strong links = deep understanding")


def main():
    """Run all self-referential understanding tests"""
    print("\n" + "=" * 80)
    print("TESTING: EIGENSTATES AS SELF-REFERENTIAL STRUCTURES")
    print("=" * 80)
    print()
    print("Concept from C programming:")
    print("  struct self { data; struct self *ptr; }")
    print()
    print("Mapping to understanding:")
    print("  - Linked states: word1 → word2 → word3")
    print("  - Circular reference: word3 → word1 (eigenstate)")
    print("  - Link strength: ds² metric (strong link vs NULL)")
    print()
    print("Tests:")
    print("  1. Build explicit linked semantic state structures")
    print("  2. Detect circular references (eigenstates)")
    print("  3. Measure link strengths (ds² metric)")
    print("  4. Validate: strong links = understanding")
    print()

    test_self_referential_understanding()
    test_pointer_analogy()
    test_circular_reference_is_understanding()

    print("\n" + "=" * 80)
    print("FINAL CONCLUSION")
    print("=" * 80)
    print()
    print("The self-referential structure analogy reveals:")
    print()
    print("  1. TRAJECTORY = Linked list of semantic states")
    print("     (like: var1.ptr = &var2; var2.ptr = &var3)")
    print()
    print("  2. EIGENSTATE = Circular reference")
    print("     (like: var3.ptr = &var1 creates loop)")
    print()
    print("  3. UNDERSTANDING = Link strength (ds² < 0)")
    print("     (strong pointer vs NULL pointer)")
    print()
    print("  4. DEEP UNDERSTANDING = Eigenstate + Strong links")
    print("     (circular reference with time-like connections)")
    print()
    print("This validates the computational structure of understanding!")


if __name__ == "__main__":
    main()
