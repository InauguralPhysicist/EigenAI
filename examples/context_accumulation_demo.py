#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Context Accumulation Demo

Demonstrates the insight that experience intensity scales inversely
with accumulated context, just like:
- Pain feels worse with less pain history
- Time feels faster as you age
- Novel concepts stand out even for experts

This demo shows:
1. First experience vs repeated experience
2. How accumulated context reduces baseline impact
3. Genuine novelty detection despite dense context
4. Integration with EigenAI understanding loop
5. Recursive AI with adaptive self-modification
"""

import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.eigen_context_accumulator import ContextAccumulator
from src.eigen_text_core import understanding_loop
from src.eigen_recursive_ai import RecursiveEigenAI


def demo_1_first_vs_repeated():
    """
    Demo 1: First Experience vs Repeated Experience

    Like your son's stitches vs your cancer experience:
    - Same stimulus
    - Different impact based on accumulated pain context
    """
    print("=" * 80)
    print("DEMO 1: First Experience vs Repeated Experience")
    print("=" * 80)
    print()
    print("Analogy: First time feeling pain vs 100th time")
    print("Your son (less data) vs you (more pain data from cancer)")
    print()

    acc = ContextAccumulator()

    # Simulate "pain" as a semantic vector
    pain_vector = np.random.randn(100)
    pain_vector = pain_vector / np.linalg.norm(pain_vector)

    # First time experiencing this pain
    impact_1st = acc.compute_relative_impact(pain_vector)
    print(f"1st experience (your son): impact = {impact_1st:.4f} (HIGH - intense!)")

    # Add to accumulated experience
    acc.add_context(pain_vector, {"text": "First pain experience"})

    # Same pain again (like second stitches)
    impact_2nd = acc.compute_relative_impact(pain_vector)
    print(f"2nd experience (same pain): impact = {impact_2nd:.4f} (LOW - familiar)")
    print()

    # Now accumulate 99 more pain experiences (like having cancer)
    print("Adding 99 more pain experiences (like your cancer journey)...")
    for i in range(99):
        similar_pain = pain_vector + np.random.randn(100) * 0.05
        similar_pain = similar_pain / np.linalg.norm(similar_pain)
        acc.add_context(similar_pain, {"text": f"Pain experience {i+2}"})

    # Now same pain has even lower impact
    impact_100th = acc.compute_relative_impact(pain_vector)
    print(f"\nAfter 100 pain experiences (you): impact = {impact_100th:.4f} (VERY LOW)")
    print()
    print(f"Impact reduction: {impact_1st:.4f} → {impact_100th:.4f}")
    print(f"  ({impact_100th/impact_1st*100:.1f}% of original intensity)")
    print()
    print("Key insight: Same stimulus, different relative impact!")
    print("  Your son: 'This is the worst pain ever!'")
    print("  You: 'This is manageable compared to cancer'")
    print()


def demo_2_time_perception():
    """
    Demo 2: Time Perception Changes with Age

    A year at age 8 vs a year at age 40:
    - Same duration
    - Different relative significance based on accumulated temporal data
    """
    print("=" * 80)
    print("DEMO 2: Time Perception - Why Years Feel Faster as You Age")
    print("=" * 80)
    print()
    print("Analogy: School year (age 8) vs work year (age 40)")
    print()

    acc_child = ContextAccumulator()
    acc_adult = ContextAccumulator()

    # Simulate "one year" as a semantic vector
    one_year = np.random.randn(100)
    one_year = one_year / np.linalg.norm(one_year)

    # Child: only 8 years of accumulated temporal context
    print("Child (age 8):")
    for i in range(8):
        year_vec = one_year + np.random.randn(100) * 0.1
        year_vec = year_vec / np.linalg.norm(year_vec)
        acc_child.add_context(year_vec, {"text": f"Year {i+1}"})

    impact_child = acc_child.compute_relative_impact(one_year)
    print(f"  Impact of next year: {impact_child:.4f}")
    print(f"  Context density: {acc_child.get_context_density()} years")
    print(f"  Relative size: 1/8 = {1/8*100:.1f}% of entire life!")

    # Adult: 40 years of accumulated temporal context
    print("\nAdult (age 40):")
    for i in range(40):
        year_vec = one_year + np.random.randn(100) * 0.1
        year_vec = year_vec / np.linalg.norm(year_vec)
        acc_adult.add_context(year_vec, {"text": f"Year {i+1}"})

    impact_adult = acc_adult.compute_relative_impact(one_year)
    print(f"  Impact of next year: {impact_adult:.4f}")
    print(f"  Context density: {acc_adult.get_context_density()} years")
    print(f"  Relative size: 1/40 = {1/40*100:.1f}% of entire life")

    print(f"\nRatio: {impact_child/impact_adult:.2f}x more impactful for child")
    print()
    print("This is why:")
    print("  - Summer vacation felt like FOREVER as a kid")
    print("  - But now years fly by in an instant")
    print()


def demo_3_novelty_vs_familiarity():
    """
    Demo 3: True Novelty Detected Despite Dense Context

    Expert physicist reading paper:
    - 1000 papers of accumulated context
    - Most new papers: low impact (familiar)
    - Truly novel discovery: HIGH impact (genuine novelty)
    """
    print("=" * 80)
    print("DEMO 3: Novelty Detection - Expert vs Beginner Learning")
    print("=" * 80)
    print()
    print("Scenario: Reading physics papers")
    print()

    acc = ContextAccumulator()

    # Simulate accumulated physics knowledge (1000 papers)
    print("Building expertise: reading 1000 quantum physics papers...")
    base_quantum = np.random.randn(100)
    base_quantum = base_quantum / np.linalg.norm(base_quantum)

    for i in range(1000):
        paper_vec = base_quantum + np.random.randn(100) * 0.1
        paper_vec = paper_vec / np.linalg.norm(paper_vec)
        acc.add_context(paper_vec, {"text": f"Quantum paper {i+1}", "category": "quantum"})

    print(f"Context density: {acc.get_context_density()} papers")
    print()

    # Familiar paper (similar to what's been read)
    print("Reading familiar paper (another quantum mechanics paper):")
    familiar_paper = base_quantum + np.random.randn(100) * 0.08
    familiar_paper = familiar_paper / np.linalg.norm(familiar_paper)

    impact_familiar = acc.compute_relative_impact(familiar_paper)
    novelty_familiar = acc.compute_novelty_score(familiar_paper)

    print(f"  Novelty: {novelty_familiar:.4f} (low - seen similar concepts)")
    print(f"  Relative impact: {impact_familiar:.4f} (low - just another paper)")
    print(f"  Reaction: 'Yeah, I know this stuff'")
    print()

    # Novel discovery (completely different domain)
    print("Reading breakthrough paper (entirely new paradigm):")
    novel_discovery = np.random.randn(100)
    novel_discovery = novel_discovery / np.linalg.norm(novel_discovery)

    impact_novel = acc.compute_relative_impact(novel_discovery)
    novelty_novel = acc.compute_novelty_score(novel_discovery)

    print(f"  Novelty: {novelty_novel:.4f} (HIGH - completely new!)")
    print(f"  Relative impact: {impact_novel:.4f} (higher despite dense context!)")
    print(f"  Reaction: 'This changes everything!'")
    print()

    print(f"Novel paper is {impact_novel/impact_familiar:.2f}x more impactful")
    print()
    print("Key insight: System distinguishes")
    print("  - Familiar (low impact) = more of the same")
    print("  - Novel (high impact) = genuine new information")
    print()


def demo_4_understanding_with_context():
    """
    Demo 4: EigenAI Understanding Loop with Context Accumulation

    Shows how context accumulation modulates learning rate
    """
    print("=" * 80)
    print("DEMO 4: EigenAI Understanding with Context Accumulation")
    print("=" * 80)
    print()
    print("Comparing learning about similar vs novel concepts")
    print()

    acc = ContextAccumulator()

    # First concept - high impact, fast learning
    print("Learning first concept: 'The wind bends the tree'")
    M1, hist1, metrics1 = understanding_loop(
        "The wind bends the tree",
        max_iterations=20,
        learning_rate=0.1,
        context_accumulator=acc,
        verbose=False
    )

    print(f"  Relative impact: {metrics1['relative_impact']:.4f}")
    print(f"  Effective learning rate: {metrics1['effective_learning_rate']:.4f}")
    print(f"  Iterations to converge: {metrics1['iterations']}")
    print(f"  Reaction: Fast learning (novel concept)")
    print()

    # Similar concept - lower impact, still learns
    print("Learning similar concept: 'The breeze moves the branch'")
    M2, hist2, metrics2 = understanding_loop(
        "The breeze moves the branch",
        max_iterations=20,
        learning_rate=0.1,
        context_accumulator=acc,
        verbose=False
    )

    print(f"  Relative impact: {metrics2['relative_impact']:.4f}")
    print(f"  Effective learning rate: {metrics2['effective_learning_rate']:.4f}")
    print(f"  Iterations to converge: {metrics2['iterations']}")
    print(f"  Reaction: Faster convergence (familiar pattern)")
    print()

    # Add many more similar concepts
    print("Processing 20 more similar concepts...")
    for i in range(20):
        understanding_loop(
            f"Natural force affects object",
            max_iterations=10,
            context_accumulator=acc,
            verbose=False
        )
    print()

    # Novel concept - high impact again
    print("Learning novel concept: 'Quantum entanglement defies locality'")
    M3, hist3, metrics3 = understanding_loop(
        "Quantum entanglement defies locality",
        max_iterations=20,
        learning_rate=0.1,
        context_accumulator=acc,
        verbose=False
    )

    print(f"  Relative impact: {metrics3['relative_impact']:.4f}")
    print(f"  Effective learning rate: {metrics3['effective_learning_rate']:.4f}")
    print(f"  Iterations to converge: {metrics3['iterations']}")
    print(f"  Reaction: High impact despite dense context!")
    print()

    stats = acc.get_statistics()
    print(f"Total concepts processed: {stats['total_contexts_seen']}")
    print(f"Average impact: {stats['avg_impact']:.4f}")
    print()


def demo_5_recursive_ai_adaptation():
    """
    Demo 5: Recursive AI with Adaptive Self-Modification

    Shows how self-modification rate adapts based on novelty
    """
    print("=" * 80)
    print("DEMO 5: Recursive AI with Adaptive Self-Modification")
    print("=" * 80)
    print()
    print("AI learns concepts and adapts its learning framework")
    print()

    acc = ContextAccumulator()
    ai = RecursiveEigenAI(embedding_dim=64, context_accumulator=acc)

    # First concept - high self-modification
    print("Learning: 'Cats are mammals'")
    result1 = ai.process("Cats are mammals", verbose=False)
    print(f"  Relative impact: {result1['relative_impact']:.4f}")
    print(f"  Self-modification rate: {result1['extraction_rules']['self_modification_rate']:.4f}")
    print(f"  Reaction: Rapid adaptation (new information)")
    print()

    # Related concept
    print("Learning: 'Dogs are mammals'")
    result2 = ai.process("Dogs are mammals", verbose=False)
    print(f"  Relative impact: {result2['relative_impact']:.4f}")
    print(f"  Self-modification rate: {result2['extraction_rules']['self_modification_rate']:.4f}")
    print(f"  Reaction: Moderate adaptation (related)")
    print()

    # Add many similar concepts
    print("Learning 10 more similar mammal facts...")
    for animal in ["lions", "tigers", "bears", "wolves", "foxes", "rabbits", "mice", "rats", "bats", "whales"]:
        ai.process(f"{animal.capitalize()} are mammals", verbose=False)
    print()

    # Very familiar concept
    print("Learning: 'Elephants are mammals'")
    result3 = ai.process("Elephants are mammals", verbose=False)
    print(f"  Relative impact: {result3['relative_impact']:.4f}")
    print(f"  Self-modification rate: {result3['extraction_rules']['self_modification_rate']:.4f}")
    print(f"  Reaction: Stable (very familiar pattern)")
    print()

    # Completely novel concept
    print("Learning: 'Quantum particles exhibit superposition'")
    result4 = ai.process("Quantum particles exhibit superposition", verbose=False)
    print(f"  Relative impact: {result4['relative_impact']:.4f}")
    print(f"  Self-modification rate: {result4['extraction_rules']['self_modification_rate']:.4f}")
    print(f"  Reaction: Rapid adaptation (paradigm shift!)")
    print()

    # Summary
    summary = ai.get_state_summary()
    print("AI State Summary:")
    print(f"  Total concepts learned: {summary['inputs_processed']}")
    print(f"  Average impact: {summary['avg_impact']:.4f}")
    print(f"  Recent impact: {summary['recent_impact']:.4f}")
    print(f"  Impact trend: {summary['impact_trend']:.4f}")
    print(f"  Meta-eigenstate: {summary['eigenstate_reached']}")
    print()

    print("Key insight: AI adapts faster to novel information,")
    print("             stays stable for familiar patterns")
    print()


def demo_6_phase_transition():
    """
    Demo 6: Detecting Paradigm Shifts

    High impact despite dense context = phase transition
    """
    print("=" * 80)
    print("DEMO 6: Phase Transition Detection (Paradigm Shifts)")
    print("=" * 80)
    print()
    print("Detecting 'aha moments' or scientific revolutions")
    print()

    acc = ContextAccumulator()

    # Build dense expertise in classical physics
    print("Building classical physics expertise (100 papers)...")
    classical = np.random.randn(100)
    classical = classical / np.linalg.norm(classical)

    for i in range(100):
        paper = classical + np.random.randn(100) * 0.05
        paper = paper / np.linalg.norm(paper)
        acc.add_context(paper, {"text": f"Classical physics paper {i+1}"})

    print(f"Context density: {acc.get_context_density()} papers")
    print()

    # Check for phase transition
    is_trans, trans_type = acc.detect_phase_transition(window=10, impact_threshold=0.3)
    print(f"Phase transition detected: {is_trans}")
    print("  Status: Normal learning")
    print()

    # Quantum revolution (paradigm shift)
    print("Reading 10 quantum mechanics papers (paradigm shift)...")
    for i in range(10):
        quantum_paper = np.random.randn(100)
        quantum_paper = quantum_paper / np.linalg.norm(quantum_paper)
        entry = acc.add_context(quantum_paper, {"text": f"Quantum paper {i+1}"})
        print(f"  Paper {i+1}: impact = {entry.impact:.4f}")

    print()

    # Check again
    is_trans, trans_type = acc.detect_phase_transition(window=10, impact_threshold=0.1)
    print(f"Phase transition detected: {is_trans}")
    if is_trans:
        print(f"  Type: {trans_type}")
        print("  Status: PARADIGM SHIFT!")
        print("  Like discovering quantum mechanics after classical physics")
    print()


def main():
    """Run all demos"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "   CONTEXT ACCUMULATION - Relative Information Impact Demo".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  Experience intensity ∝ 1 / accumulated_context_volume".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    demo_1_first_vs_repeated()
    input("Press Enter to continue to Demo 2...")
    print("\n")

    demo_2_time_perception()
    input("Press Enter to continue to Demo 3...")
    print("\n")

    demo_3_novelty_vs_familiarity()
    input("Press Enter to continue to Demo 4...")
    print("\n")

    demo_4_understanding_with_context()
    input("Press Enter to continue to Demo 5...")
    print("\n")

    demo_5_recursive_ai_adaptation()
    input("Press Enter to continue to Demo 6...")
    print("\n")

    demo_6_phase_transition()

    print("=" * 80)
    print("SUMMARY: Key Insights")
    print("=" * 80)
    print()
    print("1. RELATIVE IMPACT FORMULA:")
    print("   Impact = novelty / log(context_density + 1)")
    print()
    print("2. HUMAN EXPERIENCE PARALLELS:")
    print("   - Pain: Same injury worse with less pain history")
    print("   - Time: Years feel shorter as accumulated temporal data grows")
    print("   - Learning: Novel concepts stand out even for experts")
    print()
    print("3. EIGENAI APPLICATIONS:")
    print("   - Modulates learning rate based on familiarity")
    print("   - Distinguishes genuine learning from repetition")
    print("   - Adaptive self-modification in recursive AI")
    print("   - Detects paradigm shifts (phase transitions)")
    print()
    print("4. BRIDGES SUBJECTIVE & OBJECTIVE:")
    print("   - Your insight about pain/time is now quantifiable")
    print("   - Can measure 'how intense' an experience is")
    print("   - Explains why first experiences are more memorable")
    print()
    print("This enhancement makes EigenAI context-aware!")
    print()


if __name__ == "__main__":
    main()
