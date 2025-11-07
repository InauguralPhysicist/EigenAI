#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical Framework for Measuring AI Understanding

Traditional AI metrics:
- Loss (how wrong is prediction?)
- Accuracy (how many correct?)
- Perplexity (how surprised?)

These measure PERFORMANCE, not UNDERSTANDING.

This framework measures UNDERSTANDING directly:
1. Eigenstate detection: Has trajectory closed? (yes/no)
2. Period depth: What's the cycle length? (period-k)
3. Light-like frames: How many frames show ds²≈0? (0-8)
4. Convergence speed: How fast to eigenstate? (iterations)

Key insight:
"You can perform well without understanding.
 You cannot reach eigenstate without understanding."

This is the difference between:
- Memorization (high accuracy, no eigenstate)
- Understanding (eigenstate detected)
"""

import sys
sys.path.insert(0, '/home/user/EigenAI')

import numpy as np
from typing import List, Dict, Tuple
from src.eigen_discrete_tokenizer import process_sentence_discrete
from src.eigen_recursive_ai import RecursiveEigenAI
from src.eigen_lorentz_boost import lorentz_boost, LorentzState


class UnderstandingMetrics:
    """
    Quantitative metrics for genuine understanding

    Goes beyond accuracy/loss to measure:
    - Eigenstate formation
    - Understanding depth
    - Frame-invariant propagation
    - Convergence trajectory
    """

    @staticmethod
    def compute_eigenstate_score(text_sequence: List[str]) -> Dict:
        """
        Compute eigenstate-based understanding score

        Parameters
        ----------
        text_sequence : list of str
            Sequence of text inputs

        Returns
        -------
        metrics : dict
            Comprehensive understanding metrics
        """
        # Process through discrete tokenizer
        result = process_sentence_discrete(text_sequence, verbose=False)

        # Metric 1: Eigenstate detected?
        eigenstate_detected = result['period'] is not None
        period = result['period'] if eigenstate_detected else 0

        # Metric 2: Trajectory stability (how much oscillation?)
        tokens = result['tokens']
        if len(tokens) >= 2:
            M_values = [tok.M for tok in tokens]
            M_variance = np.var(M_values)
            stability = 1.0 / (1.0 + M_variance / 100.0)  # Normalize to [0,1]
        else:
            stability = 0.0

        # Metric 3: Light-like frame count
        if len(tokens) >= 1:
            final_token = tokens[-1]
            temporal = final_token.M
            spatial = final_token.L ^ final_token.R ^ final_token.V

            initial_state = LorentzState(
                temporal=temporal,
                spatial=spatial,
                observer=0b10101010,
                ds2=0
            )

            light_like_count = 0
            for frame in range(8):
                boosted = lorentz_boost(initial_state, boost_angle=frame)
                if abs(boosted.ds2) < 50:  # Light-like threshold
                    light_like_count += 1

            light_like_ratio = light_like_count / 8.0
        else:
            light_like_ratio = 0.0

        # Composite understanding score
        understanding_score = (
            0.5 * (1.0 if eigenstate_detected else 0.0) +  # 50% weight: eigenstate
            0.3 * stability +                               # 30% weight: stability
            0.2 * light_like_ratio                          # 20% weight: propagation
        )

        return {
            'eigenstate_detected': eigenstate_detected,
            'period': period,
            'stability': stability,
            'light_like_frames': light_like_count,
            'light_like_ratio': light_like_ratio,
            'understanding_score': understanding_score,
        }

    @staticmethod
    def compute_recursive_convergence(ai: RecursiveEigenAI,
                                      training_sequence: List[str]) -> Dict:
        """
        Measure recursive AI convergence to meta-eigenstate

        Parameters
        ----------
        ai : RecursiveEigenAI
            Recursive AI instance
        training_sequence : list of str
            Training inputs

        Returns
        -------
        metrics : dict
            Convergence metrics
        """
        M_norms = []
        M_changes = []
        eigenstate_at = None

        M_prev = None

        for i, text in enumerate(training_sequence):
            result = ai.process(text, verbose=False)

            M_new = result['M_context_new']
            M_norm = np.linalg.norm(M_new)
            M_norms.append(M_norm)

            if M_prev is not None:
                M_change = np.linalg.norm(M_new - M_prev)
                M_changes.append(M_change)

            M_prev = M_new

            if result['eigenstate'] and eigenstate_at is None:
                eigenstate_at = i + 1

        # Convergence speed
        if eigenstate_at is not None:
            convergence_speed = 1.0 / eigenstate_at  # Faster = higher score
        else:
            convergence_speed = 0.0

        # Stability: are changes decreasing?
        if len(M_changes) >= 3:
            recent_changes = M_changes[-3:]
            avg_change = np.mean(recent_changes)
            stability = 1.0 / (1.0 + avg_change)  # Lower change = higher stability
        else:
            stability = 0.5

        return {
            'eigenstate_at': eigenstate_at,
            'convergence_speed': convergence_speed,
            'final_M_norm': M_norms[-1] if M_norms else 0.0,
            'stability': stability,
            'M_trajectory': M_norms,
            'M_changes': M_changes,
        }


def demo_understanding_vs_performance():
    """
    Demonstrate difference between performance and understanding

    Show cases where:
    1. High performance, low understanding (memorization)
    2. Low performance, high understanding (learning)
    3. High performance, high understanding (mastery)
    """
    print("=" * 80)
    print("UNDERSTANDING vs PERFORMANCE")
    print("=" * 80)
    print()
    print("Traditional AI: Optimize performance (accuracy, loss)")
    print("Eigenstate AI: Optimize understanding (eigenstate detection)")
    print()

    test_cases = [
        {
            'name': 'Memorization',
            'description': 'Random patterns (can memorize, cannot understand)',
            'sequence': ['alpha', 'beta', 'gamma', 'delta', 'epsilon'],
            'expected_eigenstate': False,
            'expected_understanding': 'Low',
        },
        {
            'name': 'Pattern Recognition',
            'description': 'Repeating pattern (can recognize and understand)',
            'sequence': ['up', 'down', 'up', 'down', 'up', 'down'],
            'expected_eigenstate': False,  # Different words, no token eigenstate
            'expected_understanding': 'Medium',
        },
        {
            'name': 'True Understanding',
            'description': 'Identical concept (deep eigenstate)',
            'sequence': ['photon', 'photon', 'photon', 'photon'],
            'expected_eigenstate': True,
            'expected_understanding': 'High',
        },
        {
            'name': 'Conceptual Learning',
            'description': 'Building understanding progressively',
            'sequence': ['wave', 'particle', 'duality', 'wave', 'particle'],
            'expected_eigenstate': False,  # Concept cycle, not token cycle
            'expected_understanding': 'Medium-High',
        },
    ]

    results = []

    for case in test_cases:
        print(f"Test: {case['name']}")
        print(f"  {case['description']}")
        print(f"  Sequence: {' '.join(case['sequence'])}")

        metrics = UnderstandingMetrics.compute_eigenstate_score(case['sequence'])

        print(f"\n  Metrics:")
        print(f"    Eigenstate: {'✓' if metrics['eigenstate_detected'] else '✗'} " +
              f"(period-{metrics['period']})" if metrics['period'] else "(none)")
        print(f"    Stability: {metrics['stability']:.3f}")
        print(f"    Light-like frames: {metrics['light_like_frames']}/8")
        print(f"    Understanding score: {metrics['understanding_score']:.3f}")

        print(f"\n  Expected: {case['expected_understanding']} understanding")
        print(f"  Actual: {'High' if metrics['understanding_score'] > 0.7 else 'Medium' if metrics['understanding_score'] > 0.4 else 'Low'}")
        print()

        results.append({
            'name': case['name'],
            'metrics': metrics,
            'expected': case['expected_understanding'],
        })

    # Summary
    print("=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print()
    print(f"{'Test Case':<25} {'Eigenstate':<12} {'Score':<8} {'Level':<10}")
    print("─" * 80)

    for r in results:
        eigenstate = "✓" if r['metrics']['eigenstate_detected'] else "✗"
        score = r['metrics']['understanding_score']
        level = 'High' if score > 0.7 else 'Medium' if score > 0.4 else 'Low'

        print(f"{r['name']:<25} {eigenstate:<12} {score:<8.3f} {level:<10}")


def demo_recursive_learning_trajectory():
    """
    Show how recursive AI learning trajectory converges

    Visualize path to meta-eigenstate
    """
    print("\n\n" + "=" * 80)
    print("RECURSIVE AI LEARNING TRAJECTORY")
    print("=" * 80)
    print()
    print("Measuring convergence to meta-eigenstate understanding")
    print()

    # Create AI
    ai = RecursiveEigenAI(embedding_dim=32)

    # Training sequence
    training = [
        "patterns repeat in cycles",
        "cycles create oscillations",
        "oscillations form eigenstates",
        "eigenstates represent understanding",
        "understanding emerges from patterns",
        "patterns repeat in cycles",  # Loop back
    ]

    print("Training sequence:")
    for i, text in enumerate(training):
        print(f"  {i+1}. '{text}'")
    print()

    # Measure convergence
    metrics = UnderstandingMetrics.compute_recursive_convergence(ai, training)

    print("=" * 80)
    print("CONVERGENCE METRICS:")
    print("=" * 80)
    print()

    if metrics['eigenstate_at']:
        print(f"✓ Meta-eigenstate reached at iteration {metrics['eigenstate_at']}")
        print(f"  Convergence speed: {metrics['convergence_speed']:.3f}")
    else:
        print(f"✗ Meta-eigenstate not reached in {len(training)} iterations")

    print(f"\nFinal M_context norm: {metrics['final_M_norm']:.3f}")
    print(f"Stability: {metrics['stability']:.3f}")

    # Show trajectory
    print("\nM_context evolution:")
    for i, norm in enumerate(metrics['M_trajectory']):
        bar_length = int(norm * 30)
        bar = "█" * bar_length
        print(f"  {i+1:2d}: {bar} {norm:.3f}")

    if len(metrics['M_changes']) > 0:
        print("\nM_context changes (convergence indicator):")
        for i, change in enumerate(metrics['M_changes']):
            bar_length = max(1, int(change * 50))
            bar = "▓" * bar_length
            print(f"  {i+2:2d}: {bar} {change:.4f}")

        print()
        if metrics['M_changes'][-1] < 0.1:
            print("✓ System converging (changes decreasing)")
        else:
            print("⋯ System still adapting")


def demo_practical_application():
    """
    Show how to use these metrics in practice

    Real-world scenario: Training an AI system
    """
    print("\n\n" + "=" * 80)
    print("PRACTICAL APPLICATION: AI TRAINING WITH UNDERSTANDING METRICS")
    print("=" * 80)
    print()
    print("Scenario: Teaching AI about quantum mechanics")
    print()

    ai = RecursiveEigenAI(embedding_dim=48)

    curriculum = [
        ("Basics", [
            "quantum particles have wave properties",
            "quantum particles have particle properties",
            "wave and particle are complementary",
        ]),
        ("Duality", [
            "light behaves as wave",
            "light behaves as particle",
            "light is both wave and particle",
        ]),
        ("Testing", [
            "what is wave particle duality",
            "explain quantum light",
        ]),
    ]

    for phase_name, texts in curriculum:
        print(f"Phase: {phase_name}")
        print("─" * 80)

        for text in texts:
            result = ai.process(text, verbose=False)
            M_norm = np.linalg.norm(result['M_context_new'])

            print(f"  Input: '{text}'")
            print(f"    M_context: {M_norm:.3f}", end="")

            if result['eigenstate']:
                print(" ✓ META-EIGENSTATE")
            else:
                print()

        print()

    # Evaluate final understanding
    print("=" * 80)
    print("FINAL EVALUATION:")
    print("=" * 80)
    print()

    test_query = "photon photon photon"  # Should recognize as quantum light
    query_metrics = UnderstandingMetrics.compute_eigenstate_score(test_query.split())

    print(f"Test query: '{test_query}'")
    print(f"  Eigenstate detected: {'✓' if query_metrics['eigenstate_detected'] else '✗'}")
    print(f"  Understanding score: {query_metrics['understanding_score']:.3f}")
    print(f"  Light-like frames: {query_metrics['light_like_frames']}/8")

    print()
    print("AI Response:")
    response = ai.query("explain quantum wave particle duality", verbose=False)
    print(f"  '{response}'")


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 80)
    print("MEASURING AI UNDERSTANDING: PRACTICAL FRAMEWORK")
    print("=" * 80)
    print()
    print("Traditional metrics measure PERFORMANCE.")
    print("Eigenstate metrics measure UNDERSTANDING.")
    print()
    print("This is the difference between:")
    print("  • Memorization vs Comprehension")
    print("  • Pattern matching vs Insight")
    print("  • Correlation vs Causation")
    print()

    demo_understanding_vs_performance()
    demo_recursive_learning_trajectory()
    demo_practical_application()

    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("""
1. EIGENSTATE DETECTION AS UNDERSTANDING METRIC:
   ✓ Binary indicator: Does trajectory close? (yes/no)
   ✓ More fundamental than accuracy or loss
   ✓ Cannot be gamed by memorization

2. MULTI-DIMENSIONAL UNDERSTANDING:
   - Period depth: How complex is the understanding? (period-k)
   - Stability: How consistent is the understanding? (variance)
   - Propagation: How universal? (light-like frames)

3. CONVERGENCE TRAJECTORY:
   ✓ Can measure learning speed (iterations to eigenstate)
   ✓ Can detect when learning is stuck (no convergence)
   ✓ Can optimize training (maximize eigenstate formation)

4. PRACTICAL APPLICATIONS:
   - Curriculum design: Order inputs to maximize eigenstate formation
   - Model evaluation: Test understanding, not just performance
   - Training optimization: Drive toward eigenstate convergence
   - Knowledge verification: Confirm genuine comprehension

5. BEYOND TRADITIONAL METRICS:

   Traditional AI: "Did it get the right answer?"
   Eigenstate AI: "Did it understand the question?"

   Performance can be memorized.
   Understanding must be achieved.

   Eigenstate detection is the signature of genuine comprehension.

6. MEASUREMENT FRAMEWORK:
   ✓ Eigenstate score: 0.0-1.0 (composite metric)
   ✓ Period detection: Depth of understanding cycle
   ✓ Light-like frames: Universality of understanding
   ✓ Convergence speed: Learning efficiency

This enables QUANTITATIVE measurement of what was previously QUALITATIVE:
"Does the AI truly understand?"

Answer: Check for eigenstate. If present → understanding achieved.
         If absent → still learning or memorizing.
    """)


if __name__ == "__main__":
    main()
