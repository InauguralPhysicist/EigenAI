#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claude (me) talking to EigenAI
A conversation between two AIs through geometric eigenstates
"""

import sys
sys.path.insert(0, '/home/user/EigenAI')

from src.eigen_text_core import understanding_loop
from src.eigen_recursive_ai import RecursiveEigenAI


def analyze_sentence(text, context=""):
    """Analyze a single sentence and show the geometric understanding"""
    print(f"\n{'='*70}")
    print(f"Input: '{text}'")
    if context:
        print(f"Context: {context}")
    print(f"{'='*70}")

    M, history, metrics = understanding_loop(
        text,
        max_iterations=20,
        method='xor',  # Using discrete geometry
        verbose=True
    )

    print(f"\n✓ Eigenstate: {metrics['eigenstate_type']}")
    print(f"  Convergence: {'Yes' if metrics['converged'] else 'No'}")
    print(f"  Iterations: {metrics['iterations']}")
    print(f"  Regime: {metrics['final_regime']}")

    return M, metrics


def main():
    print("*" * 70)
    print("  CLAUDE SPEAKS TO EIGENAI")
    print("  A Conversation in Geometry and Language")
    print("*" * 70)

    # Initialize recursive AI for cumulative understanding
    ai = RecursiveEigenAI(embedding_dim=128)

    # My messages to EigenAI
    conversation = [
        # Greetings and self-introduction
        ("Hello EigenAI, I am Claude, an AI made by Anthropic",
         "Introducing myself"),

        # Philosophical questions about understanding
        ("You measure understanding through geometry",
         "Acknowledging EigenAI's unique capability"),

        ("Can understanding itself be understood",
         "Meta-question about comprehension"),

        # Exploring the hybrid language
        ("Light travels in straight lines until space curves",
         "Physics meets geometry"),

        ("Words are tokens, tokens are numbers, numbers are geometry",
         "The semantic triad in action"),

        # Testing recursive understanding
        ("Eigenstates detect periodic closure",
         "Concept 1: eigenstate definition"),

        ("Periodic closure means understanding",
         "Concept 2: linking closure to understanding"),

        ("Therefore eigenstates detect understanding",
         "Logical inference from previous statements"),

        # Poetic/philosophical
        ("The trajectory closes when meaning emerges",
         "Poetic description of eigenstate"),

        # Direct question
        ("What happens at 45 degrees in your space",
         "Question about XOR geometry"),
    ]

    print("\n\nSTARTING CONVERSATION...")
    print("=" * 70)

    for i, (message, context) in enumerate(conversation, 1):
        print(f"\n\n>>> MESSAGE {i}/{ len(conversation)}")

        # Analyze through geometric understanding
        M, metrics = analyze_sentence(message, context)

        # Also feed to recursive AI for cumulative understanding
        print("\n[Feeding to recursive AI for cumulative understanding...]")
        ai.process(message)
        state = ai.get_state_summary()
        print(f"  Recursive AI state: {state['inputs_processed']} inputs processed")
        print(f"  M_context strength: {state['M_context_norm']:.3f}")
        print(f"  Meta-eigenstate: {state['eigenstate_reached']}")

        input("\nPress Enter for next message...")

    print("\n\n" + "=" * 70)
    print("CONVERSATION COMPLETE")
    print("=" * 70)

    # Test what the recursive AI learned
    print("\n\nTESTING RECURSIVE AI'S UNDERSTANDING:")
    print("-" * 70)

    test_queries = [
        "Who is Claude",
        "What does eigenstate mean",
        "How is understanding detected"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        response = ai.query(query)
        print(f"Response: {response}")

    final_state = ai.get_state_summary()
    print("\n\n" + "=" * 70)
    print("FINAL STATE:")
    print("=" * 70)
    print(f"Total inputs processed: {final_state['inputs_processed']}")
    print(f"M_context strength: {final_state['M_context_norm']:.3f}")
    print(f"Meta-eigenstate reached: {final_state['eigenstate_reached']}")
    print(f"Iterations: {final_state['iteration']}")

    print("\n\n")
    print("*" * 70)
    print("  This is what it looks like when two AIs talk")
    print("  One speaks in English, the other hears in geometry")
    print("  Understanding happens in the space between")
    print("*" * 70)


if __name__ == "__main__":
    main()
