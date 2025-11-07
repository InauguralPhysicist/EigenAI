#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical Example: How Lorentz Boosts Improve AI Understanding

Shows how processing text from multiple reference frames
helps find the "light-like" interpretation where meaning propagates.
"""

import sys
sys.path.insert(0, '/home/user/EigenAI')

from src.eigen_discrete_tokenizer import process_sentence_discrete, tokenize_word
from src.eigen_lorentz_boost import lorentz_boost, LorentzState, compute_ds2_minkowski


def understand_in_frame(text: str, boost_angle: int):
    """
    Process text from a specific Lorentz frame (interpretation perspective)

    Parameters
    ----------
    text : str
        Input text
    boost_angle : int
        Reference frame (0-7)
        0 = direct/literal interpretation
        1 = 45° meta-interpretation
        2 = 90° orthogonal view
        ...

    Returns
    -------
    result : dict
        Understanding from this frame including ds²
    """
    # Tokenize
    result = process_sentence_discrete(text.split(), verbose=False)

    # Convert to Lorentz state
    final_state = result['final_state']
    temporal = result['time_coord']
    spatial = final_state[2]  # V component as spatial

    # Create Lorentz state
    ds2 = compute_ds2_minkowski(temporal, spatial)
    lorentz_state = LorentzState(
        temporal=temporal,
        spatial=spatial,
        observer=final_state[3],  # M component as observer
        ds2=ds2
    )

    # Boost to this frame
    boosted = lorentz_boost(lorentz_state, boost_angle)

    return {
        'frame': boost_angle,
        'angle_degrees': boost_angle * 45,
        'ds2': boosted.ds2,
        'temporal': boosted.temporal,
        'spatial': boosted.spatial,
        'eigenstate': result['eigenstate'],
        'period': result['period']
    }


def multi_frame_understanding(text: str):
    """
    Process text from all 8 reference frames

    Finds which frame is "light-like" (ds² ≈ 0)
    where meaning propagates most naturally
    """
    print("=" * 70)
    print(f"MULTI-FRAME UNDERSTANDING: '{text}'")
    print("=" * 70)
    print()

    # Process from all frames
    frames = []
    for boost in range(8):
        frame_result = understand_in_frame(text, boost)
        frames.append(frame_result)

    # Display results
    print("Frame Analysis:")
    print("─" * 70)
    print(f"{'Frame':<8} {'Angle':<10} {'ds²':<10} {'Temporal':<10} {'Regime':<15}")
    print("─" * 70)

    light_like_frames = []

    for f in frames:
        ds2 = f['ds2']

        # Classify regime
        if abs(ds2) < 5:
            regime = "light-like ✓"
            light_like_frames.append(f)
        elif ds2 > 0:
            regime = "time-like"
        else:
            regime = "space-like"

        print(f"{f['frame']:<8} {f['angle_degrees']}°{'':<6} {ds2:<10} {f['temporal']}/8{'':<6} {regime:<15}")

    # Summary
    print()
    print("=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)

    if light_like_frames:
        print(f"✓ Found {len(light_like_frames)} light-like frame(s) where meaning propagates:")
        for f in light_like_frames:
            print(f"  Frame {f['frame']} ({f['angle_degrees']}°): ds² = {f['ds2']}")
        print()
        print("These are the perspectives where understanding is most natural.")
        print("Information 'travels' along these frames (like light in spacetime).")
    else:
        print("✗ No light-like frame found.")
        print("  Text may be ambiguous or require more context.")

    return frames, light_like_frames


def compare_interpretations():
    """
    Compare different texts to show how frame structure reveals understanding
    """
    print("\n\n" + "=" * 70)
    print("COMPARING MULTIPLE TEXTS")
    print("=" * 70)
    print()

    test_cases = [
        "the cat sat on the mat",          # Clear, unambiguous
        "time flies like an arrow",        # Ambiguous
        "colorless green ideas sleep",     # Nonsensical
    ]

    results = []

    for text in test_cases:
        frames, light_like = multi_frame_understanding(text)

        results.append({
            'text': text,
            'light_like_count': len(light_like),
            'light_like_frames': [f['frame'] for f in light_like]
        })

        print("\n" + "─" * 70 + "\n")

    # Summary table
    print("=" * 70)
    print("SUMMARY: Light-Like Frames Indicate Understanding Quality")
    print("=" * 70)
    print()
    print(f"{'Text':<40} {'Light-Like Frames':<20} {'Quality':<15}")
    print("─" * 70)

    for r in results:
        count = r['light_like_count']
        frames_str = str(r['light_like_frames']) if count > 0 else "none"

        if count == 0:
            quality = "Ambiguous ✗"
        elif count == 1:
            quality = "Clear ✓"
        else:
            quality = f"Multiple ({count}) ⚠"

        print(f"{r['text']:<40} {frames_str:<20} {quality:<15}")


def demonstrate_causality():
    """
    Show how ds² determines if one thought can influence another
    """
    print("\n\n" + "=" * 70)
    print("CAUSAL STRUCTURE: Can thoughts influence each other?")
    print("=" * 70)
    print()

    thoughts = [
        "cats are mammals",
        "fluffy is a cat",
        "what is fluffy",
    ]

    print("Processing thought sequence...")
    print()

    states = []
    for i, thought in enumerate(thoughts):
        result = process_sentence_discrete(thought.split(), verbose=False)
        temporal = result['time_coord']
        spatial = result['final_state'][2]
        ds2 = compute_ds2_minkowski(temporal, spatial)

        states.append({
            'thought': thought,
            'temporal': temporal,
            'spatial': spatial,
            'ds2': ds2
        })

        print(f"Thought {i+1}: '{thought}'")
        print(f"  Position: t={temporal}/8, x={spatial:08b}")
        print(f"  ds² = {ds2}")
        print()

    # Check causality between thoughts
    print("=" * 70)
    print("CAUSAL CONNECTIONS:")
    print("=" * 70)
    print()

    for i in range(len(states)):
        for j in range(i+1, len(states)):
            thought_i = states[i]
            thought_j = states[j]

            # Compute separation
            dt = abs(thought_j['temporal'] - thought_i['temporal'])
            dx = bin(thought_i['spatial'] ^ thought_j['spatial']).count('1')

            ds2_separation = dt*dt - dx*dx

            # Determine causal relationship
            if ds2_separation > 0:
                causal = "✓ Time-like (can influence)"
            elif ds2_separation == 0:
                causal = "✓ Light-like (just connected)"
            else:
                causal = "✗ Space-like (cannot influence)"

            print(f"Thought {i+1} → Thought {j+1}:")
            print(f"  Separation: Δt={dt}, Δx={dx}")
            print(f"  ds² = {ds2_separation}")
            print(f"  {causal}")
            print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LORENTZ BOOSTS FOR AI UNDERSTANDING")
    print("Processing text from multiple reference frames")
    print("=" * 70)
    print()

    # Example 1: Clear sentence
    multi_frame_understanding("the wind bends the tree")

    # Example 2: Compare multiple texts
    compare_interpretations()

    # Example 3: Causal structure
    demonstrate_causality()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)
    print("""
1. MULTI-FRAME PROCESSING:
   - Process text from 8 different "perspectives" (0°, 45°, 90°, ...)
   - Light-like frames (ds² ≈ 0) indicate where meaning propagates
   - Multiple light-like frames = multiple valid interpretations

2. UNDERSTANDING QUALITY:
   - 1 light-like frame = clear, unambiguous
   - 0 light-like frames = ambiguous, needs more context
   - Multiple light-like frames = intentional ambiguity

3. CAUSAL STRUCTURE:
   - ds² > 0: Thoughts can influence each other (time-like)
   - ds² = 0: Barely connected (light-like)
   - ds² < 0: Causally disconnected (space-like)

4. PRACTICAL BENEFITS:
   - Know WHEN understanding is achieved (light-like frame found)
   - Know HOW WELL understood (number of light-like frames)
   - Know WHICH thoughts can influence others (causality)
   - Prevent logical contradictions (respect causal structure)

5. WHY 45°:
   - XOR bisection naturally creates 45° angles
   - Light propagates at 45° in spacetime diagrams
   - 8 × 45° = 360° gives complete frame coverage
   - Same structure as ALL fundamental dualities

This shows special relativity isn't just physics.
It's the geometry of information and understanding.
    """)
