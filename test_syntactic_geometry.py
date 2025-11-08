#!/usr/bin/env python3
"""
Test syntactic (intrinsic) geometry vs weighted (imposed) geometry

The key question: Does the sentence's own grammatical structure
create better measurement angles than our arbitrary weighting?
"""

import numpy as np
from src.eigen_text_core import (
    extract_LRV_from_sentence,  # Weighted approach
    extract_LRV_syntactic,       # Syntactic approach
    compute_M_geometric
)

print("=" * 80)
print("INTRINSIC GEOMETRY TEST: Does the sentence create its own angles?")
print("=" * 80)

test_sentences = [
    "The cat sat on the mat",
    "Water flows downhill due to gravity",
    "Light travels through space",
    "The apple fell from the tree",
    "Cat the mat on sat the",  # Scrambled - should have weird geometry
    "apple apple apple apple",  # Repetition - degenerate geometry
]

print("\n1. COMPARING L-R-V ORTHOGONALITY (closer to 90° = better basis)")
print("-" * 80)

for sent in test_sentences:
    print(f"\n'{sent}':")

    # Weighted approach
    triad_weighted = extract_LRV_from_sentence(sent, embedding_dim=100)
    LR_weighted = np.dot(triad_weighted.L, triad_weighted.R)
    LV_weighted = np.dot(triad_weighted.L, triad_weighted.V)
    RV_weighted = np.dot(triad_weighted.R, triad_weighted.V)

    # Syntactic approach
    try:
        triad_syntactic = extract_LRV_syntactic(sent, embedding_dim=100)
        LR_syntactic = np.dot(triad_syntactic.L, triad_syntactic.R)
        LV_syntactic = np.dot(triad_syntactic.L, triad_syntactic.V)
        RV_syntactic = np.dot(triad_syntactic.R, triad_syntactic.V)

        print(f"  WEIGHTED (imposed):   L·R={LR_weighted:.3f}, L·V={LV_weighted:.3f}, R·V={RV_weighted:.3f}")
        print(f"  SYNTACTIC (intrinsic): L·R={LR_syntactic:.3f}, L·V={LV_syntactic:.3f}, R·V={RV_syntactic:.3f}")

        # Ideal orthogonal basis would have all dot products = 0
        weighted_orthogonality = abs(LR_weighted) + abs(LV_weighted) + abs(RV_weighted)
        syntactic_orthogonality = abs(LR_syntactic) + abs(LV_syntactic) + abs(RV_syntactic)

        if syntactic_orthogonality < weighted_orthogonality:
            print(f"  → Syntactic is MORE orthogonal ({syntactic_orthogonality:.3f} vs {weighted_orthogonality:.3f}) ✓")
        else:
            print(f"  → Weighted is MORE orthogonal ({weighted_orthogonality:.3f} vs {syntactic_orthogonality:.3f})")

    except Exception as e:
        print(f"  SYNTACTIC failed: {e}")


print("\n" + "=" * 80)
print("2. WHAT DID SYNTACTIC EXTRACTION FIND?")
print("-" * 80)

for sent in test_sentences[:4]:  # Just the good sentences
    print(f"\n'{sent}':")
    doc = extract_LRV_syntactic._nlp(sent) if hasattr(extract_LRV_syntactic, '_nlp') else None

    try:
        # Import spacy to inspect parsing
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sent)

        subjects = [tok.text for tok in doc if "subj" in tok.dep_]
        verbs = [tok.text for tok in doc if tok.pos_ == "VERB"]
        objects = [tok.text for tok in doc if "obj" in tok.dep_ or tok.dep_ == "pobj"]

        print(f"  L (subject):  {subjects if subjects else '[fallback]'}")
        print(f"  R (verb):     {verbs if verbs else '[fallback]'}")
        print(f"  V (object):   {objects if objects else '[fallback]'}")
    except:
        pass


print("\n" + "=" * 80)
print("3. DOES SYNTACTIC GEOMETRY DISTINGUISH BETTER?")
print("-" * 80)

pairs = [
    ("Grammatical", "The cat sat on the mat"),
    ("Scrambled", "Cat the mat on sat the"),
]

print("\nUsing WEIGHTED approach:")
M_weighted = []
for name, sent in pairs:
    triad = extract_LRV_from_sentence(sent, embedding_dim=100)
    M = compute_M_geometric(triad.L, triad.R, triad.V)
    M_weighted.append(M)
    print(f"  {name}: {sent}")

similarity_weighted = np.dot(M_weighted[0], M_weighted[1])
print(f"  Similarity: {similarity_weighted:.6f}")

print("\nUsing SYNTACTIC approach:")
M_syntactic = []
for name, sent in pairs:
    try:
        triad = extract_LRV_syntactic(sent, embedding_dim=100)
        M = compute_M_geometric(triad.L, triad.R, triad.V)
        M_syntactic.append(M)
        print(f"  {name}: {sent}")
    except Exception as e:
        print(f"  {name} failed: {e}")
        M_syntactic.append(np.zeros(100))

if len(M_syntactic) == 2:
    similarity_syntactic = np.dot(M_syntactic[0], M_syntactic[1])
    print(f"  Similarity: {similarity_syntactic:.6f}")

    if similarity_syntactic < similarity_weighted:
        print(f"\n  ✓✓ SYNTACTIC distinguishes BETTER ({similarity_syntactic:.3f} < {similarity_weighted:.3f})")
    else:
        print(f"\n  Weighted still better ({similarity_weighted:.3f} < {similarity_syntactic:.3f})")


print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("Intrinsic geometry means the sentence tells us its own measurement angles")
print("via grammatical structure (subject-verb-object), rather than us imposing")
print("arbitrary weighting schemes.")
print("\nIf syntactic extraction produces more orthogonal L-R-V and better")
print("distinguishes grammatical from scrambled, then the sentence DOES have")
print("intrinsic geometry - like mass creating gravity.")
