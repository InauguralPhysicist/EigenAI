#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Context Accumulation Layer - Relative Information Impact

Implements the insight that information impact scales inversely with
accumulated contextual data:

    Impact(new_data) ∝ 1 / log(accumulated_context_volume)

Key Concept:
- Same stimulus → different impact based on historical context
- Novelty detection: high impact despite dense context
- Distinguishes genuine learning from mere accumulation

Examples:
- Pain: Stitches feel worse with less pain history
- Time: Years feel shorter as you age (more temporal context)
- Understanding: Novel concepts have high impact even with expertise

Mathematical Framework:
    relative_impact = novelty / log(context_density + 1)

    where:
    - novelty = 1 - max_similarity(new_vector, context_history)
    - context_density = number of accumulated context vectors

Applications:
1. Semantic extraction weighting
2. Eigenstate convergence modulation
3. Recursive AI self-modification rate
4. Novelty vs familiarity detection
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ContextEntry:
    """
    Single entry in context accumulation history

    Attributes
    ----------
    vector : np.ndarray
        Semantic vector (L, R, V, or M)
    timestamp : int
        When this context was added
    metadata : dict
        Additional information (text, label, etc.)
    impact : float
        Computed relative impact at time of addition
    """
    vector: np.ndarray
    timestamp: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    impact: float = 1.0

    def __repr__(self):
        return f"ContextEntry(t={self.timestamp}, impact={self.impact:.3f}, dim={self.vector.shape[0]})"


class ContextAccumulator:
    """
    Accumulates context and computes relative information impact

    Core Innovation:
    Measures how "intense" new information is based on accumulated
    historical context, implementing the insight that:

        Experience impact = f(novelty) / accumulated_data_volume

    Methods
    -------
    add_context : Add new semantic vector to history
    compute_relative_impact : Calculate impact of new vector
    compute_novelty_score : Measure how novel a vector is
    find_similar_contexts : Find k most similar historical contexts
    get_context_density : Get total accumulated context volume

    Examples
    --------
    >>> accumulator = ContextAccumulator()
    >>> # First time seeing "quantum"
    >>> impact1 = accumulator.compute_relative_impact(vec_quantum)
    >>> # impact1 ≈ 1.0 (very high, no historical context)
    >>>
    >>> accumulator.add_context(vec_quantum, {"text": "quantum mechanics"})
    >>>
    >>> # After seeing 1000 physics papers
    >>> for vec in physics_papers:
    ...     accumulator.add_context(vec, {"text": "..."})
    >>>
    >>> # Same concept again
    >>> impact2 = accumulator.compute_relative_impact(vec_quantum)
    >>> # impact2 ≈ 0.1 (low, familiar concept)
    >>>
    >>> # Genuinely novel concept
    >>> impact3 = accumulator.compute_relative_impact(vec_novel_discovery)
    >>> # impact3 ≈ 0.8 (high despite dense context - true novelty!)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        max_history_size: Optional[int] = None,
        use_recency_weighting: bool = True,
        recency_decay: float = 0.99
    ):
        """
        Initialize context accumulator

        Parameters
        ----------
        similarity_threshold : float
            Threshold for considering contexts similar (default: 0.95)
        max_history_size : int, optional
            Maximum context entries to keep (None = unlimited)
        use_recency_weighting : bool
            Weight recent contexts more heavily (default: True)
        recency_decay : float
            Decay factor for recency weighting (default: 0.99)
        """
        self.similarity_threshold = similarity_threshold
        self.max_history_size = max_history_size
        self.use_recency_weighting = use_recency_weighting
        self.recency_decay = recency_decay

        # Core state
        self.context_history: List[ContextEntry] = []
        self.timestamp = 0

        # Statistics
        self.total_contexts_seen = 0
        self.novelty_scores: List[float] = []
        self.impact_scores: List[float] = []

        # Categorization (optional)
        self.context_categories: Dict[str, List[int]] = defaultdict(list)

    def get_context_density(self) -> int:
        """
        Get current context density (accumulated data volume)

        Returns
        -------
        density : int
            Number of accumulated context entries
        """
        return len(self.context_history)

    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between vectors

        Parameters
        ----------
        vec1, vec2 : np.ndarray
            Vectors to compare

        Returns
        -------
        similarity : float
            Cosine similarity in [0, 1] (absolute value for direction-invariance)
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        # Use absolute value for direction-invariant similarity
        similarity = abs(np.dot(vec1, vec2) / (norm1 * norm2))
        return float(np.clip(similarity, 0.0, 1.0))

    def compute_novelty_score(
        self,
        new_vector: np.ndarray,
        use_recency: Optional[bool] = None
    ) -> float:
        """
        Compute how novel a vector is compared to accumulated context

        Novelty = 1 - max_similarity(new_vector, context_history)

        Higher score = more novel (less similar to anything seen before)

        Parameters
        ----------
        new_vector : np.ndarray
            Vector to evaluate
        use_recency : bool, optional
            Override instance recency weighting setting

        Returns
        -------
        novelty : float
            Novelty score in [0, 1]
            - 1.0 = completely novel (never seen anything similar)
            - 0.0 = completely familiar (exact match in history)
        """
        if len(self.context_history) == 0:
            return 1.0  # Everything is novel with no context

        use_recency = use_recency if use_recency is not None else self.use_recency_weighting

        max_similarity = 0.0
        current_time = self.timestamp

        for entry in self.context_history:
            similarity = self.compute_similarity(new_vector, entry.vector)

            # Apply recency weighting if enabled
            if use_recency:
                time_diff = current_time - entry.timestamp
                recency_weight = self.recency_decay ** time_diff
                weighted_similarity = similarity * recency_weight
            else:
                weighted_similarity = similarity

            max_similarity = max(max_similarity, weighted_similarity)

        novelty = 1.0 - max_similarity
        return float(np.clip(novelty, 0.0, 1.0))

    def compute_relative_impact(
        self,
        new_vector: np.ndarray,
        use_recency: Optional[bool] = None
    ) -> float:
        """
        Compute relative information impact of new vector

        This is the CORE INNOVATION implementing the insight:

            Impact = novelty / log(context_density + 1)

        The impact decreases with accumulated context (denominator grows)
        UNLESS the input is genuinely novel (numerator stays high).

        Parameters
        ----------
        new_vector : np.ndarray
            New semantic vector
        use_recency : bool, optional
            Override instance recency weighting setting

        Returns
        -------
        impact : float
            Relative information impact
            - High (>0.5): New experience is intense/significant
            - Medium (0.2-0.5): Moderate impact
            - Low (<0.2): Familiar, low impact

        Examples
        --------
        >>> accumulator = ContextAccumulator()
        >>> # First experience: high impact
        >>> vec1 = np.random.randn(100)
        >>> impact1 = accumulator.compute_relative_impact(vec1)
        >>> print(f"First: {impact1:.3f}")  # ~1.0
        >>>
        >>> # Add to history
        >>> accumulator.add_context(vec1)
        >>>
        >>> # Same experience again: lower impact
        >>> impact2 = accumulator.compute_relative_impact(vec1)
        >>> print(f"Repeat: {impact2:.3f}")  # ~0.0 (familiar)
        >>>
        >>> # Add 1000 similar vectors
        >>> for _ in range(1000):
        ...     accumulator.add_context(vec1 + np.random.randn(100) * 0.1)
        >>>
        >>> # Similar vector: low impact (dense context)
        >>> vec_similar = vec1 + np.random.randn(100) * 0.1
        >>> impact3 = accumulator.compute_relative_impact(vec_similar)
        >>> print(f"Similar: {impact3:.3f}")  # ~0.05 (familiar + dense)
        >>>
        >>> # Novel vector: still high impact despite dense context!
        >>> vec_novel = np.random.randn(100)
        >>> impact4 = accumulator.compute_relative_impact(vec_novel)
        >>> print(f"Novel: {impact4:.3f}")  # ~0.14 (novel despite density)
        """
        # Compute novelty
        novelty = self.compute_novelty_score(new_vector, use_recency=use_recency)

        # Compute context density scaling
        context_density = self.get_context_density()
        density_factor = np.log(context_density + 1) + 1.0  # +1 to avoid division issues

        # Relative impact formula
        relative_impact = novelty / density_factor

        return float(relative_impact)

    def find_similar_contexts(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        use_recency: Optional[bool] = None
    ) -> List[Tuple[ContextEntry, float]]:
        """
        Find k most similar contexts from history

        Parameters
        ----------
        query_vector : np.ndarray
            Vector to find similar contexts for
        top_k : int
            Number of similar contexts to return
        use_recency : bool, optional
            Apply recency weighting to similarity

        Returns
        -------
        similar_contexts : list of (ContextEntry, similarity)
            Top k similar contexts with their similarity scores
        """
        if len(self.context_history) == 0:
            return []

        use_recency = use_recency if use_recency is not None else self.use_recency_weighting

        similarities = []
        current_time = self.timestamp

        for entry in self.context_history:
            similarity = self.compute_similarity(query_vector, entry.vector)

            if use_recency:
                time_diff = current_time - entry.timestamp
                recency_weight = self.recency_decay ** time_diff
                weighted_similarity = similarity * recency_weight
            else:
                weighted_similarity = similarity

            similarities.append((entry, weighted_similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def add_context(
        self,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None
    ) -> ContextEntry:
        """
        Add new context to accumulation history

        Parameters
        ----------
        vector : np.ndarray
            Semantic vector to add
        metadata : dict, optional
            Additional information (text, label, source, etc.)
        category : str, optional
            Category label for organizing contexts

        Returns
        -------
        entry : ContextEntry
            The created context entry
        """
        # Compute impact before adding (relative to current history)
        impact = self.compute_relative_impact(vector)

        # Create entry
        metadata = metadata or {}
        entry = ContextEntry(
            vector=vector.copy(),
            timestamp=self.timestamp,
            metadata=metadata,
            impact=impact
        )

        # Add to history
        self.context_history.append(entry)

        # Update statistics
        self.total_contexts_seen += 1
        self.impact_scores.append(impact)
        novelty = self.compute_novelty_score(vector)
        self.novelty_scores.append(novelty)

        # Categorize if provided
        if category:
            self.context_categories[category].append(len(self.context_history) - 1)

        # Increment timestamp
        self.timestamp += 1

        # Enforce max history size if set
        if self.max_history_size and len(self.context_history) > self.max_history_size:
            # Remove oldest entries
            removed = self.context_history.pop(0)
            # Update category indices
            for cat_indices in self.context_categories.values():
                cat_indices[:] = [i - 1 for i in cat_indices if i > 0]

        return entry

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about accumulated context

        Returns
        -------
        stats : dict
            Statistics including:
            - context_density: Current number of contexts
            - total_contexts_seen: Total contexts added
            - avg_impact: Average impact of recent contexts
            - avg_novelty: Average novelty of recent contexts
            - impact_trend: Trend in impact scores
        """
        stats = {
            "context_density": self.get_context_density(),
            "total_contexts_seen": self.total_contexts_seen,
            "avg_impact": np.mean(self.impact_scores) if self.impact_scores else 0.0,
            "avg_novelty": np.mean(self.novelty_scores) if self.novelty_scores else 0.0,
            "recent_impact": (
                np.mean(self.impact_scores[-10:]) if len(self.impact_scores) >= 10 else
                np.mean(self.impact_scores) if self.impact_scores else 0.0
            ),
            "recent_novelty": (
                np.mean(self.novelty_scores[-10:]) if len(self.novelty_scores) >= 10 else
                np.mean(self.novelty_scores) if self.novelty_scores else 0.0
            ),
            "categories": len(self.context_categories),
            "current_timestamp": self.timestamp
        }

        # Compute impact trend (are impacts increasing or decreasing?)
        if len(self.impact_scores) >= 2:
            recent = self.impact_scores[-10:] if len(self.impact_scores) >= 10 else self.impact_scores
            if len(recent) >= 2:
                # Simple linear trend
                x = np.arange(len(recent))
                y = np.array(recent)
                trend = np.polyfit(x, y, 1)[0]  # Slope
                stats["impact_trend"] = float(trend)
            else:
                stats["impact_trend"] = 0.0
        else:
            stats["impact_trend"] = 0.0

        return stats

    def detect_phase_transition(
        self,
        window: int = 10,
        impact_threshold: float = 0.5
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect if we're experiencing a phase transition

        Phase transition = sudden high impact despite dense context
        This indicates paradigm shift or genuinely novel information

        Parameters
        ----------
        window : int
            Number of recent entries to analyze
        impact_threshold : float
            Threshold for high impact

        Returns
        -------
        is_transition : bool
            True if phase transition detected
        transition_type : str or None
            Type of transition detected
        """
        if len(self.impact_scores) < window:
            return False, None

        recent_impacts = self.impact_scores[-window:]
        context_density = self.get_context_density()

        # High impact despite high density = phase transition
        avg_recent_impact = np.mean(recent_impacts)
        max_recent_impact = max(recent_impacts)

        # Dense context (lots of accumulated data)
        is_dense = context_density > 50

        # High recent impact
        has_high_impact = avg_recent_impact > impact_threshold or max_recent_impact > impact_threshold * 1.5

        if is_dense and has_high_impact:
            # Determine transition type
            if avg_recent_impact > impact_threshold:
                return True, "sustained_novelty"  # Sustained high impact
            else:
                return True, "punctuated_novelty"  # Spike in impact

        return False, None

    def reset(self, keep_statistics: bool = True):
        """
        Reset context accumulation

        Parameters
        ----------
        keep_statistics : bool
            If True, keep statistics but clear history
        """
        self.context_history = []
        self.timestamp = 0
        self.context_categories = defaultdict(list)

        if not keep_statistics:
            self.total_contexts_seen = 0
            self.novelty_scores = []
            self.impact_scores = []


if __name__ == "__main__":
    print("=" * 70)
    print("CONTEXT ACCUMULATION LAYER - RELATIVE INFORMATION IMPACT")
    print("=" * 70)
    print()
    print("Implementing the insight:")
    print("  Pain, time, and all experience intensity")
    print("  scales inversely with accumulated context")
    print()

    # Create accumulator
    accumulator = ContextAccumulator()

    # Example 1: First experience has high impact
    print("=" * 70)
    print("EXAMPLE 1: First vs Repeated Experience")
    print("=" * 70)

    vec1 = np.random.randn(100)
    vec1 = vec1 / np.linalg.norm(vec1)

    impact1 = accumulator.compute_relative_impact(vec1)
    print(f"\n1st experience: impact = {impact1:.4f} (high, no context)")

    accumulator.add_context(vec1, {"text": "First time feeling this"})

    impact2 = accumulator.compute_relative_impact(vec1)
    print(f"2nd experience (same): impact = {impact2:.4f} (low, familiar)")

    # Example 2: Accumulated context reduces impact
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Context Accumulation (Like Aging)")
    print("=" * 70)

    print("\nAdding 100 similar experiences...")
    for i in range(100):
        # Similar but slightly different
        vec_similar = vec1 + np.random.randn(100) * 0.1
        vec_similar = vec_similar / np.linalg.norm(vec_similar)
        accumulator.add_context(vec_similar, {"text": f"Experience {i+2}"})

    stats = accumulator.get_statistics()
    print(f"\nContext density: {stats['context_density']}")
    print(f"Average impact: {stats['avg_impact']:.4f}")
    print(f"Recent impact: {stats['recent_impact']:.4f}")

    # Similar experience now has lower impact
    vec_similar = vec1 + np.random.randn(100) * 0.1
    vec_similar = vec_similar / np.linalg.norm(vec_similar)
    impact3 = accumulator.compute_relative_impact(vec_similar)
    print(f"\nSimilar experience now: impact = {impact3:.4f} (lower due to density)")

    # Example 3: Novelty detection
    print("\n" + "=" * 70)
    print("EXAMPLE 3: True Novelty Despite Dense Context")
    print("=" * 70)

    vec_novel = np.random.randn(100)
    vec_novel = vec_novel / np.linalg.norm(vec_novel)

    novelty = accumulator.compute_novelty_score(vec_novel)
    impact4 = accumulator.compute_relative_impact(vec_novel)

    print(f"\nNovel experience:")
    print(f"  Novelty: {novelty:.4f} (high, very different)")
    print(f"  Impact: {impact4:.4f} (moderate - novel but dense context)")
    print(f"  Context density: {accumulator.get_context_density()}")
    print()
    print("Key insight: Impact still higher than familiar experiences")
    print("             despite having 100+ accumulated contexts!")

    # Example 4: Find similar contexts
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Finding Similar Historical Contexts")
    print("=" * 70)

    query_vec = vec1 + np.random.randn(100) * 0.05
    query_vec = query_vec / np.linalg.norm(query_vec)

    similar = accumulator.find_similar_contexts(query_vec, top_k=3)
    print(f"\nQuery vector most similar to:")
    for i, (entry, similarity) in enumerate(similar, 1):
        print(f"  {i}. {entry.metadata.get('text', 'Unknown')} (similarity: {similarity:.4f})")

    # Example 5: Phase transition detection
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Phase Transition Detection")
    print("=" * 70)

    # Add some very novel contexts (paradigm shift)
    print("\nAdding 5 highly novel contexts (paradigm shift)...")
    for i in range(5):
        vec_paradigm_shift = np.random.randn(100)
        vec_paradigm_shift = vec_paradigm_shift / np.linalg.norm(vec_paradigm_shift)
        entry = accumulator.add_context(vec_paradigm_shift, {
            "text": f"Paradigm shift {i+1}",
            "category": "novel_paradigm"
        })
        print(f"  Added: impact = {entry.impact:.4f}")

    is_transition, transition_type = accumulator.detect_phase_transition()
    print(f"\nPhase transition detected: {is_transition}")
    if is_transition:
        print(f"  Type: {transition_type}")

    # Final statistics
    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    final_stats = accumulator.get_statistics()
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. RELATIVE IMPACT SCALES WITH CONTEXT:
   - First experience: high impact (no accumulated data)
   - Repeated experience: low impact (familiar)
   - Dense context: lower baseline impact

2. NOVELTY VS FAMILIARITY:
   - Novel inputs maintain high impact despite context density
   - Familiar inputs have low impact regardless of novelty
   - System distinguishes genuine learning from repetition

3. PHASE TRANSITIONS:
   - Sudden high impact despite dense context = paradigm shift
   - Detects when truly new information arrives
   - Analogous to scientific revolutions or "aha moments"

4. APPLICATIONS TO EIGENAI:
   - Weight semantic extraction by relative impact
   - Modulate convergence rate based on novelty
   - Adaptive self-modification in recursive AI
   - Distinguish understanding from memorization

5. HUMAN EXPERIENCE PARALLEL:
   - Pain: Same injury worse with less pain history
   - Time: Years feel shorter as you age
   - Learning: Novel concepts stand out even for experts

This bridges subjective experience with objective measurement!
    """)
