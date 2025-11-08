"""
EigenAI: A framework for measuring AI understanding through eigenstate detection.

This package provides tools for analyzing semantic understanding through
geometric eigenstate detection in text and other domains.
"""

__version__ = "0.1.0"

# Core text understanding
from .eigen_text_core import (
    SemanticTriad,
    understanding_loop,
    extract_LRV_from_sentence,
    extract_LRV_syntactic,
    extract_LRV_syntactic_entropy_weighted,
    compute_M_geometric,
    compute_M_xor,
    detect_eigenstate,
    measure_understanding_change,
    SPACY_AVAILABLE,
)

# Recursive AI system
from .eigen_recursive_ai import (
    RecursiveEigenAI,
    RecursiveState,
)

# Discrete tokenization
from .eigen_discrete_tokenizer import (
    DiscreteToken,
    process_sentence_discrete,
    analyze_sentence,
    tokenize_word,
    detect_cycle_in_trajectory,
)

# Domain-specific modules (optional imports)
try:
    from .eigen_em_field import analyze_em_field
except ImportError:
    analyze_em_field = None

try:
    from .eigen_gravity_inertia import geodesic_trajectory
except ImportError:
    geodesic_trajectory = None

try:
    from .eigen_quantum_xp import evolve_wavefunction
except ImportError:
    evolve_wavefunction = None

# Make the most important classes/functions easily accessible
__all__ = [
    # Version
    "__version__",

    # Core classes
    "SemanticTriad",
    "RecursiveEigenAI",
    "RecursiveState",
    "DiscreteToken",

    # Main functions
    "understanding_loop",
    "process_sentence_discrete",
    "analyze_sentence",

    # Core utilities
    "extract_LRV_from_sentence",
    "extract_LRV_syntactic",
    "extract_LRV_syntactic_entropy_weighted",
    "compute_M_geometric",
    "compute_M_xor",
    "detect_eigenstate",
    "measure_understanding_change",
    "tokenize_word",
    "detect_cycle_in_trajectory",
    "SPACY_AVAILABLE",

    # Domain-specific
    "analyze_em_field",
    "geodesic_trajectory",
    "evolve_wavefunction",
]
