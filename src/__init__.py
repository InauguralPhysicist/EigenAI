"""
EigenAI: A framework for measuring AI understanding through eigenstate detection.

This package provides tools for analyzing semantic understanding through
geometric eigenstate detection in text and other domains.
"""

__version__ = "1.2.0"

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

# Context accumulation
from .eigen_context_accumulator import (
    ContextAccumulator,
    ContextEntry,
)

# Discrete tokenization
from .eigen_discrete_tokenizer import (
    DiscreteToken,
    process_sentence_discrete,
    analyze_sentence,
    tokenize_word,
    detect_cycle_in_trajectory,
)

# Logic gate eigenstate detection
from .eigen_logic_gate import (
    LogicState,
    eigengate,
    eigengate_with_components,
    simulate_eigengate_feedback,
    XOR,
    XNOR,
    OR,
    connect_to_eigenstate_framework,
)

# Geometric property tests
from .eigen_geometric_tests import (
    check_rupert_property,
    create_unit_cube,
    create_cube,
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
    "ContextAccumulator",
    "ContextEntry",
    "DiscreteToken",
    "LogicState",
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
    # Logic gate functions
    "eigengate",
    "eigengate_with_components",
    "simulate_eigengate_feedback",
    "XOR",
    "XNOR",
    "OR",
    "connect_to_eigenstate_framework",
    # Geometric tests
    "check_rupert_property",
    "create_unit_cube",
    "create_cube",
    # Domain-specific
    "analyze_em_field",
    "geodesic_trajectory",
    "evolve_wavefunction",
]
