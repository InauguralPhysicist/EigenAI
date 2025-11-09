"""
Environment banner test - prints diagnostic info for CI debugging.

This test runs first and prints the Python version, platform, and
key dependency versions to help diagnose environment-specific issues.
"""

import sys
import platform

try:
    import numpy as np
except Exception:
    np = None

try:
    import spacy
except Exception:
    spacy = None


def test_env_banner():
    """Print environment information for CI debugging."""
    print("\n" + "=" * 60)
    print("ENVIRONMENT INFO")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"NumPy: {getattr(np, '__version__', 'not installed')}")
    print(f"spaCy: {getattr(spacy, '__version__', 'not installed')}")
    print("=" * 60 + "\n")
    assert True
