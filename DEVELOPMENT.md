# Development Guide for EigenAI

This guide explains how to set up and work on the EigenAI codebase.

## What Changed (Recent Improvements)

We've upgraded the repository with industry-standard practices:

### 1. Proper Python Package Structure
- Added `setup.py` - Makes the project installable with pip
- Added `__init__.py` files - Makes `src/` a proper Python package
- Added `pytest.ini` - Configuration for the test framework

### 2. Testing with pytest
- Converted all tests to use pytest (industry standard)
- Tests are cleaner and easier to run
- Added `requirements-dev.txt` for development dependencies

### 3. Better Error Handling
- Core functions now validate inputs
- Clear error messages instead of silent failures
- Proper exceptions (TypeError, ValueError) with helpful messages

## Getting Started

### Installation

```bash
# Install the package in development mode
pip install -e .

# Install development dependencies (pytest, black, etc.)
pip install -r requirements-dev.txt
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run a specific test file
pytest tests/test_core.py -v

# Run tests with coverage report
pytest --cov=src --cov-report=html
```

### Using the Package

After installation, you can import from anywhere:

```python
# Import from anywhere on your system
from src import RecursiveEigenAI, understanding_loop

# Or import specific functions
from src.eigen_text_core import extract_LRV_from_sentence

# Create an AI instance
ai = RecursiveEigenAI(embedding_dim=128)
ai.process("Cats are mammals")
response = ai.query("What is Fluffy?")
```

## Project Structure

```
EigenAI/
├── src/                    # Main source code (now a proper package!)
│   ├── __init__.py        # Package initialization (NEW!)
│   ├── eigen_text_core.py # Core text understanding
│   ├── eigen_recursive_ai.py
│   └── ...
├── tests/                  # Test suite
│   ├── __init__.py        # Makes tests a package (NEW!)
│   ├── test_core.py       # Now uses pytest! (UPDATED!)
│   └── ...
├── examples/               # Example scripts
│   ├── __init__.py        # Package marker (NEW!)
│   └── ...
├── setup.py               # Package configuration (NEW!)
├── pytest.ini             # Pytest configuration (NEW!)
├── requirements.txt       # Core dependencies
├── requirements-dev.txt   # Development dependencies (NEW!)
├── README.md              # Main documentation
├── UNDERSTANDING.md       # Theoretical foundation
└── USAGE.md              # Usage guide
```

## Code Quality

### Running Code Formatters

```bash
# Format code with black (once installed)
black src/ tests/

# Check code style
flake8 src/

# Type checking
mypy src/
```

### Writing Tests

Tests should be in `tests/` directory and start with `test_`:

```python
def test_my_feature():
    """Test description"""
    result = my_function(input_data)
    assert result == expected_output
```

pytest will automatically discover and run these tests.

## What the Error Handling Does

Before (silent failure):
```python
M = compute_M_geometric(L, R, V)  # Returns zeros if input is bad
```

After (clear error):
```python
M = compute_M_geometric(L, R, V)  # Raises ValueError with helpful message
# ValueError: L, R, V must have the same shape. Got L:(100,), R:(50,), V:(100,)
```

## Next Steps to Learn

1. **Virtual Environments**: Learn to use `venv` or `conda` to isolate dependencies
2. **Git Branches**: Create feature branches for new work
3. **CI/CD**: Set up GitHub Actions to run tests automatically
4. **Documentation**: Generate API docs with Sphinx
5. **Type Checking**: Use mypy to catch type errors before runtime

## Common Commands

```bash
# Run all tests
pytest -v

# Run tests with coverage
pytest --cov=src --cov-report=term

# Format code
black src/ tests/

# Install in development mode
pip install -e .

# Check what's installed
pip list | grep eigenai
```

## Getting Help

- Run `pytest --help` for testing options
- Check `README.md` for usage examples
- Read `UNDERSTANDING.md` for theoretical background

## Contributing

When adding new features:

1. Write tests first (test-driven development)
2. Add error handling to functions
3. Document with docstrings
4. Run pytest to ensure all tests pass
5. Format code with black

---

**You're doing great work! Keep learning and building!** 🚀
