# GitHub Actions Guide

## What is GitHub Actions?

GitHub Actions is like having a **robot assistant** that automatically checks your code every time you push changes. It's completely free for public repositories!

## What We Set Up

We created **2 automated workflows** for your repository:

### 1. Tests Workflow (`.github/workflows/tests.yml`)
- **Runs**: Every time you push code
- **What it does**:
  - Tests your code on Python 3.8, 3.9, 3.10, and 3.11
  - Runs all your pytest tests
  - Shows code coverage (how much of your code is tested)
  - Tells you if tests pass or fail

### 2. Code Quality Workflow (`.github/workflows/code-quality.yml`)
- **Runs**: Every time you push code
- **What it does**:
  - Checks if code is formatted nicely (Black)
  - Checks for common Python mistakes (flake8)
  - Helps keep code clean and consistent

## How to See the Results

After you push code to GitHub:

1. Go to your repository on GitHub
2. Click the "Actions" tab at the top
3. You'll see a list of all runs
4. Click on any run to see details
5. Green checkmark ✓ = everything passed
6. Red X ✗ = something failed

## What the Badges Show

On your README, you'll see badges like this:

![Tests](https://github.com/InauguralPhysicist/EigenAI/workflows/Tests/badge.svg)

- **Green badge** = All tests passing
- **Red badge** = Tests failing
- **Gray badge** = Tests running

## Example Workflow

```
You write code locally
     ↓
You commit: git commit -m "Add new feature"
     ↓
You push: git push
     ↓
GitHub Actions automatically:
  1. Downloads your code
  2. Sets up Python
  3. Installs dependencies
  4. Runs all tests
  5. Shows you results
     ↓
You see results in the Actions tab
```

## What Happens When Tests Fail?

If you push code that breaks tests:

1. You'll get an email from GitHub (if enabled)
2. The badge turns red
3. The Actions tab shows what failed
4. You can click to see exactly which test failed and why
5. Fix the problem and push again
6. GitHub Actions runs again automatically

## Common Scenarios

### Scenario 1: All Tests Pass
```
✓ Tests on Python 3.8 - Passed
✓ Tests on Python 3.9 - Passed
✓ Tests on Python 3.10 - Passed
✓ Tests on Python 3.11 - Passed
✓ Code Quality - Passed

Your code is good to go! 🎉
```

### Scenario 2: Test Fails on One Python Version
```
✓ Tests on Python 3.8 - Passed
✗ Tests on Python 3.9 - FAILED
✓ Tests on Python 3.10 - Passed
✓ Tests on Python 3.11 - Passed

Something in your code doesn't work on Python 3.9.
Click the failed run to see details.
```

### Scenario 3: Code Quality Issues
```
✓ Tests - All Passed
⚠ Code Quality - Warnings

Tests work, but code could be formatted better.
Run: black src/ tests/
Then push again.
```

## How to Temporarily Disable Actions

If you don't want actions to run on a specific branch:

1. The workflows are configured to run on `main`, `master`, and `claude/**` branches
2. To skip, use a different branch name pattern
3. Or add `[skip ci]` to your commit message:
   ```bash
   git commit -m "Work in progress [skip ci]"
   ```

## Viewing Detailed Logs

When a test fails:

1. Go to Actions tab
2. Click the failed run
3. Click the failed job
4. Expand each step to see what happened
5. The pytest output shows exactly which test failed

Example output:
```
FAILED tests/test_core.py::test_compute_M_geometric - ValueError: Cannot compute M...
```

This tells you:
- Which file: `tests/test_core.py`
- Which test: `test_compute_M_geometric`
- Why it failed: `ValueError: Cannot compute M...`

## Benefits of GitHub Actions

1. **Catch bugs early** - Know immediately if new code breaks something
2. **Test on multiple Python versions** - Make sure code works everywhere
3. **Code quality** - Keep code clean and consistent
4. **Professional** - Shows others you care about code quality
5. **Free** - No cost for public repositories
6. **Automatic** - No manual work needed

## Local Testing vs GitHub Actions

**Before pushing** (recommended):
```bash
# Run tests locally first
pytest -v

# Check formatting
black src/ tests/

# Then push
git push
```

**After pushing**:
- GitHub Actions runs automatically
- You get confirmation everything works on a clean system

## Advanced: Adding More Checks

You can add more automated checks:

- **Type checking**: Add mypy to check types
- **Security**: Add bandit to find security issues
- **Documentation**: Build docs automatically
- **Deploy**: Automatically publish to PyPI

Want to learn more about any of these? Just ask!

## Troubleshooting

### "Workflow not running"
- Check that you pushed to the right branch (main, master, or claude/*)
- Look at the "Actions" tab for any messages

### "Tests fail on GitHub but pass locally"
- Different Python version
- Missing dependency in requirements.txt
- Different operating system behavior

### "Actions taking too long"
- GitHub Actions is free but can be slower than local
- Usually takes 2-5 minutes to run all tests

---

**You now have professional continuous integration (CI) set up!** Every push automatically tests your code. This is what big tech companies use. 🚀
