# Release Checklist for EigenAI

This document provides a comprehensive checklist for preparing and releasing new versions of EigenAI.

## Pre-Release Preparation

### 1. Code Quality

- [ ] **Run all tests**: `python -m pytest tests/ -v`
  - All tests should pass
  - Address any skipped tests if they're critical
  - Fix any failing tests

- [ ] **Run code formatting**: `python -m black src/ tests/ examples/`
  - Standardize code style across the project
  - Currently 31 files need reformatting

- [ ] **Run linting**: `python -m flake8 src/ tests/ examples/ --max-line-length=100`
  - Address critical errors (E-codes)
  - Fix unused imports (F401)
  - Consider fixing other warnings

- [ ] **Run type checking**: `python -m mypy src/`
  - Address any type errors
  - Add type hints where missing

### 2. Documentation

- [ ] **Update README.md**
  - Verify all examples work
  - Check all links are valid
  - Update badges if needed
  - Verify installation instructions

- [ ] **Update CHANGELOG.md**
  - Document all changes since last release
  - Add release date
  - Follow [Keep a Changelog](https://keepachangelog.com/) format
  - Use semantic versioning for version numbers

- [ ] **Review UNDERSTANDING.md**
  - Ensure theoretical documentation is accurate
  - Update with any new discoveries
  - Check examples are correct

- [ ] **Review USAGE.md**
  - Verify API documentation matches code
  - Update examples if API changed
  - Add new features to usage guide

- [ ] **Review DEVELOPMENT.md**
  - Update contributing guidelines
  - Check setup instructions work

### 3. Version Management

- [ ] **Update version in setup.py**
  - Follow [Semantic Versioning](https://semver.org/)
  - MAJOR.MINOR.PATCH format
  - Update `version="X.Y.Z"`

- [ ] **Update version in CHANGELOG.md**
  - Change `[Unreleased]` to `[X.Y.Z] - YYYY-MM-DD`
  - Add new `[Unreleased]` section for future changes

- [ ] **Tag version in git** (after release)
  - `git tag -a vX.Y.Z -m "Release version X.Y.Z"`
  - `git push origin vX.Y.Z`

### 4. Testing

- [ ] **Run all examples**
  - `python examples/simple_demo.py`
  - `python examples/recursive_ai_demo.py`
  - `python examples/test_universal_pattern.py`
  - `python examples/integrated_text_eigenstates.py`
  - `python examples/measure_ai_understanding.py`
  - `python examples/lorentz_ai_understanding.py`
  - `python examples/test_discrete_oscillations.py`
  - `python examples/try_your_own.py`

- [ ] **Test package build**
  - `python -m build --sdist --wheel .`
  - Verify no errors during build
  - Check dist/ contains both .tar.gz and .whl

- [ ] **Test package installation** (in clean environment)
  ```bash
  python -m venv test_env
  source test_env/bin/activate
  pip install dist/eigenai-X.Y.Z-py3-none-any.whl
  python -c "from src.eigen_text_core import understanding_loop; print('✓ Import successful')"
  deactivate
  rm -rf test_env
  ```

### 5. Dependencies

- [ ] **Review requirements.txt**
  - Verify all dependencies are necessary
  - Check version constraints are appropriate
  - Test with minimum versions

- [ ] **Review requirements-dev.txt**
  - Ensure development tools are current
  - Test that dev environment works

### 6. Legal and Licensing

- [ ] **Verify LICENSE file**
  - Confirm MIT license is appropriate
  - Check copyright year is current

- [ ] **Check third-party licenses**
  - Ensure all dependencies have compatible licenses
  - Document any license requirements

### 7. GitHub Preparation

- [ ] **Ensure all changes are committed**
  - `git status` should show clean working tree
  - All changes should be on appropriate branch

- [ ] **GitHub Actions passing**
  - Check that CI/CD workflows pass
  - Fix any failing tests or quality checks

- [ ] **Create release notes**
  - Draft release notes on GitHub
  - Include highlights from CHANGELOG
  - Add upgrade instructions if needed

## Release Process

### PyPI Release (when ready)

1. [ ] **Create PyPI account** (if not already done)
   - Register at https://pypi.org/
   - Set up 2FA

2. [ ] **Create test PyPI account**
   - Register at https://test.pypi.org/
   - Test releases here first

3. [ ] **Install twine**
   ```bash
   pip install twine
   ```

4. [ ] **Test upload to Test PyPI**
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```

5. [ ] **Test install from Test PyPI**
   ```bash
   pip install --index-url https://test.pypi.org/simple/ eigenai
   ```

6. [ ] **Upload to PyPI**
   ```bash
   python -m twine upload dist/*
   ```

7. [ ] **Verify package on PyPI**
   - Check https://pypi.org/project/eigenai/
   - Verify README renders correctly
   - Test installation: `pip install eigenai`

### GitHub Release

1. [ ] **Create git tag**
   ```bash
   git tag -a vX.Y.Z -m "Release version X.Y.Z"
   git push origin vX.Y.Z
   ```

2. [ ] **Create GitHub release**
   - Go to Releases → Draft a new release
   - Select the tag you just created
   - Title: "Version X.Y.Z"
   - Description: Copy from CHANGELOG
   - Attach dist files (.tar.gz and .whl)
   - Publish release

3. [ ] **Announce release**
   - Update repository description if needed
   - Consider blog post or announcement
   - Share on relevant communities (if appropriate)

## Post-Release

- [ ] **Verify installation works**
  ```bash
  pip install eigenai
  python -c "from src.eigen_text_core import understanding_loop; print('✓')"
  ```

- [ ] **Test examples with released version**
  - Install from PyPI in clean environment
  - Run several examples
  - Ensure everything works as expected

- [ ] **Update documentation sites** (if any)
  - ReadTheDocs
  - GitHub Pages
  - Project website

- [ ] **Monitor for issues**
  - Watch GitHub issues
  - Check PyPI download stats
  - Respond to user feedback

- [ ] **Prepare for next release**
  - Add `[Unreleased]` section to CHANGELOG
  - Consider creating milestone for next version
  - Plan next features or fixes

## Version Numbering Guide

Following [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** (X.0.0): Incompatible API changes
  - Breaking changes to public API
  - Removal of deprecated features
  - Major restructuring

- **MINOR** (0.X.0): New features, backward compatible
  - New functionality added
  - New eigenstate types discovered
  - New measurement metrics
  - Deprecated features (but still work)

- **PATCH** (0.0.X): Bug fixes, backward compatible
  - Bug fixes
  - Performance improvements
  - Documentation updates
  - Minor code cleanup

## Current Status (v0.1.0 Preparation)

### Completed ✓
- [x] Setup.py updated with author information
- [x] CHANGELOG.md created
- [x] All tests passing (54 passed, 8 skipped)
- [x] All examples verified working
- [x] MANIFEST.in created
- [x] Package builds successfully
- [x] Release checklist documentation created

### Pending Work
- [ ] Run black formatting on all files
- [ ] Address flake8 linting issues (311 items)
- [ ] Review and update documentation for accuracy
- [ ] Consider fixing test warnings (return values)
- [ ] Set release date in CHANGELOG.md
- [ ] Create git tag for v0.1.0
- [ ] Build final distribution packages
- [ ] (Optional) Upload to PyPI when ready

### Code Quality Issues

**Black Formatting** (31 files need reformatting):
```bash
python -m black src/ tests/ examples/
```

**Flake8 Issues** (311 total):
- 56 line length violations (E501)
- 36 unused imports (F401)
- 98 f-strings without placeholders (F541)
- 49 import ordering issues (E402)
- Others: indentation, spacing, etc.

**Test Warnings** (7 warnings):
- Some test functions return values instead of None
- Division by zero warning in one test
- Non-critical, but should be addressed

## Notes

- This is a research project, so perfect code quality is less critical than correctness
- Focus on ensuring tests pass and examples work
- Documentation accuracy is crucial for research reproducibility
- Consider community feedback before major releases
- Keep the theoretical foundation solid

## Contact

For questions about the release process:
- Jon McReynolds
- mcreynolds.jon@gmail.com
