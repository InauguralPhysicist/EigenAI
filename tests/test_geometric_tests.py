"""
Tests for geometric property tests module.

Tests the test_rupert_property function and related geometric utilities.
"""

import numpy as np
import pytest
from src.eigen_geometric_tests import (
    check_rupert_property,
    create_unit_cube,
    create_cube,
)


class TestRupertProperty:
    """Tests for the Rupert property testing function."""

    def test_basic_cube(self):
        """Test that a basic unit cube can be tested for Rupert property."""
        vertices = create_unit_cube()
        attempts, has_passage = check_rupert_property(vertices, n_samples=100)

        # Should have made some attempts
        assert attempts > 0
        assert attempts <= 100

        # Result should be boolean
        assert isinstance(has_passage, bool)

    def test_custom_cube(self):
        """Test with a custom-sized cube."""
        vertices = create_cube(side_length=2.0, center=[0, 0, 0])
        attempts, has_passage = check_rupert_property(vertices, n_samples=50)

        assert attempts > 0
        assert attempts <= 50
        assert isinstance(has_passage, bool)

    def test_invalid_vertices_none(self):
        """Test that None vertices raises ValueError."""
        with pytest.raises(ValueError, match="vertices cannot be None or empty"):
            check_rupert_property(None, n_samples=10)

    def test_invalid_vertices_empty(self):
        """Test that empty vertices raises ValueError."""
        with pytest.raises(ValueError, match="vertices cannot be None or empty"):
            check_rupert_property(np.array([]), n_samples=10)

    def test_invalid_vertices_shape(self):
        """Test that incorrect vertex shape raises ValueError."""
        # Wrong dimensions (2D instead of 3D)
        vertices_2d = np.array([[0, 0], [1, 1]])
        with pytest.raises(ValueError, match="vertices must be shape"):
            check_rupert_property(vertices_2d, n_samples=10)

        # Wrong shape (1D array)
        vertices_1d = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="vertices must be shape"):
            check_rupert_property(vertices_1d, n_samples=10)

    def test_invalid_n_samples(self):
        """Test that invalid n_samples raises ValueError."""
        vertices = create_unit_cube()

        with pytest.raises(ValueError, match="n_samples must be positive"):
            check_rupert_property(vertices, n_samples=0)

        with pytest.raises(ValueError, match="n_samples must be positive"):
            check_rupert_property(vertices, n_samples=-10)

    def test_vertices_as_list(self):
        """Test that vertices can be provided as a list."""
        vertices_list = [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ]
        attempts, has_passage = check_rupert_property(vertices_list, n_samples=50)

        assert attempts > 0
        assert isinstance(has_passage, bool)

    def test_large_n_samples(self):
        """Test with a larger number of samples."""
        vertices = create_unit_cube()
        attempts, has_passage = check_rupert_property(vertices, n_samples=1000)

        # With more samples, we're more likely to find a passage
        assert attempts > 0
        assert attempts <= 1000
        assert isinstance(has_passage, bool)

    def test_deterministic_with_seed(self):
        """Test that results are reproducible with same random seed."""
        vertices = create_unit_cube()

        # First run
        np.random.seed(42)
        attempts1, has_passage1 = check_rupert_property(vertices, n_samples=100)

        # Second run with same seed
        np.random.seed(42)
        attempts2, has_passage2 = check_rupert_property(vertices, n_samples=100)

        # Results should be identical
        assert attempts1 == attempts2
        assert has_passage1 == has_passage2


class TestCreateCube:
    """Tests for cube creation utilities."""

    def test_create_unit_cube_shape(self):
        """Test that unit cube has correct shape."""
        vertices = create_unit_cube()
        assert vertices.shape == (8, 3)

    def test_create_unit_cube_centered(self):
        """Test that unit cube is centered at origin."""
        vertices = create_unit_cube()
        center = vertices.mean(axis=0)
        np.testing.assert_array_almost_equal(center, [0, 0, 0])

    def test_create_unit_cube_size(self):
        """Test that unit cube has correct size."""
        vertices = create_unit_cube()
        # Each coordinate should range from -0.5 to 0.5
        for i in range(3):
            assert vertices[:, i].min() == pytest.approx(-0.5)
            assert vertices[:, i].max() == pytest.approx(0.5)

    def test_create_cube_default(self):
        """Test create_cube with default parameters."""
        vertices = create_cube()
        assert vertices.shape == (8, 3)

        # Should be centered at origin
        center = vertices.mean(axis=0)
        np.testing.assert_array_almost_equal(center, [0, 0, 0])

    def test_create_cube_custom_size(self):
        """Test create_cube with custom size."""
        vertices = create_cube(side_length=4.0)
        assert vertices.shape == (8, 3)

        # Size should be 4.0 (ranging from -2 to 2)
        for i in range(3):
            assert vertices[:, i].min() == pytest.approx(-2.0)
            assert vertices[:, i].max() == pytest.approx(2.0)

    def test_create_cube_custom_center(self):
        """Test create_cube with custom center."""
        center = [1.0, 2.0, 3.0]
        vertices = create_cube(side_length=2.0, center=center)
        assert vertices.shape == (8, 3)

        # Should be centered at specified location
        computed_center = vertices.mean(axis=0)
        np.testing.assert_array_almost_equal(computed_center, center)

    def test_create_cube_all_custom(self):
        """Test create_cube with all custom parameters."""
        side_length = 3.0
        center = [5.0, -2.0, 1.0]
        vertices = create_cube(side_length=side_length, center=center)

        assert vertices.shape == (8, 3)

        # Check center
        computed_center = vertices.mean(axis=0)
        np.testing.assert_array_almost_equal(computed_center, center)

        # Check size
        half = side_length / 2.0
        for i in range(3):
            assert vertices[:, i].min() == pytest.approx(center[i] - half)
            assert vertices[:, i].max() == pytest.approx(center[i] + half)


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_very_small_cube(self):
        """Test with a very small cube."""
        vertices = create_cube(side_length=0.001)
        attempts, has_passage = check_rupert_property(vertices, n_samples=10)

        assert attempts > 0
        assert isinstance(has_passage, bool)

    def test_very_large_cube(self):
        """Test with a very large cube."""
        vertices = create_cube(side_length=1000.0)
        attempts, has_passage = check_rupert_property(vertices, n_samples=10)

        assert attempts > 0
        assert isinstance(has_passage, bool)

    def test_non_cube_polyhedron(self):
        """Test with a non-cube polyhedron (tetrahedron)."""
        # Simple tetrahedron
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, np.sqrt(3)/2, 0],
            [0.5, np.sqrt(3)/6, np.sqrt(2/3)],
        ])
        attempts, has_passage = check_rupert_property(vertices, n_samples=50)

        assert attempts > 0
        assert isinstance(has_passage, bool)

    def test_single_sample(self):
        """Test with just a single sample."""
        vertices = create_unit_cube()
        attempts, has_passage = check_rupert_property(vertices, n_samples=1)

        assert attempts == 1
        assert isinstance(has_passage, bool)
