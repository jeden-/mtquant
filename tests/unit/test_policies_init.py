"""
Tests for mtquant.agents.policies module.

This tests the policies package __init__.py to increase coverage.
"""

import pytest
import mtquant.agents.policies


def test_policies_module_exists():
    """Test that policies module can be imported."""
    assert mtquant.agents.policies is not None


def test_policies_module_version():
    """Test that policies module has __version__."""
    assert hasattr(mtquant.agents.policies, '__version__')
    assert isinstance(mtquant.agents.policies.__version__, str)


def test_policies_module_all():
    """Test that policies module has __all__."""
    assert hasattr(mtquant.agents.policies, '__all__')
    assert isinstance(mtquant.agents.policies.__all__, list)

