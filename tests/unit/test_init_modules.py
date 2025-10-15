"""
Tests for __init__.py modules to increase coverage.

These are simple tests that just import modules to ensure they load correctly
and increase coverage statistics.
"""

import pytest


def test_import_data_fetchers():
    """Test importing data.fetchers module."""
    import mtquant.data.fetchers
    assert mtquant.data.fetchers is not None


def test_import_data_storage():
    """Test importing data.storage module."""
    import mtquant.data.storage
    assert mtquant.data.storage is not None


def test_import_mcp_integration():
    """Test importing mcp_integration module."""
    import mtquant.mcp_integration
    assert mtquant.mcp_integration is not None


def test_import_risk_management():
    """Test importing risk_management module."""
    import mtquant.risk_management
    assert mtquant.risk_management is not None


def test_import_utils():
    """Test importing utils module."""
    import mtquant.utils
    assert mtquant.utils is not None


def test_import_agents():
    """Test importing agents module."""
    import mtquant.agents
    assert mtquant.agents is not None


def test_import_main_package():
    """Test importing main mtquant package."""
    import mtquant
    assert mtquant is not None
    assert hasattr(mtquant, '__version__')


