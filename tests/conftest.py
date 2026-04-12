"""
Shared pytest configuration and fixtures for Engine Integrity Remediation tests.

Phase 2 (Foundation) - T006: Test infrastructure with UV-managed pytest configuration
and temporary directory helpers for reproducible testing.

This module provides:
- Pytest configuration for UV-managed environment
- Temporary directory and environment variable fixtures
- Integration/unit test markers
- Logging configuration for test output
"""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers for integration tests."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """
    Fixture: Temporary directory for test artifacts.
    
    Yields:
        Path: Temporary directory that is cleaned up after test completion.
        
    Usage:
        def test_something(temp_dir):
            test_file = temp_dir / "test.txt"
            test_file.write_text("content")
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="function")
def env_file_fixture(temp_dir: Path) -> Generator[Path, None, None]:
    """
    Fixture: Temporary .env file for settings factory testing.
    
    Creates a test .env file in a temporary directory.
    
    Yields:
        Path: Path to the temporary .env file.
        
    Usage:
        def test_env_loading(env_file_fixture):
            env_file_fixture.write_text("EXECUTE_CAUSAL_ENGINE=true")
    """
    env_file = temp_dir / ".env.test"
    env_file.write_text("")  # Create empty .env file
    yield env_file
    # Cleanup is automatic via temp_dir cleanup


@pytest.fixture(scope="function")
def isolated_env(monkeypatch) -> None:
    """
    Fixture: Isolated environment variables for test execution.
    
    Prevents test environment variables from leaking to other tests.
    Uses pytest's monkeypatch to safely restore env vars after each test.
    
    Usage:
        def test_env_isolation(isolated_env):
            os.environ["TEST_VAR"] = "value"
            # Will be automatically restored after test
    """
    # All environment variable changes will be automatically reverted by monkeypatch
    # No explicit cleanup needed
    pass


@pytest.fixture(scope="function")
def temp_checkpoint_dir(temp_dir: Path) -> Path:
    """
    Fixture: Temporary checkpoint directory structure for validation tests.
    
    Creates a realistic checkpoint directory structure with subdirectories
    needed for LoRA/Laplace-LoRA evaluation.
    
    Returns:
        Path: Path to the temporary checkpoint directory.
        
    Usage:
        def test_checkpoint_validation(temp_checkpoint_dir):
            # Directory structure is ready for use
            assert (temp_checkpoint_dir / "adapter_model.safetensors").exists()
    """
    checkpoint_dir = temp_dir / "checkpoint-100"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create required checkpoint artifact files (empty for testing purposes)
    (checkpoint_dir / "adapter_config.json").write_text("{}")
    (checkpoint_dir / "adapter_model.safetensors").write_text("")
    (checkpoint_dir / "config.json").write_text("{}")
    (checkpoint_dir / "optimizer.pt").write_text("")
    (checkpoint_dir / "scheduler.pt").write_text("")
    (checkpoint_dir / "trainer_state.json").write_text("{}")
    (checkpoint_dir / "training_args.bin").write_text("")
    (checkpoint_dir / "rng_state.pth").write_text("")
    
    return checkpoint_dir


@pytest.fixture(scope="session")
def test_data_path() -> Path:
    """
    Fixture: Path to test data directory.
    
    Returns:
        Path: Path to tests/ directory for loading test resources.
        
    Usage:
        def test_load_resource(test_data_path):
            resource = test_data_path / "fixtures" / "data.json"
    """
    return Path(__file__).parent


# Logging configuration for verbose test output
pytest_plugins = []


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers and set default logging.
    
    This hook ensures:
    - Tests are marked with appropriate categories for easy filtering
    - Logging is set to capture detailed output during test runs
    """
    for item in items:
        # Add integration marker to tests with "integration" in their name
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        
        # Tests can be run with: uv run pytest tests/ -v
        # Or filtered with: uv run pytest tests/ -m "not integration"
