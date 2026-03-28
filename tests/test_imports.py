import pytest
from ca_biositing.ai_exploration.sandbox_setup import CBORGLLM

def test_package_import():
    """Simple smoke test to ensure package is importable."""
    assert CBORGLLM is not None

def test_available_models():
    """Verify models list is accessible."""
    from ca_biositing.ai_exploration.sandbox_setup import AVAILABLE_MODELS
    assert "gemini-2.0-flash" in AVAILABLE_MODELS
