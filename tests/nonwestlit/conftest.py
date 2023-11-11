import pytest


@pytest.fixture(scope="package")
def _test_model_name():
    # No predefined pad_token in GPT2 and small model for test purposes.
    return "gpt2"


@pytest.fixture(scope="package")
def _seed():
    return 42
