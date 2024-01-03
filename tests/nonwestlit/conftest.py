from typing import Tuple

import pytest

from nonwestlit import TEST_DATA_DIR


@pytest.fixture(scope="package")
def _test_model_name():
    # No predefined pad_token in GPT2 and small model for test purposes.
    return "gpt2"


@pytest.fixture(scope="package")
def _test_singlelabel_data() -> Tuple[str, int]:
    """Returns single-label test data path and num labels"""
    return (TEST_DATA_DIR / "toy_test_singlelabel.json").as_posix(), 3


@pytest.fixture(scope="package")
def _test_multilabel_data() -> Tuple[str, int]:
    """Returns multi-label test data path and num labels"""
    return (TEST_DATA_DIR / "toy_test_multilabel.json").as_posix(), 7


@pytest.fixture(scope="package")
def _seed():
    return 42
