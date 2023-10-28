import warnings

import pytest
from transformers import set_seed
from transformers.trainer_utils import TrainOutput

from nonwestlit import OUTPUTS_DIR, TEST_DATA_DIR
from nonwestlit.model_ops import train
from tests.utils import assert_almost_equal


@pytest.fixture(scope="function")
def expected_out_train():
    return 3.3881847858428955


@pytest.fixture(scope="function")
def expected_out_sequence_classifier_lora():
    return 8.697151184082031


@pytest.fixture(scope="function")
def expected_out_sequence_classifier_lora_4bit():
    return 0.0008904933929443359


def test_train_sequence_classifier(_seed, _test_model_name, expected_out_train):
    set_seed(_seed)
    actual: TrainOutput = train(
        model_name_or_path=_test_model_name,
        data_path=TEST_DATA_DIR / "toy_dataset.json",
        n_epochs=2,
        device="cpu",
        output_dir=OUTPUTS_DIR,
    )
    assert_almost_equal(actual.training_loss, expected_out_train)


def test_train_sequence_classifier_lora(_seed, _test_model_name, expected_out_sequence_classifier_lora):
    set_seed(_seed)
    actual: TrainOutput = train(
        model_name_or_path=_test_model_name,
        data_path=TEST_DATA_DIR / "toy_dataset.json",
        n_epochs=2,
        device="cpu",
        output_dir=OUTPUTS_DIR,
        adapter="lora",
        quantization="f16",  # use half-precision for cpu
    )
    assert_almost_equal(actual.training_loss, expected_out_sequence_classifier_lora)


def test_train_sequence_classifier_lora_4bit(
    _seed, _test_model_name, expected_out_sequence_classifier_lora_4bit
):
    set_seed(_seed)
    with pytest.warns(UserWarning, match="4 and 8 bit quantization"):
        actual: TrainOutput = train(
            model_name_or_path=_test_model_name,
            data_path=TEST_DATA_DIR / "toy_dataset.json",
            n_epochs=2,
            device="cpu",
            output_dir=OUTPUTS_DIR,
            quantization="4bit",
            adapter="lora",
            lora_target_modules=["c_attn"],  # cross attention module for GPT2
        )
    assert_almost_equal(actual.training_loss, expected_out_sequence_classifier_lora_4bit)
