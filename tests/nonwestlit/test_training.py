import pytest
from transformers import set_seed
from transformers.trainer_utils import TrainOutput

from nonwestlit import TEST_DATA_DIR
from nonwestlit.training import train
from tests.utils import assert_almost_equal


@pytest.fixture(scope="function")
def expected_out_train():
    return 3.3881847858428955


@pytest.fixture(scope="function")
def expected_out_sequence_classifier_lora():
    return 9.0419921875


@pytest.fixture(scope="function")
def expected_out_sequence_classifier_lora_4bit():
    return 0.0016488938126713037


def test_train_sequence_classifier(_seed, _test_model_name, expected_out_train, tmp_path):
    set_seed(_seed)
    output_dir = tmp_path / "test_train_dir"
    actual: TrainOutput = train(
        model_name_or_path=_test_model_name,
        data_path=TEST_DATA_DIR / "toy_dataset.json",
        output_dir=output_dir.as_posix(),
        num_train_epochs=2,
        use_cpu=True,
        num_labels=3,
        per_device_train_batch_size=2,
    )
    assert_almost_equal(actual.training_loss, expected_out_train)


def test_train_sequence_classifier_lora(
    _seed, _test_model_name, expected_out_sequence_classifier_lora, tmp_path
):
    set_seed(_seed)
    output_dir = tmp_path / "test_train_dir"
    actual: TrainOutput = train(
        model_name_or_path=_test_model_name,
        data_path=TEST_DATA_DIR / "toy_dataset.json",
        output_dir=output_dir.as_posix(),
        num_train_epochs=2,
        adapter="lora",
        fp16=True,  # use half-precision
        use_cpu=False,  # half-precision is not supported on CPU.
        num_labels=3,
        per_device_train_batch_size=2,
    )
    assert_almost_equal(actual.training_loss, expected_out_sequence_classifier_lora)


def test_train_sequence_classifier_lora_4bit(
    _seed, _test_model_name, expected_out_sequence_classifier_lora_4bit, tmp_path
):
    set_seed(_seed)
    output_dir = tmp_path / "test_train_dir"
    output_dir.mkdir()
    with pytest.warns(UserWarning, match="4 and 8 bit quantization"):
        actual: TrainOutput = train(
            model_name_or_path=_test_model_name,
            data_path=TEST_DATA_DIR / "toy_dataset.json",
            num_train_epochs=2,
            use_cpu=True,
            output_dir=output_dir.as_posix(),
            bnb_quantization="4bit",
            adapter="lora",
            lora_target_modules=["c_attn"],  # cross attention module for GPT2
            num_labels=3,
            per_device_train_batch_size=2,
        )
    assert_almost_equal(actual.training_loss, expected_out_sequence_classifier_lora_4bit)
