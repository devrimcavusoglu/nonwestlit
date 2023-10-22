import pytest
from transformers import set_seed
from transformers.trainer_utils import TrainOutput

from src import OUTPUTS_DIR, TEST_DATA_DIR
from src.train_falcon import predict, train
from tests.utils import assert_almost_equal

TEST_MODEL_NAME = "albert-base-v2"
_SEED = 42


@pytest.fixture(scope="function")
def expected_out_train():
    return 0.5832080841064453


@pytest.fixture(scope="function")
def expected_out_predict():
    return [{"label": "LABEL_0", "score": 0.6958711743354797}]


def test_train(expected_out_train):
    set_seed(_SEED)
    actual: TrainOutput = train(
        model_name_or_path=TEST_MODEL_NAME,
        data_path=TEST_DATA_DIR / "toy_dataset.json",
        n_epochs=2,
        device="cpu",
        output_dir=OUTPUTS_DIR,
    )
    assert_almost_equal(actual.training_loss, expected_out_train)


def test_predict(expected_out_predict):
    set_seed(_SEED)
    actual = predict(TEST_MODEL_NAME, ["РЫБАчья ХижиНА нА. БЕРЕТАХЪ ПОРМАНДНИ."], device="cpu")
    assert_almost_equal(actual, expected_out_predict)
