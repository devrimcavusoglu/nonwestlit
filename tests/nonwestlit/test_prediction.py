import pytest
from transformers import set_seed

from nonwestlit.prediction import predict
from tests.utils import assert_almost_equal


@pytest.fixture(scope="function")
def expected_out_predict():
    return [{"label": "LABEL_0", "score": 0.9946860074996948}]


def test_predict(expected_out_predict, _seed, _test_model_name):
    set_seed(_seed)
    actual = predict(_test_model_name, ["РЫБАчья ХижиНА нА. БЕРЕТАХЪ ПОРМАНДНИ."], device="cpu")
    assert_almost_equal(actual, expected_out_predict)
