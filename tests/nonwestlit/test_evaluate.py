import pytest
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from nonwestlit.evaluate import evaluate
from tests.utils import assert_almost_equal


@pytest.fixture(scope="function")
def expected_out_single_label():
    return {
        "val_accuracy": 1.0,
        "val_f1_macro": 1.0,
        "val_f1_weighted": 1.0,
        "val_precision_0": 1.0,
        "val_recall_0": 1.0,
        "val_f1_0": 1.0,
        "mAP": 0.3333333333333333,
        "mAP_weighted": 1.0,
    }


@pytest.fixture(scope="function")
def expected_out_multi_label():
    return {
        "AP@0.5": 0.4,
        "AR@0.5": 1.0,
        "AF1@0.5": 0.5714285714285715,
        "PMP": 0.4030627705627706,
        "PMR": 0.9375,
        "PMF": 0.5500718725718724,
        "mAP": 0.2857142857142857,
        "mAP_weighted": 0.75,
    }


def _get_tokenizer_and_model(model_name, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Fix padding token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    return tokenizer, model


def test_evaluate_single_label(
    expected_out_single_label, _seed, _test_model_name, _test_singlelabel_data
):
    set_seed(_seed)
    data_path, num_labels = _test_singlelabel_data
    tokenizer, model = _get_tokenizer_and_model(_test_model_name, num_labels)
    actual = evaluate(
        model_name_or_path=model,
        tokenizer=tokenizer,
        data_path=data_path,
        num_labels=num_labels,
        task_type="sequence-classification",
        in_sample_batch_size=8,
        max_sequence_length=tokenizer.model_max_length,
    )
    assert_almost_equal(actual, expected_out_single_label)


def test_evaluate_multi_label(expected_out_multi_label, _seed, _test_model_name, _test_multilabel_data):
    set_seed(_seed)
    data_path, num_labels = _test_multilabel_data
    tokenizer, model = _get_tokenizer_and_model(_test_model_name, num_labels)
    actual = evaluate(
        model_name_or_path=model,
        tokenizer=tokenizer,
        data_path=data_path,
        num_labels=num_labels,
        task_type="multilabel-sequence-classification",
        in_sample_batch_size=8,
        max_sequence_length=tokenizer.model_max_length,
    )
    assert_almost_equal(actual, expected_out_multi_label)
