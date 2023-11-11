from deepdiff import DeepDiff


def assert_almost_equal(actual, desired, decimal=3, exclude_paths=None, **kwargs):
    diff = DeepDiff(desired, actual, significant_digits=decimal, exclude_paths=exclude_paths, **kwargs)
    assert diff == {}, f"Actual and Desired Dicts are not Almost Equal."
