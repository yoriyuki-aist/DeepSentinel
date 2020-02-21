import pandas as pd
import pytest

from deep_sentinel import dataset


@pytest.mark.parametrize(
    "given,expected", [
        ({
             'a': [1, 2, 3],
         }, {
             'a': 2.0,
         }),
        ({
             'a': [1, 2, 3],
             'b': [3, 2, 1],
             'c': [1, 1, 1],
         }, {
             'a': 2.0,
             'b': 2.0,
             'c': 1.0
         })
    ]
)
def test_get_mean(given, expected):
    expected = pd.Series(expected)
    actual = dataset.get_mean(pd.DataFrame(given))
    pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize(
    "given,expected", [
        ({
             'a': [1, 2, 3],
         }, {
             'a': 1.0,
         }),
        ({
             'a': [1, 2, 3],
             'b': [3, 2, 1],
             'c': [1, 1, 1],
         }, {
             'a': 1.0,
             'b': 1.0,
             'c': 0.0
         }),
    ]
)
def test_get_std(given, expected):
    expected = pd.Series(expected)
    actual = dataset.get_std(pd.DataFrame(given))
    pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize(
    "given,mean_val,std_val,expected", [
        ({'a': [1, 2, 3]}, None, None, {'a': [-1.0, 0.0, 1.0]}),
        ({'a': [1, 2, 3]}, [3], [1], {'a': [-2.0, -1.0, 0.0]}),
        ({
             'a': [1, 2, 3],
             'b': [3, 2, 1],
             'c': [1, 3, 5],
         },
         None,
         None,
         {
             'a': [-1.0, 0.0, 1.0],
             'b': [1.0, 0.0, -1.0],
             'c': [-1.0, 0.0, 1.0]
         }),
    ]
)
def test_normalize(given, mean_val, std_val, expected):
    expected = pd.DataFrame(expected)
    if mean_val is not None:
        mean_val = pd.Series(dict(zip(given.keys(), mean_val)))
    if std_val is not None:
        std_val = pd.Series(dict(zip(given.keys(), std_val)))
    actual = dataset.normalize(pd.DataFrame(given), mean_val, std_val)
    pd.testing.assert_frame_equal(actual, expected, check_less_precise=True, check_names=False)
