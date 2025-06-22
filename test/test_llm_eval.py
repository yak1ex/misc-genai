import pytest
import llm_eval


@pytest.mark.parametrize(
    'before,after,distance_default,distance_delete,distance_insert',
    [
        ('abcd',     '',             1.0, 1.0, 0.0),
        ('abcd',     'efgh',         2.0, 1.0, 1.0),
        ('abcd',     'abcd',         0.0, 0.0, 0.0),
        ('abcd',     'ABcd',         1.0, 0.5, 0.5),
        ('abcd',     'cdAB',         1.0, 0.5, 0.5),
        ('abcd',     'ab',           0.5, 0.5, 0.0),
        ('abcd',     'abcdef',       0.5, 0.0, 0.5),
        ('abcdefgh', '',             1.0, 1.0, 0.0),
        ('abcdefgh', 'ijklmnop',     2.0, 1.0, 1.0),
        ('abcdefgh', 'abcdefgh',     0.0, 0.0, 0.0),
        ('abcdefgh', 'ABcdEFgh',     1.0, 0.5, 0.5),
        ('abcdefgh', 'cdgABhEF',     1.0, 0.5, 0.5),
        ('abcdefgh', 'abcd',         0.5, 0.5, 0.0),
        ('abcdefgh', 'abcdefghijkl', 0.5, 0.0, 0.5),
    ]
)
def test_distance_default(before, after, distance_default, distance_delete, distance_insert):
    assert llm_eval.distance(before, after) == pytest.approx(distance_default)
    assert llm_eval.distance(before, after, 'default') == pytest.approx(distance_default)
    assert llm_eval.distance(before, after, 'delete') == pytest.approx(distance_delete)
    assert llm_eval.distance(before, after, 'insert') == pytest.approx(distance_insert)


@pytest.mark.parametrize(
    'before,after,penalty,exception,value',
    [
        ('', '', 'default', ValueError, "'before' MUST NOT be empty string"),
        ('', 'abc', 'default', ValueError, "'before' MUST NOT be empty string"),
        ('abc', 'def', '', ValueError, "'penalty' MUST BE one of 'default', 'insert' and 'delete'"),
        ('abc', 'def', 'invalid', ValueError, "'penalty' MUST BE one of 'default', 'insert' and 'delete'"),
        ('abc', 'def', 5, ValueError, "'penalty' MUST BE one of 'default', 'insert' and 'delete'"),
        ('abc', 'def', None, ValueError, "'penalty' MUST BE one of 'default', 'insert' and 'delete'"),
    ]
)
def test_distance_exception(before, after, penalty, exception, value):
    with pytest.raises(exception) as excinfo:
        llm_eval.distance(before, after, penalty)
    assert value in str(excinfo.value)


@pytest.mark.parametrize(
    'values,result',
    [
        ([], 0.0),
        ([2.0], 2.0),
        ([3.0, -1.0], 1.0),
        ([1.0, 2.0, 3.0, 4.0], 2.5),
    ]
)
def test_average(values, result):
    llm_eval.average(values) == pytest.approx(result)
