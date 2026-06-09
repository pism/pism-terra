import pytest

from pism_terra.glacier.run import _nullable_string

@pytest.mark.parametrize(
    'argument_string,expected',
    [
        ('None', None),
        ('none', None),
        (' NONE ', None),
        ('foobar', 'foobar'),
    ],
)
def test_nullable_string(argument_string, expected):
    assert _nullable_string(argument_string) == expected
