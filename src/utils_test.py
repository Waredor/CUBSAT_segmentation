import pytest
import utils

@pytest.mark.parametrize("input", "expected", [
    ([], {})
])

def test_load_config