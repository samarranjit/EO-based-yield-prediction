import warnings

import pytest

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def dummy_embed_dim():
    return 32
