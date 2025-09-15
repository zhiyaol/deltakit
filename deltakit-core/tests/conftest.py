# (c) Copyright Riverlane 2020-2025.
import numpy as np
import pytest

from pathlib import Path


@pytest.fixture(scope="session")
def random_generator():
    return np.random.default_rng()


@pytest.fixture(scope="session")
def git_root():
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def reference_data_dir(git_root: Path):
    return git_root / "tests" / "reference_data"
