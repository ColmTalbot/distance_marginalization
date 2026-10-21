"""Shared fixtures; enables SciPy array API support before SciPy is imported."""
import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import json
import warnings
from functools import lru_cache
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
PRIOR_NAMES = ["usf_100_5000", "usf_10_50000", "pl2_50_5000", "plm1_50_5000"]


def load_json(name):
    """Load a JSON fixture.

    Parameters
    ----------
    name : str
        Path relative to ``fixtures/``.

    Returns
    -------
    dict
    """
    with open(FIXTURES / name) as f:
        return json.load(f)


def load_prior(name):
    """Load a prior fixture.

    Parameters
    ----------
    name : str
        Prior name, e.g. ``"usf_100_5000"``.

    Returns
    -------
    dict
        With keys ``distance`` and ``ln_prior``.
    """
    return load_json(f"priors/{name}.json")


@lru_cache(maxsize=None)
def make_dm(name, phase, **kwargs):
    """Build (and cache) a :class:`DistanceMarginalization` for a fixture prior.

    Parameters
    ----------
    name : str
        Prior name.
    phase : bool
        Phase marginalization.
    **kwargs
        Constructor options.

    Returns
    -------
    DistanceMarginalization

    Raises
    ------
    FitWarning
        Promoted to an error: fixture priors must fit cleanly.
    """
    from distmarg import DistanceMarginalization

    pr = load_prior(name)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return DistanceMarginalization(pr["distance"], pr["ln_prior"], phase, **kwargs)


@pytest.fixture(params=PRIOR_NAMES)
def prior_name(request):
    """Parametrize over the four smooth fixture priors.

    Parameters
    ----------
    request : pytest.FixtureRequest

    Returns
    -------
    str
    """
    return request.param


@pytest.fixture(params=[False, True], ids=["nophase", "phase"])
def phase(request):
    """Parametrize over the phase-marginalization flag.

    Parameters
    ----------
    request : pytest.FixtureRequest

    Returns
    -------
    bool
    """
    return request.param
