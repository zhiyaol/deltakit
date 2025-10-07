# (c) Copyright Riverlane 2020-2025.
"""Description of ``deltakit.explorer.analysis`` namespace here."""

from deltakit_explorer.analysis._analysis import (
    calculate_lambda_and_lambda_stddev, calculate_lep_and_lep_stddev,
    get_exp_fit, get_lambda_fit, compute_logical_error_per_round,LogicalErrorRatePerRoundResults)

# List only public members in `__all__`.
__all__ = [s for s in dir() if not s.startswith("_")]
