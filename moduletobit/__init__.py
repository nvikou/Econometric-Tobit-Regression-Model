"""Module Tobit - Tobit Regression Model Implementation.

This module provides a maximum-likelihood estimator for Tobit (censored) regression models.
"""

from .tobit import (
    TobitModel,
    split_left_right_censored,
    tobit_neg_log_likelihood,
    tobit_neg_log_likelihood_der,
    tobit_observed_hessian,
)

__all__ = [
    'TobitModel',
    'split_left_right_censored',
    'tobit_neg_log_likelihood',
    'tobit_neg_log_likelihood_der',
    'tobit_observed_hessian',
]
__version__ = '1.1.0'
