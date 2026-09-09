"""Tobit Regression Model Implementation.

This module provides a maximum-likelihood estimator for Tobit
(censored) regression models. Tobit models are useful for datasets
where the dependent variable is censored.
"""
import math
import warnings

import numpy as np
import pandas as pd
import scipy.stats
from scipy.optimize import minimize
from scipy.special import log_ndtr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def split_left_right_censored(x, y, cens):
    """Split data into left-, uncensored, and right-censored rows.

    Args:
        x: Feature matrix (pandas DataFrame or numpy array)
        y: Target variable (pandas Series or numpy array)
        cens: Censoring indicator (-1 left, 0 uncensored, 1 right)

    Returns:
        tuple: (xs, ys) lists of [left, uncensored, right] splits
    """
    counts = cens.value_counts()
    if -1 not in counts and 1 not in counts:
        warnings.warn(
            "No censored observations; use regression methods "
            "for uncensored data"
        )
    xs = []
    ys = []

    for value in [-1, 0, 1]:
        if value in counts:
            split = cens == value
            y_split = np.squeeze(y[split].values)
            x_split = x[split].values
        else:
            y_split, x_split = None, None
        xs.append(x_split)
        ys.append(y_split)
    return xs, ys


def tobit_neg_log_likelihood(xs, ys, params):
    """Negative log-likelihood for the Tobit model.

    Args:
        xs: Tuple of (x_left, x_mid, x_right) feature matrices
        ys: Tuple of (y_left, y_mid, y_right) target values
        params: Model parameters [coefficients..., sigma]

    Returns:
        float: Negative log-likelihood value
    """
    x_left, x_mid, x_right = xs
    y_left, y_mid, y_right = ys

    b = params[:-1]
    s = params[-1]

    to_cat = []

    cens = False
    if y_left is not None:
        cens = True
        left = (y_left - np.dot(x_left, b))
        to_cat.append(left)
    if y_right is not None:
        cens = True
        right = (np.dot(x_right, b) - y_right)
        to_cat.append(right)
    if cens:
        concat_stats = np.concatenate(to_cat, axis=0) / s
        log_cum_norm = scipy.stats.norm.logcdf(concat_stats)
        cens_sum = log_cum_norm.sum()
    else:
        cens_sum = 0

    if y_mid is not None:
        mid_stats = (y_mid - np.dot(x_mid, b)) / s
        mid = (
            scipy.stats.norm.logpdf(mid_stats)
            - math.log(max(np.finfo('float').resolution, s))
        )
        mid_sum = mid.sum()
    else:
        mid_sum = 0

    loglik = cens_sum + mid_sum

    return -loglik


def tobit_neg_log_likelihood_der(xs, ys, params):
    """Gradient of the Tobit negative log-likelihood.

    Args:
        xs: Tuple of (x_left, x_mid, x_right) feature matrices
        ys: Tuple of (y_left, y_mid, y_right) target values
        params: Model parameters [coefficients..., sigma]

    Returns:
        numpy.ndarray: Gradient vector
    """
    x_left, x_mid, x_right = xs
    y_left, y_mid, y_right = ys

    b = params[:-1]
    s = params[-1]

    beta_jac = np.zeros(len(b))
    sigma_jac = 0

    if y_left is not None:
        left_stats = (y_left - np.dot(x_left, b)) / s
        l_pdf = scipy.stats.norm.logpdf(left_stats)
        l_cdf = log_ndtr(left_stats)
        left_frac = np.exp(l_pdf - l_cdf)
        beta_left = np.dot(left_frac, x_left / s)
        beta_jac -= beta_left

        left_sigma = np.dot(left_frac, left_stats)
        sigma_jac -= left_sigma

    if y_right is not None:
        right_stats = (np.dot(x_right, b) - y_right) / s
        r_pdf = scipy.stats.norm.logpdf(right_stats)
        r_cdf = log_ndtr(right_stats)
        right_frac = np.exp(r_pdf - r_cdf)
        beta_right = np.dot(right_frac, x_right / s)
        beta_jac += beta_right

        right_sigma = np.dot(right_frac, right_stats)
        sigma_jac -= right_sigma

    if y_mid is not None:
        mid_stats = (y_mid - np.dot(x_mid, b)) / s
        beta_mid = np.dot(mid_stats, x_mid / s)
        beta_jac += beta_mid

        mid_sigma = (np.square(mid_stats) - 1).sum()
        sigma_jac += mid_sigma

    # Chain rule for sigma derivative
    combo_jac = np.append(beta_jac, sigma_jac / s)

    return -combo_jac


def tobit_observed_hessian(xs, ys, params, eps=1e-5):
    """Observed Hessian of the Tobit negative log-likelihood.

    Built by finite differences on the analytical gradient.
    Equals the observed Fisher information matrix at the MLE.

    Args:
        xs: Tuple of (x_left, x_mid, x_right) feature matrices
        ys: Tuple of (y_left, y_mid, y_right) target values
        params: Parameter vector at which to evaluate
        eps: Step size for finite differences

    Returns:
        numpy.ndarray: Symmetric Hessian matrix
    """
    params = np.asarray(params, dtype=float)
    n_params = len(params)
    hess = np.zeros((n_params, n_params))
    grad0 = tobit_neg_log_likelihood_der(xs, ys, params)

    for i in range(n_params):
        params_eps = params.copy()
        params_eps[i] += eps
        grad_eps = tobit_neg_log_likelihood_der(xs, ys, params_eps)
        hess[:, i] = (grad_eps - grad0) / eps

    return 0.5 * (hess + hess.T)


class TobitModel:
    """Tobit (censored) regression via maximum likelihood.

    Attributes:
        fit_intercept (bool): Whether to fit an intercept term
        coef_ (numpy.ndarray): Estimated slope coefficients
        intercept_ (float): Estimated intercept
        sigma_ (float): Estimated error standard deviation
        params_ (numpy.ndarray): Full MLE vector [b..., sigma]
        fisher_info_ (numpy.ndarray): Observed Fisher information
        cov_ (numpy.ndarray): Parameter covariance (Fisher inverse)
        stderr_ (numpy.ndarray): Standard errors
        zvalues_ (numpy.ndarray): z-statistics
        pvalues_ (numpy.ndarray): Two-sided normal p-values
        loglik_ (float): Maximized log-likelihood
        n_obs_ (int): Number of observations
        param_names_ (list): Names aligned with params_
        ols_coef_ (numpy.ndarray): OLS coefficients for comparison
        ols_intercept (float): OLS intercept for comparison
    """

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.ols_coef_ = None
        self.ols_intercept = None
        self.coef_ = None
        self.intercept_ = None
        self.sigma_ = None
        self.params_ = None
        self.fisher_info_ = None
        self.cov_ = None
        self.stderr_ = None
        self.zvalues_ = None
        self.pvalues_ = None
        self.loglik_ = None
        self.n_obs_ = None
        self.param_names_ = None
        self.n_cens_left_ = None
        self.n_cens_right_ = None
        self.n_uncensored_ = None

    def fit(self, x, y, cens, verbose=False):
        """Fit a maximum-likelihood Tobit regression.

        Args:
            x: DataFrame (n_samples, n_features)
            y: Series (n_samples,)
            cens: Series (-1 left, 0 uncensored, 1 right)
            verbose: Show optimizer diagnostics

        Returns:
            TobitModel: Fitted instance (self)
        """
        x_copy = x.copy()
        feature_names = list(x_copy.columns)

        if self.fit_intercept:
            x_copy.insert(0, 'intercept', 1.0)
            param_names = ['intercept'] + feature_names + ['sigma']
        else:
            x_copy.scale(with_mean=True, with_std=False, copy=False)
            param_names = feature_names + ['sigma']

        init_reg = LinearRegression(fit_intercept=False).fit(x_copy, y)
        b0 = init_reg.coef_
        y_pred = init_reg.predict(x_copy)
        resid = y - y_pred
        resid_var = np.var(resid)
        s0 = np.sqrt(resid_var)
        params0 = np.append(b0, s0)
        xs, ys = split_left_right_censored(x_copy, y, cens)

        result = minimize(
            lambda params: tobit_neg_log_likelihood(xs, ys, params),
            params0,
            method='BFGS',
            jac=lambda params: tobit_neg_log_likelihood_der(
                xs, ys, params
            ),
            options={'disp': verbose},
        )
        if verbose:
            print(result)

        params = np.asarray(result.x, dtype=float)
        self.params_ = params
        self.param_names_ = param_names
        self.n_obs_ = int(len(y))
        self.n_cens_left_ = int((cens == -1).sum())
        self.n_cens_right_ = int((cens == 1).sum())
        self.n_uncensored_ = int((cens == 0).sum())
        self.loglik_ = float(
            -tobit_neg_log_likelihood(xs, ys, params)
        )

        if self.fit_intercept:
            self.ols_intercept = b0[0]
            self.ols_coef_ = b0[1:]
            self.intercept_ = params[0]
            self.coef_ = params[1:-1]
        else:
            self.coef_ = params[:-1]
            self.intercept_ = 0.0
            self.ols_coef_ = b0
            self.ols_intercept = 0.0
        self.sigma_ = params[-1]

        self._compute_inference(xs, ys, params)
        return self

    def _compute_inference(self, xs, ys, params):
        """Fill Fisher information, cov, stderr, z and p-values."""
        fisher = tobit_observed_hessian(xs, ys, params)
        self.fisher_info_ = fisher

        try:
            cov = np.linalg.inv(fisher)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Fisher information singular; using pseudo-inverse"
            )
            cov = np.linalg.pinv(fisher)

        # Guard tiny negatives from numerical Hessian noise
        diag = np.diag(cov).copy()
        if np.any(diag < 0):
            warnings.warn(
                "Negative variance on diagonal; clipped to zero"
            )
            diag = np.maximum(diag, 0.0)
            cov = cov.copy()
            np.fill_diagonal(cov, diag)

        self.cov_ = cov
        self.stderr_ = np.sqrt(diag)

        with np.errstate(divide='ignore', invalid='ignore'):
            zvalues = params / self.stderr_
        zvalues = np.where(np.isfinite(zvalues), zvalues, np.nan)
        self.zvalues_ = zvalues
        self.pvalues_ = 2.0 * (
            1.0 - scipy.stats.norm.cdf(np.abs(zvalues))
        )

    def predict(self, x):
        """Predict using the Tobit latent mean Xb.

        Args:
            x: Feature matrix (DataFrame or array)

        Returns:
            numpy.ndarray: Predicted values
        """
        return self.intercept_ + np.dot(x, self.coef_)

    def score(self, x, y, scoring_function=mean_absolute_error):
        """Calculate prediction score.

        Args:
            x: Feature matrix
            y: True target values
            scoring_function: Metric (default MAE)

        Returns:
            float: Score value
        """
        y_pred = self.predict(x)
        return scoring_function(y, y_pred)

    def summary(self, alpha=0.05):
        """Return a text summary of MLE estimates and inference.

        Args:
            alpha: Significance level for Wald confidence intervals

        Returns:
            str: Formatted coefficient table and fit statistics

        Raises:
            ValueError: If the model has not been fitted
        """
        if self.params_ is None or self.stderr_ is None:
            raise ValueError("Model is not fitted yet; call fit()")

        z_crit = scipy.stats.norm.ppf(1.0 - alpha / 2.0)
        lower = self.params_ - z_crit * self.stderr_
        upper = self.params_ + z_crit * self.stderr_
        ci_low = f"{alpha / 2:.3f}"
        ci_high = f"{1.0 - alpha / 2:.3f}"

        lines = []
        lines.append("Tobit Model Results (MLE)")
        lines.append("=" * 78)
        lines.append(f"No. Observations:     {self.n_obs_}")
        lines.append(
            f"Uncensored / Left / Right: "
            f"{self.n_uncensored_} / "
            f"{self.n_cens_left_} / "
            f"{self.n_cens_right_}"
        )
        lines.append(f"Log-Likelihood:       {self.loglik_:.6f}")
        lines.append("-" * 78)
        header = (
            f"{'':<16}"
            f"{'coef':>10}"
            f"{'std err':>10}"
            f"{'z':>10}"
            f"{'P>|z|':>10}"
            f"{'[' + ci_low:>8}"
            f"{ci_high + ']':>9}"
        )
        lines.append(header)
        lines.append("-" * 78)

        for name, coef, se, z, p, lo, hi in zip(
            self.param_names_,
            self.params_,
            self.stderr_,
            self.zvalues_,
            self.pvalues_,
            lower,
            upper,
        ):
            lines.append(
                f"{name:<16}"
                f"{coef:>10.4f}"
                f"{se:>10.4f}"
                f"{z:>10.4f}"
                f"{p:>10.4f}"
                f"{lo:>9.4f}"
                f"{hi:>9.4f}"
            )

        lines.append("=" * 78)
        lines.append(
            "Covariance from observed Fisher information "
            "(Hessian of NLL)."
        )
        return "\n".join(lines)

    def summary_frame(self, alpha=0.05):
        """Return inference results as a DataFrame.

        Args:
            alpha: Significance level for Wald confidence intervals

        Returns:
            pandas.DataFrame: coef, stderr, z, pvalue, CI bounds
        """
        if self.params_ is None or self.stderr_ is None:
            raise ValueError("Model is not fitted yet; call fit()")

        z_crit = scipy.stats.norm.ppf(1.0 - alpha / 2.0)
        return pd.DataFrame(
            {
                'coef': self.params_,
                'std_err': self.stderr_,
                'z': self.zvalues_,
                'P>|z|': self.pvalues_,
                f'[{alpha / 2:.3f}': (
                    self.params_ - z_crit * self.stderr_
                ),
                f'{1.0 - alpha / 2:.3f}]': (
                    self.params_ + z_crit * self.stderr_
                ),
            },
            index=self.param_names_,
        )
