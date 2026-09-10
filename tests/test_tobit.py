"""Tests for Tobit MLE and observed Fisher inference."""
import numpy as np
import pandas as pd
import pytest

from moduletobit import TobitModel


def _left_censored_design(n=2500, seed=42):
    """Simulate left-censored Tobit data at zero."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    intercept = 1.0
    beta = np.array([-0.5, 0.8])
    sigma = 1.0
    y_star = (
        intercept
        + beta[0] * x1
        + beta[1] * x2
        + rng.normal(0.0, sigma, size=n)
    )
    y = np.maximum(y_star, 0.0)
    cens = np.where(y == 0.0, -1.0, 0.0)
    x = pd.DataFrame({'x1': x1, 'x2': x2})
    return (
        x,
        pd.Series(y),
        pd.Series(cens),
        intercept,
        beta,
        sigma,
    )


def test_fit_recovers_synthetic_coefficients():
    x, y, cens, intercept, beta, sigma = _left_censored_design()
    model = TobitModel(fit_intercept=True).fit(x, y, cens)

    assert model.sigma_ == pytest.approx(sigma, abs=0.12)
    assert model.intercept_ == pytest.approx(intercept, abs=0.12)
    np.testing.assert_allclose(model.coef_, beta, atol=0.12)


def test_fisher_inference_attributes():
    x, y, cens, *_ = _left_censored_design(n=800, seed=7)
    model = TobitModel().fit(x, y, cens)

    n_params = len(model.params_)
    assert model.fisher_info_.shape == (n_params, n_params)
    assert model.cov_.shape == (n_params, n_params)
    assert model.stderr_.shape == (n_params,)
    assert model.zvalues_.shape == (n_params,)
    assert model.pvalues_.shape == (n_params,)
    assert np.all(model.stderr_ > 0)
    assert np.isfinite(model.loglik_)
    np.testing.assert_allclose(
        model.cov_ @ model.fisher_info_,
        np.eye(n_params),
        atol=1e-3,
    )


def test_summary_and_frame():
    x, y, cens, *_ = _left_censored_design(n=500, seed=3)
    model = TobitModel().fit(x, y, cens)

    text = model.summary()
    assert 'Tobit Model Results' in text
    assert 'sigma' in text

    frame = model.summary_frame()
    assert list(frame.index) == model.param_names_
    assert 'coef' in frame.columns
    assert 'std_err' in frame.columns


def test_predict_and_score():
    x, y, cens, *_ = _left_censored_design(n=400, seed=11)
    model = TobitModel().fit(x, y, cens)
    pred = model.predict(x)

    assert pred.shape == (len(y),)
    assert np.isfinite(pred).all()
    score = model.score(x, y)
    assert np.isfinite(score)


def test_affairs_smoke_order_of_magnitude():
    """Regression smoke test on Affairs (left-censored at 0)."""
    path = 'data/tobit_data.txt'
    df = pd.read_table(path, sep=' ')
    df['gender'] = (df['gender'] == 'male').astype(float)
    df['children'] = (df['children'] == 'yes').astype(float)
    df = df.astype(float)

    y = df['affairs']
    x = df.drop(
        ['affairs', 'gender', 'education', 'children'],
        axis=1,
    )
    cens = pd.Series(np.zeros(len(y)))
    cens[y == 0] = -1

    model = TobitModel().fit(x, y, cens)

    assert model.n_obs_ == 601
    assert model.n_cens_left_ == 451
    assert model.loglik_ == pytest.approx(-705.576, abs=0.5)
    assert model.sigma_ == pytest.approx(8.25, abs=0.5)
    assert model.intercept_ == pytest.approx(8.17, abs=1.0)
