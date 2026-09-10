# Tobit Regression (MLE)

[![CI](https://github.com/nvikou/Econometric-Tobit-Regression-Model/actions/workflows/ci.yml/badge.svg)](https://github.com/nvikou/Econometric-Tobit-Regression-Model/actions/workflows/ci.yml)

Maximum-likelihood Tobit estimator for left-, right-, and doubly
censored outcomes, with a scikit-learn-style API
(`fit` / `predict` / `score`) and Wald inference from the observed
Fisher information.

## Why Tobit, not OLS

When \(y\) is censored (corner solutions, detection limits, truncated
hours, etc.), OLS on the observed sample is inconsistent for the
latent linear index. Tobit models the latent variable
\(y^* = x\beta + u\), \(u \sim \mathcal{N}(0,\sigma^2)\), and observes
only the censored mapping of \(y^*\). Estimation is by maximizing the
censored normal likelihood, not by least squares.

This repository implements that estimator in NumPy/SciPy — not a
wrapper around an external Tobit package.

## What is implemented

| Piece | Detail |
| --- | --- |
| Likelihood | Left / uncensored / right contributions |
| Optimization | BFGS on the negative log-likelihood |
| Gradient | Analytical score (not pure finite differences) |
| Inference | Observed Fisher via Hessian of the NLL → `cov_`, `stderr_`, \(z\), \(p\) |
| API | `TobitModel` mirrors common sklearn estimator patterns |
| Reference check | Affairs data; coefficients comparable to R `censReg` (see notebook) |
| Tests / CI | `pytest` on synthetic recovery + Affairs smoke; GitHub Actions |

Also stores OLS coefficients fitted on the same design for a quick
bias illustration — not as a competing estimator.

## Decisions and trade-offs

- **MLE over OLS on censored \(y\)** — required for consistency under
  the Tobit DGP; OLS is kept only as a didactic baseline.
- **Analytical gradient + BFGS** — faster and more stable than
  derivative-free methods on this smooth likelihood; BFGS’s internal
  `hess_inv` is *not* used for reported standard errors.
- **Observed Fisher from the NLL Hessian** — finite differences on the
  analytical score, then inverted. This is the usual MLE sandwich
  for asymptotically normal \(\hat\theta\) under correct
  specification. Expected information (outer product of scores) is
  not computed separately.
- **Unconstrained \(\sigma\) in the optimizer** — same pattern as many
  textbook implementations; callers should check `sigma_ > 0` after
  fit. A log-\(\sigma\) reparameterization would be a natural hardening
  step.
- **Scope** — classical Type I Tobit with normal errors. No hurdle /
  selection (Heckman), no heteroskedasticity, no panel random
  effects.

## Fisher inference

After `fit`, the observed information matrix is

\[
\mathcal{I}(\hat\theta)
= \nabla^2_{\theta} \bigl(-\ell(\hat\theta)\bigr),
\qquad
\widehat{\mathrm{Var}}(\hat\theta)
= \mathcal{I}(\hat\theta)^{-1}.
\]

Exposed attributes:

- `fisher_info_` — observed information
- `cov_` — parameter covariance
- `stderr_`, `zvalues_`, `pvalues_`
- `loglik_`, `params_`, `param_names_`

```python
print(model.summary())
# or
model.summary_frame()
```

`summary()` prints coefficients, standard errors, Wald \(z\)-tests,
and normal confidence intervals. If the information matrix is
numerically singular, a pseudo-inverse is used and a warning is
emitted.

## Install

```bash
pip install -r requirements.txt
```

Core runtime: `numpy`, `pandas`, `scipy`, `scikit-learn`.  
`matplotlib` / Jupyter are only needed for the notebooks.

## Tests

```bash
pytest
```

CI runs the same suite on Python 3.10 and 3.12 for every push/PR to
`main`.

## Quick start

```python
from moduletobit import TobitModel
import pandas as pd

# cens: -1 left-censored, 0 uncensored, 1 right-censored
model = TobitModel(fit_intercept=True)
model.fit(x, y, cens)

y_hat = model.predict(x)
print(model.summary())
```

Runnable example on the Affairs dataset:

```bash
python usage_tobit.py
```

Notebooks:

- `notebooks/tobit.ipynb` — synthetic recovery of known \(\beta\), then
  Affairs vs R `censReg`
- `notebooks/tobit_cour_pratique.ipynb` — worked course-style example

## Layout

```
moduletobit/
  tobit.py          # likelihood, score, Fisher, TobitModel
  __init__.py
data/
  tobit_data.txt    # Affairs (AER)
notebooks/
usage_tobit.py
requirements.txt
```

## References

- Tobin, J. (1958). Estimation of relationships for limited dependent
  variables. *Econometrica*.
- Standard censored-normal MLE; R reference implementation:
  [`censReg`](https://cran.r-project.org/package=censReg).

## License

MIT — see `LICENSE`.
