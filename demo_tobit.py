"""Short demo: Tobit MLE + Wald CIs on Affairs.

Runs in a few seconds. Prints a compact summary and writes
assets/demo_wald_ci.png for the README.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from moduletobit import TobitModel


ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'tobit_data.txt'
OUT = ROOT / 'assets' / 'demo_wald_ci.png'


def load_affairs():
    """Load Affairs and build Tobit design."""
    df = pd.read_table(DATA, sep=' ')
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
    return x, y, cens


def plot_wald_ci(model, path, alpha=0.05):
    """Horizontal coef plot with normal Wald intervals."""
    # Drop sigma: different scale than betas
    names = model.param_names_[:-1]
    coefs = model.params_[:-1]
    ses = model.stderr_[:-1]
    z = float(norm.ppf(1.0 - alpha / 2.0))
    lo = coefs - z * ses
    hi = coefs + z * ses

    y_pos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.axvline(0.0, color='#888888', lw=0.8)
    ax.hlines(y_pos, lo, hi, color='#1f4e79', lw=2)
    ax.plot(coefs, y_pos, 'o', color='#1f4e79', ms=6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Coefficient (95% Wald CI)')
    ax.set_title('Tobit MLE — Affairs (left-censored at 0)')
    ax.invert_yaxis()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    x, y, cens = load_affairs()
    model = TobitModel().fit(x, y, cens)
    print(model.summary())
    plot_wald_ci(model, OUT)
    print(f'\nWrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
