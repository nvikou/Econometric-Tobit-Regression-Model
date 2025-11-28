"""
Example usage of the TobitModel class.

This script demonstrates how to use the Tobit regression model
on the Affairs dataset from the AER package.
"""
import numpy as np
import pandas as pd
from moduletobit import TobitModel


def load_affairs_data():
    """Load and prepare the Affairs dataset."""
    print("Loading Affairs dataset...")
    df = pd.read_table('data/tobit_data.txt', sep=' ')
    
    # Convert categorical variables to numeric
    df.loc[df.gender == 'male', 'gender'] = 1
    df.loc[df.gender == 'female', 'gender'] = 0
    df.loc[df.children == 'yes', 'children'] = 1
    df.loc[df.children == 'no', 'children'] = 0
    df = df.astype(float)
    
    return df


def prepare_tobit_data(df):
    """Prepare features, target, and censoring indicator."""
    # Target variable
    y = df.affairs
    
    # Features (excluding some variables for simplicity)
    x = df.drop(['affairs', 'gender', 'education', 'children'], axis=1)
    
    # Censoring indicator: -1 for left-censored (affairs == 0)
    cens = pd.Series(np.zeros((len(y),)))
    cens[y == 0] = -1
    
    return x, y, cens


def main():
    """Main execution function."""
    # Load data
    df = load_affairs_data()
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}\n")
    
    # Prepare data for Tobit model
    x, y, cens = prepare_tobit_data(df)
    
    # Display censoring statistics
    print("Censoring distribution:")
    print(cens.value_counts())
    print(f"Proportion of censored observations: {(cens == -1).sum() / len(cens):.2%}\n")
    
    # Fit Tobit model
    print("Fitting Tobit model...")
    model = TobitModel(fit_intercept=True)
    model.fit(x, y, cens, verbose=False)
    
    # Display results
    print("\n" + "="*60)
    print("TOBIT MODEL RESULTS")
    print("="*60)
    print(f"\n{'Variable':<20} {'Coefficient':>15}")
    print("-"*40)
    
    for col, coef in zip(x.columns, model.coef_):
        print(f"{col:<20} {coef:>15.4f}")
    
    print(f"{'Intercept':<20} {model.intercept_:>15.4f}")
    print(f"{'Sigma':<20} {model.sigma_:>15.4f}")
    print("="*60)
    
    # Make predictions
    predictions = model.predict(x)
    print(f"\nPredictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    print(f"Actual values range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Compare with OLS coefficients
    print("\n" + "="*60)
    print("COMPARISON: TOBIT vs OLS COEFFICIENTS")
    print("="*60)
    print(f"\n{'Variable':<20} {'Tobit':>12} {'OLS':>12} {'Difference':>12}")
    print("-"*60)
    
    for col, tobit_coef, ols_coef in zip(x.columns, model.coef_, model.ols_coef_):
        diff = tobit_coef - ols_coef
        print(f"{col:<20} {tobit_coef:>12.4f} {ols_coef:>12.4f} {diff:>12.4f}")
    
    print(f"{'Intercept':<20} {model.intercept_:>12.4f} {model.ols_intercept:>12.4f} "
          f"{model.intercept_ - model.ols_intercept:>12.4f}")
    print("="*60)
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
