# Tobit Regression Model 📊

Implementation d'un modèle de régression Tobit (régression censurée) utilisant l'estimation par maximum de vraisemblance en Python.

## Description

Ce projet fournit une implémentation complète du modèle de régression Tobit, utile pour les ensembles de données où la variable dépendante est censurée (tronquée). Le modèle Tobit est particulièrement pertinent pour :
- Les données avec censure à gauche (valeurs minimales tronquées)
- Les données avec censure à droite (valeurs maximales tronquées)
- Les données doublement censurées

## Structure du Projet

```
codespaces-jupyter/
├── moduletobit/
│   ├── __init__.py      # Initialisation du module
│   └── tobit.py         # Classe TobitModel
├── usage_tobit.py       # Script d'exemple d'utilisation
├── requirements.txt     # Dépendances Python
├── data/
│   └── tobit_data.txt   # Données d'exemple (Affairs dataset)
└── notebooks/
    └── tobit.ipynb      # Notebook Jupyter avec exemples et comparaisons
```

## Installation

Installez les dépendances requises :

```bash
pip install -r requirements.txt
```

## Utilisation

### Utilisation Basique

```python
from moduletobit import TobitModel
import pandas as pd
import numpy as np

# Préparer vos données
x = pd.DataFrame(...)  # Variables explicatives
y = pd.Series(...)     # Variable cible
cens = pd.Series(...)  # Indicateur de censure (-1: gauche, 0: non censuré, 1: droite)

# Ajuster le modèle
model = TobitModel(fit_intercept=True)
model.fit(x, y, cens, verbose=False)

# Faire des prédictions
predictions = model.predict(x)

# Afficher les coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Sigma:", model.sigma_)
```

### Exemples dans le Notebook

Le notebook `notebooks/tobit.ipynb` contient deux exemples complets :

1. **Données artificielles** : Récupération des vrais coefficients sur des données de régression censurées générées artificiellement
2. **Données réelles** : Analyse du dataset Affairs avec comparaison aux résultats du package R `censReg`

## Caractéristiques

- ✅ Estimation par maximum de vraisemblance avec optimisation BFGS
- ✅ Support de la censure à gauche, à droite et double
- ✅ Calcul analytique du gradient pour une optimisation efficace
- ✅ Comparaison avec les coefficients OLS
- ✅ Interface compatible avec scikit-learn
- ✅ Documentation complète avec docstrings
- ✅ Validation sur des données réelles (compatible avec R censReg)

## Dépendances

- numpy
- pandas
- scipy
- scikit-learn
- matplotlib (pour les visualisations dans le notebook)

## Références

Le modèle Tobit a été introduit par James Tobin en 1958. Cette implémentation suit la méthodologie standard d'estimation par maximum de vraisemblance avec l'hypothèse d'erreurs normalement distribuées.

## Licence

Voir le fichier LICENSE pour plus de détails.
