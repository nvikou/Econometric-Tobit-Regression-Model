***

## 📚 1. Révision générale : Économétrie des variables qualitatives

### Qu'est-ce que c'est ?
L'économétrie des variables qualitatives traite des situations où la **variable dépendante (y)** n'est pas continue et normale, mais :
- **Dicotomique** (0 ou 1) : ex. être employé/chômeur
- **Polytomique** (plusieurs catégories) : ex. choix de transport (bus, train, voiture)
- **ordonnée** : ex. niveau de satisfaction (faible, moyen, élevé)
- **Censurée/limitée** : ex. dépenses nulles pour beaucoup de ménages 

### Les grands modèles (hiérarchie)

| Type de variable | Modèle principal | Quand l'utiliser |
|---|---|---|
| Binaire (0/1) | **Logit**, **Probit** | Probabilité d'un événement   |
| Multinomiale (non ordonnée) | Logit multinomial | Choix entre plusieurs options sans ordre   |
| Ordinale | Logit/Probit ordonné | Catégories ordonnées (ex. échelle 1-5)   |
| **Censurée** (continue mais bornée) | **Tobit** | Variable continue avec valeurs limitées à 0 ou autre seuil   |

**Prérequis** : Statistiques et économétrie niveau L3 (MCO, maximum de vraisemblance, distribution normale) 

***

## 🎯 2. Le modèle Tobit : explication détaillée

### 📌 Définition
Le **modèle Tobit** est une technique économétrique développée par **James Tobin en 1958** pour analyser des données où la **variable dépendante est censurée** (observable seulement dans un certain intervalle). 

### 🔍 Quand l'utiliser ?
Tu utilises Tobit quand :
- La variable dépendante est **continue** MAIS **bornée/censurée** à une extrémité
- Beaucoup d'observations ont la **même valeur limite** (souvent 0)
- Les MCO (Moindres Carrés Ordinaires) donnent des résultats **biaisés** 

### 📊 Exemples concrets pour ton cours

| Exemple | Pourquoi Tobit ? |
|---|---|
| **Dépenses en biens durables** | Beaucoup de ménages dépensent 0 €, mais ont une "propension à dépenser" latente  
| **Salaires avec salaire minimum** | Données bornées au salaire minimum   |
| **Demande d'automobiles** | Beaucoup n'achètent pas (0 €), mais la demande latente existe   |
| **Heures de travail** | Beaucoup travaillent 0 heure (chômeurs), mais ont une offre de travail latente |

### 🧠 Concept clé : la variable latente

Le Tobit suppose qu'il existe une **variable latente non observable** $y^*$ :

$$
y^* = X\beta + \varepsilon
$$

$$
y = \begin{cases}
y^* & \text{si } y^* > 0 \\
0 & \text{si } y^* \leq 0
\end{cases}
$$

- $y$ = variable observée (censurée)
- $y^*$ = variable latente (non observable)
- $X\beta$ = partie déterministe
- $\varepsilon \sim N(0, \sigma^2)$ 

### ⚙️ Estimation : Maximum de Vraisemblance

Contrairement aux MCO, le Tobit utilise le **maximum de vraisemblance** car :
- La distribution des erreurs est **normale** (hypothèse cruciale)
- On estime la relation linéaire sur la **variable latente**, pas sur la variable observée 

La fonction de vraisemblance combine :
- Probabilité que \( y = 0 \) (partie discrète)
- Densité de probabilité que \( y > 0 \) (partie continue)

### 📐 Effets marginaux (important pour l'interprétation)

Dans le Tobit, il faut distinguer **deux effets** :

1. **Effet sur la variable latente** $E[y^*|X]$ :

$$
\frac{\partial E[y^*|X]}{\partial X_j} = \beta_j
$$

2. **Effet sur la variable observée** $E[y|X]$ :

$$
\frac{\partial E[y|X]}{\partial X_j} = \beta_j \cdot \Phi\left(\frac{X\beta}{\sigma}\right)
$$

avec $\Phi$ la fonction de répartition de la normale standard.

**À retenir** : les coefficients $\beta$ ne sont pas directement interprétables comme dans les MCO !

### ⚠️ Hypothèses du Tobit

| Hypothèse | Conséquence si violée |
|---|---|
| Erreurs **normales** | Estimation biaisée   |
| **Homoscédasticité** | Erreurs-types incorrectes   |
| Relation **linéaire** sur \( y^* \) | Mauvaise spécification |

### 🔁 Variantes du Tobit

- **Tobit I** : Tobit simple (censure à un seuil) 
- **Tobit II** : Sample selection (Heckman)
- **Tobit III** : Censure avec erreur non normale
- **Tobit généralisé** : Censure à plusieurs seuils 

***

## 🆚 3. Différence clé : Probit/Logit vs Tobit

| Critère | Probit/Logit | Tobit |
|---|---|---|
| **Variable dépendante** | Binaire (0 ou 1)   | Continue mais censurée  |
| **Type de problème** | Probabilité d'un événement   | Données bornées/censurées   |
| **Distribution** | Normale (Probit) ou Logistique (Logit)   | Normale des erreurs   |
| **Estimation** | Maximum de vraisemblance   | Maximum de vraisemblance   |

**Exemple simple** :
- Probit : "Est-ce que tu achètes une voiture ?" (oui/non)
- Tobit : "Combien dépenses-tu pour une voiture ?" (0 € pour beaucoup, mais montant continu sinon) 



