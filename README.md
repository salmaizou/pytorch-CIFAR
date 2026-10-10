# Classification d'images CIFAR-10

Projet de classification des images du jeu de données CIFAR-10 avec un réseau de neurones convolutif écrit en PyTorch.

Le projet télécharge CIFAR-10, convertit ses lots en images organisées par classe, prépare les images en tenseurs, entraîne un CNN et mesure ses performances sur l'ensemble de test.

## Structure du projet

```text
src/
  load_data.py       Télécharge CIFAR-10 et convertit les lots en images
  explore_data.py    Vérifie et affiche des exemples des données
  preprocessing.py   Convertit une image RGB en tenseur PyTorch
  dataloader.py      Associe les images à leurs étiquettes et crée les lots
  model.py           Définit le réseau convolutif CIFAR10CNN
  train.py           Entraîne le modèle et enregistre ses poids
  evaluate.py        Évalue le modèle sur les images de test
data/                Données téléchargées et images converties (non versionnées)
models/              Poids du modèle entraîné (non versionnés)
```

## Installation

Depuis la racine du projet, crée et active un environnement virtuel si nécessaire :

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Installe les dépendances :

```powershell
python -m pip install -r requirements.txt
```

Le téléchargement utilise KaggleHub. Configure l'accès à Kaggle dans ton environnement selon les instructions officielles si KaggleHub te le demande. Ne publie jamais de jeton ou d'identifiant secret.

## Télécharger et préparer les données

Depuis la racine du projet :

```powershell
python -m src.load_data
```

Les images converties sont enregistrées dans `data/images/train` et `data/images/test`, avec un sous-dossier par classe. Le jeu comprend 50 000 images d'entraînement et 10 000 images de test.

Pour vérifier les fichiers et afficher des exemples :

```powershell
python -m src.explore_data
```

## Prétraitement

`preprocessing.py` convertit chaque image en tenseur de forme `[3, 32, 32]` (canaux rouge, vert, bleu ; hauteur ; largeur). Les valeurs RGB sont converties en nombres flottants et divisées par 255, donc elles se trouvent entre 0 et 1. Aucune moyenne n'est soustraite et aucun écart-type n'est appliqué.

## Entraînement

```powershell
python -m src.train
```

Par défaut, l'entraînement parcourt les 50 000 images pendant 10 époques, avec des lots de 64 images et l'optimiseur Adam. Le script choisit CUDA si un GPU compatible est disponible, sinon le CPU. Les poids sont enregistrés dans `models/cifar10_cnn.pth`.

Après un changement du prétraitement, réentraîne le modèle avant l'évaluation : les poids existants ont été appris avec l'ancienne représentation des pixels.

## Évaluation

Après l'entraînement :

```powershell
python -m src.evaluate
```

L'évaluation charge les poids enregistrés et calcule la perte et l'accuracy sur les 10 000 images de test. Ces images ne sont pas utilisées pour mettre à jour les poids.

## Données et fichiers générés

Les répertoires `data/`, `venv/`, les poids `models/*.pth` et les caches Python sont exclus de Git par `.gitignore`. Il faut télécharger les données et entraîner le modèle localement avant de lancer l'évaluation.