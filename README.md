# Projet d'Introduction à l'apprentissage automatique
## Musicnet

Les données utilisées proviennent du [jeu de données Musicnet sur Kaggle](https://www.kaggle.com/imsparsh/musicnet-dataset)

## Tâches à solutionner

### Classification d'instruments

#### Réseau neuronal à convolution
- Exécuter le fichier creer_labels_instruments.py
- Exécuter le fichier pretraitement_data.py (Attention: très long)
- Dans le fichier apprendre_instruments_pytorch.py:
  - Ajuster la fraction du dataset à utiliser avec l'argument fraction d'InstrumentsDataset.
  - Ajuster le nombre d'époques d'entraînement avec la varible num_epochs.
  - Lancer le programme.

### Classification de compositeurs
- Ajuster les paramètres des appels à la fonction "train" dans le fichier compositeur.py
- Lancer le programme. Cela produit les graphiques relatifs à la composition des données pour les compositeurs et fait l'entrainement de deux réseaux de neurones à convolution dont les graphiques de scores sont également produits. 
