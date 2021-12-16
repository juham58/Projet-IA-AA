import pickle
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from tqdm import tqdm



data = []
labels = np.array([]).reshape(0, 11)
tous_labels = np.load(Path.cwd()/"musicnet"/"instrument_labels.npy")
dir = Path.cwd()/"musicnet"/"data"
files = dir.glob("*.npy")
compteur = 0
for i, file in tqdm(enumerate(files)):
    if compteur % 329 == 0:
        data.append(np.load(file))
        labels = np.vstack((labels, tous_labels[i]))
    compteur += 1
data = np.array(data)
print(data.shape)

print("Données chargées")
mlp = MLPClassifier()
multi_mlp = MultiOutputClassifier(mlp)
print("Entraînement commencé")
multi_mlp.fit(data, labels)
print("Entrainement terminé")
print("Score en entraînement: ", multi_mlp.score())
