import pickle
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm


print("Prétraitement du dataset")
dir = Path.cwd()/"musicnet"/"data"
files = dir.glob("*.wav")
data = []
taille_max = 0
labels = pd.read_csv(Path.cwd()/"musicnet"/"instrument_labels.csv")
for file in tqdm(files):
    x, sr = librosa.load(file)
    #x = librosa.resample(x, sr, 10000)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    data.append(Xdb)
    if Xdb.shape[1] > taille_max:
        taille_max = Xdb.shape[1]
print("Padding+sauvegarde")
for i, file in enumerate(data):
    nom = labels.iloc[i]["ID"]
    file = np.pad(file, ((0, 0), (0, taille_max-file.shape[1])), "empty")
    np.save(Path.cwd()/"musicnet"/"data"/str(nom), file)

print("Fin du prétraitement")
