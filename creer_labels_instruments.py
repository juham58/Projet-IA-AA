# créer les labels d'instruments pour chaque pièce
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.inf)

def instrument_vers_label(instrument, labels):
    if instrument == 1:
        labels[0] = 1
    if instrument == 7:
        labels[1] = 1
    if instrument == 41:
        labels[2] = 1
    if instrument == 42:
        labels[3] = 1
    if instrument == 43:
        labels[4] = 1
    if instrument == 44:
        labels[5] = 1
    if instrument == 61:
        labels[6] = 1
    if instrument == 69:
        labels[7] = 1
    if instrument == 71:
        labels[8] = 1
    if instrument == 72:
        labels[9] = 1
    if instrument == 74:
        labels[10] = 1
    return labels

def instrument_hachage(instrument):
    if instrument == 1:
        return 0
    if instrument == 7:
        return 1
    if instrument == 41:
        return 2
    if instrument == 42:
        return 3
    if instrument == 43:
        return 4
    if instrument == 44:
        return 5
    if instrument == 61:
        return 6
    if instrument == 69:
        return 7
    if instrument == 71:
        return 8
    if instrument == 72:
        return 9
    if instrument == 74:
        return 10

instruments = pd.DataFrame(columns=["ID", "Instruments"])
dir = Path.cwd()/"musicnet"/"labels"
files = dir.glob("*.csv")
instruments = np.array([]).reshape(0, 11)
matrice_piece = []
instruments_csv = pd.DataFrame(columns=["ID", "Instruments"])
for file in files:
    liste_instruments = []
    array_instruments = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    data = pd.read_csv(file)
    liste_des_comptes = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for instrument in data["instrument"]:
        if instrument not in liste_instruments:
            liste_instruments.append(instrument)
            liste_des_comptes[instrument_hachage(instrument)] += 1
        array_instruments = instrument_vers_label(instrument, array_instruments)
    matrice_piece.append(liste_des_comptes)
    liste_instruments.sort()
    Id = str(file)[-8:-4]
    instruments = np.vstack((instruments, array_instruments))
    instruments_csv = instruments_csv.append({"ID": int(Id), "Instruments": liste_instruments}, ignore_index=True)
np.save(Path.cwd()/"musicnet"/"instrument_labels", instruments)
instruments_csv.set_index("ID", inplace=True)
instruments_csv.to_csv(Path.cwd()/"musicnet"/"instrument_labels.csv")




M = np.array(matrice_piece)

frequence_instruments = np.sum(M,axis=0)




titre = ['piano',  'clavecin', 'violon', 'alto', 'violoncelle', 'contrebasse', 'cor  français', 'hautbois','basson', 'clarinette','flûte']

fig, subfig = plt.subplots(1, 1, tight_layout=True)
plt.bar(np.arange(frequence_instruments.shape[0]), frequence_instruments, color='b', width=0.25)
subfig.xaxis.set_ticks(np.arange(frequence_instruments.shape[0]), titre)
subfig.set_title("Nombre  d'occurences des instruments dans le jeux de données")
plt.setp(subfig.xaxis.get_ticklabels(), rotation="vertical")
plt.show()
