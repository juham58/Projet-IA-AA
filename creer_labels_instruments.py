# créer les labels d'instruments pour chaque pièce
import pandas as pd
import numpy as np
from pathlib import Path

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

instruments = pd.DataFrame(columns=["ID", "Instruments"])
dir = Path.cwd()/"musicnet"/"labels"
files = dir.glob("*.csv")
instruments = np.array([]).reshape(0, 11)
for file in files:
    array_instruments = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    data = pd.read_csv(file)
    for instrument in data["instrument"]:
        array_instruments = instrument_vers_label(instrument, array_instruments)
    Id = str(file)[-8:-4]
    instruments = np.vstack((instruments, array_instruments))
np.save(Path.cwd()/"musicnet"/"instrument_labels", instruments)
#instruments.set_index("ID", inplace=True)
#instruments.to_csv(Path.cwd()/"musicnet"/"instrument_labels.csv")
