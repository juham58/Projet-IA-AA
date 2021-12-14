# créer les labels d'instruments pour chaque pièce
import pandas as pd
from pathlib import Path

instruments = pd.DataFrame(columns=["ID", "Instruments"])
dir = Path.cwd()/"musicnet"/"labels"
files = dir.glob("*.csv")
for file in files:
    liste_instruments = []
    data = pd.read_csv(file)
    for instrument in data["instrument"]:
        if instrument not in liste_instruments:
            liste_instruments.append(instrument)
    id = int(str(file)[-8:-4])
    instruments = instruments.append({"ID": int(str(file)[-8:-4]), "Instruments": liste_instruments}, ignore_index=True)

instruments.set_index("ID", inplace=True)
instruments.to_csv(Path.cwd()/"musicnet"/"instrument_labels.csv")
