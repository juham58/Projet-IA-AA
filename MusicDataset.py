import numpy as np

import torch
import torchaudio.transforms as t

from torch.utils.data import Dataset
from torchaudio.backend.soundfile_backend import load


class MusicDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        file_names = data[:, 0]
        self.rng_gen = np.random.default_rng(0)
        self.data = []
        transform = t.Resample(44100, 10000)
        print("Chargement des données:")
        for i in range(file_names.shape[0]):
            self.data.append((transform(load("./MusicNet/Donnees/" + str(file_names[i]) + ".wav")[0]),
                              torch.tensor(data[i, 1])))
        self.targets = data[:, 1]
        print("Chargement terminé")

    def __getitem__(self, index):
        sound = self.data[index][0]
        author = self.data[index][1]
        size = self.data[index][0].shape[1]
        # debut = self.rng_gen.integers(0, size - 50002, 1)[0]
        # return sound[:, debut:debut+50000], author
        debut = self.rng_gen.integers(0, size - 100002, 1)[0]
        return sound[:, debut:debut + 100000], author

    def __len__(self):
        return self.targets.shape[0]
