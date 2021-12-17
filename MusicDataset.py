import numpy as np

import torch
import torchaudio.transforms as t

from torch.utils.data import Dataset
from torchaudio.backend.soundfile_backend import load


class MusicDataset(Dataset):
    def __init__(self, data, size):
        super().__init__()
        self.size = size
        self.rng_gen = np.random.default_rng(0)
        self.data = []
        transform = t.Resample(44100, 10000)
        for i in range(data.shape[0]):
            self.data.append((transform(load("./musicnet/data/" + str(data[i, 0]) + ".wav")[0]),
                              torch.tensor(data[i, 1])))
        self.targets = data[:, 1]

    def __getitem__(self, index):
        sound = self.data[index][0]
        author = self.data[index][1]
        size = self.data[index][0].shape[1]
        debut = self.rng_gen.integers(0, size - (self.size + 2), 1)[0]
        return sound[:, debut:debut + self.size], author

    def __len__(self):
        return self.targets.shape[0]
