import torch
import torch.nn as nn
import torch.nn.functional as f


class SoundNet5000(nn.Module):
    """
    Cette classe définit un réseau pleinement convolutionnel fortement inspiré de celui construit dans le devoir 4 Q1.
    Ce réseau prend des entrées de dimention 5000.
    """
    def __init__(self):
        super().__init__()
        self.name = "SoundNet5000"
        self.C1 = nn.Conv1d(1, 64, (25,), stride=(2,), bias=False)
        self.B1 = nn.BatchNorm1d(64)
        self.C2 = nn.Conv1d(64, 64, (25,), stride=(7,), bias=False)
        self.B2 = nn.BatchNorm1d(64)
        self.C3 = nn.Conv1d(64, 64, (9,), stride=(4,), bias=False)
        self.B3 = nn.BatchNorm1d(64)
        self.C4 = nn.Conv1d(64, 64, (9,), stride=(4,), bias=False)
        self.B4 = nn.BatchNorm1d(64)
        self.C5 = nn.Conv1d(64, 64, (9,), stride=(4,), bias=False)
        self.B5 = nn.BatchNorm1d(64)
        self.L1 = nn.Linear(64, 1)

    def forward(self, x):
        y = f.relu(self.B1(self.C1(x)))
        y = f.relu(self.B2(self.C2(y)))
        y = f.relu(self.B3(self.C3(y)))
        y = f.relu(self.B4(self.C4(y)))
        y = f.avg_pool1d(self.B5(self.C5(y)), 2)
        y = self.L1(y.reshape(y.shape[0], y.shape[1]))
        return torch.sigmoid(y)


class SoundNet50000(nn.Module):
    """
    Cette classe définit un réseau pleinement convolutionnel fortement inspiré de celui construit dans le devoir 4 Q1.
    Ce réseau prend des entrées de dimention 50000.
    """
    def __init__(self):
        super().__init__()
        self.name = "SoundNet50000"
        self.C1 = nn.Conv1d(1, 32, (25,), stride=(12,), bias=False)
        self.B1 = nn.BatchNorm1d(32)
        self.C2 = nn.Conv1d(32, 64, (25,), stride=(12,), bias=False)
        self.B2 = nn.BatchNorm1d(64)
        self.C3 = nn.Conv1d(64, 64, (9,), stride=(4,), bias=False)
        self.B3 = nn.BatchNorm1d(64)
        self.C4 = nn.Conv1d(64, 64, (9,), stride=(4,), bias=False)
        self.B4 = nn.BatchNorm1d(64)
        self.C5 = nn.Conv1d(64, 64, (9,), stride=(4,), bias=False)
        self.B5 = nn.BatchNorm1d(64)
        self.L1 = nn.Linear(64, 1)

    def forward(self, x):
        y = f.relu(self.B1(self.C1(x)))
        y = f.relu(self.B2(self.C2(y)))
        y = f.relu(self.B3(self.C3(y)))
        y = f.relu(self.B4(self.C4(y)))
        y = f.avg_pool1d(self.B5(self.C5(y)), 2)
        y = self.L1(y.reshape(y.shape[0], y.shape[1]))
        return torch.sigmoid(y)


class SoundNet100000(nn.Module):
    """
    Cette classe définit un réseau pleinement convolutionnel fortement inspiré de celui construit dans le devoir 4 Q1.
    Ce réseau prend des entrées de dimention 100000.
    """
    def __init__(self):
        super().__init__()
        self.name = "SoundNet100000"
        self.C1 = nn.Conv1d(1, 32, (25,), stride=(18,), bias=False)
        self.B1 = nn.BatchNorm1d(32)
        self.C2 = nn.Conv1d(32, 64, (25,), stride=(16,), bias=False)
        self.B2 = nn.BatchNorm1d(64)
        self.C3 = nn.Conv1d(64, 64, (9,), stride=(4,), bias=False)
        self.B3 = nn.BatchNorm1d(64)
        self.C4 = nn.Conv1d(64, 64, (9,), stride=(4,), bias=False)
        self.B4 = nn.BatchNorm1d(64)
        self.C5 = nn.Conv1d(64, 64, (9,), stride=(4,), bias=False)
        self.B5 = nn.BatchNorm1d(64)
        self.L1 = nn.Linear(64, 1)

    def forward(self, x):
        y = f.relu(self.B1(self.C1(x)))
        y = f.relu(self.B2(self.C2(y)))
        y = f.relu(self.B3(self.C3(y)))
        y = f.relu(self.B4(self.C4(y)))
        y = f.avg_pool1d(self.B5(self.C5(y)), 2)
        y = self.L1(y.reshape(y.shape[0], y.shape[1]))
        return torch.sigmoid(y)
