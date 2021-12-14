import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD


class SoundNet(nn.Module):
    """
    Cette classe définit un réseau pleinement convolutionnel simple
    permettant de classifier des images satellite de Venus.
    This class defines a simple fully convolutional network
    to classify satellite images from Venus.
    """

    def __init__(self):
        super().__init__()
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
        # print(x.shape)
        y = F.relu(self.B1(self.C1(x)))
        # print(y.shape)
        y = F.relu(self.B2(self.C2(y)))
        # print(y.shape)
        y = F.relu(self.B3(self.C3(y)))
        # print(y.shape)
        y = F.relu(self.B4(self.C4(y)))
        # print(y.shape)
        y = F.avg_pool1d(self.B5(self.C5(y)), 2)
        # print(y.shape)
        y = self.L1(y.reshape(y.shape[0], y.shape[1]))
        # print(y.shape)
        # raise Exception
        return torch.sigmoid(y)


class SimpleSound(nn.Module):
    def __init__(self):
        super().__init__()
        self.C1 = nn.Conv1d(1, 32, (40,), stride=(10,), bias=False)
        self.B1 = nn.BatchNorm1d(32)
        self.C2 = nn.Conv1d(32, 64, (27,), stride=(10,), bias=False)
        self.B2 = nn.BatchNorm1d(64)
        self.L1 = nn.Linear(498, 64)
        self.D1 = nn.Dropout(p=0.5)
        self.L2 = nn.Linear(64, 64)
        self.D2 = nn.Dropout(p=0.25)
        self.L3 = nn.Linear(64, 1)
        self.D3 = nn.Dropout(p=0.25)
        self.L4 = nn.Linear(64, 1)

    def forward(self, x):
        # print(x.shape)
        y = F.relu(self.B1(self.C1(x)))
        # print(y.shape)
        y = F.relu(self.B2(self.C2(y)))
        # print(y.shape)
        y = F.relu(self.L1(y))
        # print(y.shape)
        y = F.relu(self.L2(self.D1(y)))
        # print(y.shape)
        y = F.relu(self.L3(self.D2(y)))
        # print(y.shape)
        y = F.relu(self.L4(self.D3(y.reshape(y.shape[0], y.shape[1]))))
        # print(y.shape)
        # raise Exception
        return torch.sigmoid(y)


def train_model(model, train_loader, validation_loader, device, nb_epoch, learning_rate, momentum):
    def compute_accuracy(dataloader):
        training_before = model.training
        model.eval()
        all_predictions = []
        all_targets = []

        for i_batch, batch in enumerate(dataloader):
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                predictions = model(images)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        if all_predictions[0].shape[-1] > 1:
            predictions_numpy = np.concatenate(all_predictions, axis=0)
            predictions_numpy = predictions_numpy.argmax(axis=1)
            targets_numpy = np.concatenate(all_targets, axis=0)
        else:
            predictions_numpy = np.concatenate(all_predictions).squeeze(-1)
            targets_numpy = np.concatenate(all_targets)
            predictions_numpy[predictions_numpy >= 0.5] = 1.0
            predictions_numpy[predictions_numpy < 0.5] = 0.0

        if training_before:
            model.train()

        return (predictions_numpy == targets_numpy).mean()

    model.train()

    model.to(device)

    criterion = torch.nn.BCELoss()

    optimiser = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scores = []
    # Boucle d'entraînement / Training loop
    for i_epoch in range(nb_epoch):

        start_time, train_losses = time.time(), []
        for i_batch, batch in enumerate(train_loader):
            sound, targets = batch
            targets = targets.type(torch.FloatTensor).unsqueeze(-1)

            sound = sound.to(device)
            targets = targets.to(device)

            optimiser.zero_grad()

            predictions = model(sound)
            loss = criterion(predictions, targets)

            loss.backward()
            optimiser.step()

            train_losses.append(loss.item())

        if i_epoch % 20 == 0:
            train_acc = 0
            test_acc = 0
            for _ in range(10):
                train_acc += compute_accuracy(train_loader)
                test_acc += compute_accuracy(validation_loader)
            scores.append((i_epoch, train_acc / 10, test_acc / 10))
        print(' [-] epoch {:4}/{:}, train loss {:.6f} in {:.2f}s'.format(
            i_epoch + 1, nb_epoch, np.mean(train_losses), time.time() - start_time))

    # Affichage du score en test / Display test score
    train_acc = 0
    test_acc = 0
    for _ in range(10):
        train_acc += compute_accuracy(train_loader)
        test_acc += compute_accuracy(validation_loader)
    scores.append((nb_epoch, train_acc / 10, test_acc / 10))
    return np.array(scores)