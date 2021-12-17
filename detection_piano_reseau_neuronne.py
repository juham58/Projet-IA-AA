from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
import numpy as np
from matplotlib import pyplot as plt
import time
from torch.optim import SGD



def train_model(model,  train_loader, validation_loader, device, nb_epoch, learning_rate,momentum):
    def compute_accuracy(dataloader):
        # Cette fonction est tirée directement de la question 1 du devoir 4
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
        # This loop is to smooth out the curves since there is some stochasticity in how the scores are calculated
        train_acc += compute_accuracy(train_loader)
        test_acc += compute_accuracy(validation_loader)
    scores.append((nb_epoch, train_acc / 10, test_acc / 10))
    return np.array(scores)


file = np.load('pretrained_data.npy', allow_pickle=True).reshape(1)[0]



# Création du jeu de données utilisé pour la détection du piano contre tous.

class PianoDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data, self.targets = shuffle(np.array(file['data']), np.array(file['target']))
        self.norm = np.linalg.norm(self.data,axis=-1)
        self.data=[self.data[i]/self.norm[i] for i in range(len(self.data))]
        #print(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.targets.shape[0]


# Premier réseau à deux couches

class PianoNote1(nn.Module):

    def __init__(self):
        super(PianoNote1, self).__init__()


        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )


    def forward(self, x):


        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return torch.sigmoid(logits)


# Deuxième réseau à quatre couches

class PianoNote2(nn.Module):

    def __init__(self):
        super(PianoNote2, self).__init__()


        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )


    def forward(self, x):


        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return torch.sigmoid(logits)



train_set = PianoDataset()



num_items = len(train_set)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(train_set, [num_train, num_val])
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=10, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=10, shuffle=False)

# Premier modèle
piano = PianoNote1()

# Deuxième modèle
# piano = PianoNote2()



scores = train_model(piano,train_dl, val_dl,'cpu',70,0.01,0.9)
plt.plot(scores[:, 0], scores[:, 1], color="blue", label="Score en entrainement")
plt.plot(scores[:, 0], scores[:, 2], color="red", label="Score en validation")
plt.ylabel("Scores")
plt.xlabel("Nombre d\' époques")
plt.legend()
plt.title("Scores de PianoNote \n10 -> 512 Relu 512 -> 1")
plt.show()
