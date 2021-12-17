import random
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, random_split
import torch.nn.functional as F
from torch.nn import init

class InstrumentsDataset(Dataset):
    def __init__(self, fraction=1):
        super().__init__()
        self.data = []
        self.labels = np.array([]).reshape(0, 11)
        tous_labels = np.load(Path.cwd()/"musicnet"/"instrument_labels.npy")
        dir = Path.cwd()/"musicnet"/"data"
        files = dir.glob("*.npy")
        for i, file in tqdm(enumerate(files)):
            selecteur = random.randint(1, fraction)
            if selecteur == 1:
                fichier = np.load(file)
                self.data.append(fichier)
                self.labels = np.vstack((self.labels, tous_labels[i]))
                del fichier
        self.data = np.array(self.data)
        self.data = torch.as_tensor(self.data)[None, :]
        self.data = torch.permute(self.data, (1, 0, 2, 3))
        self.targets = torch.tensor(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.targets.shape[0]

train_set = InstrumentsDataset(fraction=1)
print("Données d'entraînement chargées")

num_items = len(train_set)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(train_set, [num_train, num_val])

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=15, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=15, shuffle=True)
print("Dataloaders créés")

class AudioClassifier (nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers = []

        # Premier bloc de convolutions avec Relu, Batchnorm et Dropout
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 9), stride=(2, 4), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Deuxième bloc
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(5, 9), stride=(2, 4), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Troisième bloc
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(5, 9), stride=(2, 4), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Quatrième bloc
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 3), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        self.dropout4 = nn.Dropout(0.15)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4, self.dropout4]

        # Cinquième bloc
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 3), padding=(1, 1))
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(128)
        self.dropout5 = nn.Dropout(0.15)
        init.kaiming_normal_(self.conv5.weight, a=0.1)
        self.conv5.bias.data.zero_()
        conv_layers += [self.conv5, self.relu5, self.bn5, self.dropout5]

        # Sixième bloc
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(3, 5), stride=(2, 3), padding=(1, 1))
        self.relu6 = nn.ReLU()
        self.bn6 = nn.BatchNorm2d(256)
        self.dropout6 = nn.Dropout(0.25)
        init.kaiming_normal_(self.conv6.weight, a=0.1)
        self.conv6.bias.data.zero_()
        conv_layers += [self.conv6, self.relu6, self.bn6, self.dropout6]

        # Septième bloc
        self.conv7 = nn.Conv2d(256, 512, kernel_size=(1, 5), stride=(2, 3), padding=(1, 1))
        self.relu7 = nn.ReLU()
        self.bn7 = nn.BatchNorm2d(512)
        self.dropout7 = nn.Dropout(0.25)
        init.kaiming_normal_(self.conv7.weight, a=0.1)
        self.conv7.bias.data.zero_()
        conv_layers += [self.conv7, self.relu7, self.bn7, self.dropout7]

        # Huitième bloc
        self.conv8 = nn.Conv2d(512, 1024, kernel_size=(1, 3), stride=(1, 2), padding=(1, 1))
        self.relu8 = nn.ReLU()
        self.bn8 = nn.BatchNorm2d(1024)
        self.dropout8 = nn.Dropout(0.25)
        init.kaiming_normal_(self.conv8.weight, a=0.1)
        self.conv8.bias.data.zero_()
        conv_layers += [self.conv8, self.relu8, self.bn8]

        # Classifieur linéaire
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=1024, out_features=11)

        self.conv = nn.Sequential(*conv_layers)
 
    def forward(self, x):
        x = self.conv(x)
        x = self.ap(x)

        x = x.view(x.shape[0], -1)
        x = self.lin(x)
        x = torch.sigmoid(x)
        return x

modele = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
modele = modele.to(device)
print("Modèle créé")
next(modele.parameters()).device

def inference(model, val_dl):
    correct_prediction = 0
    total_prediction = 0

    with torch.no_grad():
        vp, fp, vn, fn = 0, 0, 0, 0
        for data in val_dl:
            inputs, labels = data[0].to(device), data[1].to(device)

            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            outputs = model(inputs)
            outputs = torch.round(outputs)

            matrix = torch.zeros((2, 2))

            non_predictions = torch.where(outputs == 1, 0, 1)
            non_labels = torch.where(labels == 1, 0, 1)

            matrix[0][0] = torch.sum(torch.where(outputs+labels == 2, 1, 0))
            matrix[1][1] = torch.sum(torch.where(outputs+labels == 0, 1, 0))
            
            # Si les targets positives correspondent au contraire des predictions: faux négatif
            matrix[0][1] = torch.sum(torch.where(non_predictions+labels == 2, 1, 0))
            
            # Si les predictions positives correspondent au contraire des targets: faux positif
            matrix[1][0] = torch.sum(torch.where(outputs+non_labels == 2, 1, 0))

            correct_prediction += (outputs == labels).sum().item()
            total_prediction += outputs.shape[0]
            vp += matrix[0][0]
            fp += matrix[1][0]
            vn += matrix[1][1]
            fn += matrix[0][1]

        acc = correct_prediction/total_prediction
        precision = vp/(fp+vp)
        rappel = vp/(fn+vp)
        print(f'--VALIDATION-- Précision: {precision:.2f}, Rappel: {rappel:.2f}')
        return precision, rappel

def training(model, train_dl, num_epochs):
    liste_precision = []
    liste_precision_valid = []
    liste_rappel = []
    liste_rappel_valid = []
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=int(len(train_dl)), epochs=num_epochs, anneal_strategy='linear')
    for epoch in range(num_epochs):
        vp, fp, vn, fn = 0, 0, 0, 0
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        for i, data in tqdm(enumerate(train_dl)):
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            outputs = torch.round(outputs)

            matrix = torch.zeros((2, 2))
            non_predictions = torch.where(outputs == 1, 0, 1)
            non_labels = torch.where(labels == 1, 0, 1)

            matrix[0][0] = torch.sum(torch.where(outputs+labels == 2, 1, 0))
            matrix[1][1] = torch.sum(torch.where(outputs+labels == 0, 1, 0))
            
            # Si les targets positives correspondent au contraire des predictions: faux négatif
            matrix[0][1] = torch.sum(torch.where(non_predictions+labels == 2, 1, 0))
            
            # Si les predictions positives correspondent au contraire des targets: faux positif
            matrix[1][0] = torch.sum(torch.where(outputs+non_labels == 2, 1, 0))

            correct_prediction += (outputs == labels).sum().item()
            total_prediction += outputs.shape[0]
            vp += matrix[0][0]
            fp += matrix[1][0]
            vn += matrix[1][1]
            fn += matrix[0][1]
            print(" Précision:", matrix[0][0]/(matrix[0][0]+matrix[1][0]), " Rappel:", matrix[0][0]/(matrix[0][0]+matrix[0][1]))
            

        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        precision = vp/(fp+vp)
        rappel = vp/(fn+vp)
        liste_precision.append(precision)
        liste_rappel.append(rappel)
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Précision: {precision:.2f}, Rappel: {rappel:.2f}')
        precision_valid, rappel_valid = inference(modele, val_dl)
        liste_precision_valid.append(precision_valid)
        liste_rappel_valid.append(rappel_valid)

    print('Entraînement terminé\n')
    return liste_precision, liste_rappel, liste_precision_valid, liste_rappel_valid
  
num_epochs=18
print("Début de l'entraînement")
liste_precision, liste_rappel, liste_precision_valid, liste_rappel_valid = training(modele, train_dl, num_epochs)

torch.save(modele.state_dict(), Path.cwd()/"apprendre_instruments.pth")

print("Début de l'inférence finale")
inference(modele, val_dl)

plt.figure()
plt.plot(np.arange(num_epochs), liste_precision, color="blue", label="Précision en entraînement")
plt.plot(np.arange(num_epochs), liste_rappel, color="red", label="Rappel en entraînement")
plt.plot(np.arange(num_epochs), liste_precision_valid, color="blue", linestyle="dashed", label="Précision en validation")
plt.plot(np.arange(num_epochs), liste_rappel_valid, color="red", linestyle="dashed", label="Rappel en validation")
plt.xlabel("Epochs")
plt.ylabel("Précision et rappel")
plt.legend()
plt.grid()
plt.show()
