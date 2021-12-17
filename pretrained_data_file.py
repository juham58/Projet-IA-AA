from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
import numpy as np
from Models import train_model

file = np.load('pretrained_data.npy', allow_pickle=True).reshape(1)[0]
#
# data, target = shuffle(file['data'], file['target'])
#
# data_train = data[:300]
# target_train = target[:300]
#
# data_test = data[300:]
# target_test = target[300:]




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




class PianoNote(nn.Module):

    def __init__(self):
        super(PianoNote, self).__init__()


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


piano = PianoNote()

Sol = train_model(piano,train_dl,val_dl,'cpu',70,0.01,0.9)

print(Sol)

