import os

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from matplotlib import pyplot as plt

from MusicDataset import MusicDataset
from Models import SoundNet, SimpleSound, train_model

os.environ["OMP_NUM_THREADS"] = "1"
DEVICE = str(torch.device('cuda')) if torch.cuda.is_available() else str(torch.device('cpu'))


def create_balanced_sampler(dataset):
    def make_weights_for_balanced_classes(images, n_classes):
        count = [0] * n_classes
        for item in images:
            count[item[1]] += 1
        weight_per_class = [0.] * n_classes
        N = float(sum(count))
        for i in range(n_classes):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(images)
        for idx, val in enumerate(images):
            weight[idx] = weight_per_class[val[1]]
        return weight

    n_classes = np.unique(dataset.targets)
    weights = make_weights_for_balanced_classes(dataset.data, len(n_classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    return sampler


def get_data():
    # This reads the metadata file and stores it in an array.
    # The array is then modified so that for each music file name,
    # the associated class is 1 if the compositor is Beethoven, 0 if it is not.
    path = "./MusicNet/musicnet_metadata.csv"
    raw_data = np.asarray(pd.read_csv(path))
    raw_data = raw_data[:, [0, 1]]
    for i in range(raw_data.shape[0]):
        if raw_data[i, 1] == "Beethoven":
            raw_data[i, 1] = 1
        else:
            raw_data[i, 1] = 0
    return raw_data


def compute_confusion_matrix(model, dataloader, device):
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

        predictions_numpy = np.concatenate(all_predictions)
        targets_numpy = np.concatenate(all_targets)

        matrix = np.zeros((2, 2))
        for i in range(targets_numpy.shape[0]):
            class_predicted = 0
            if predictions_numpy[i] >= 0.5:
                class_predicted = 1
            matrix[targets_numpy[i], class_predicted] += 1
        return matrix


if __name__ == '__main__':
    # Initialisation des paramètres d'entraînement
    # Paramètres recommandés:
    # - Nombre d'epochs (nb_epoch = 10)
    # - Taux d'apprentissage (learning_rate = 0.01)
    # - Momentum (momentum = 0.9)
    # - Taille du lot (batch_size = 32)
    #
    # Initialization of training parameters
    # Recommended parameters:
    # - Number of epochs (nb_epoch = 10)
    # - Learning rate (learning_rate = 0.01)
    # - Momentum (momentum = 0.9)
    # - Batch size (batch_size = 32)
    nb_epoch = 2000
    learning_rate = 0.005
    momentum = 0.9
    batch_size = 32

    sets = get_data()
    rng = np.random.default_rng(0)
    rng.shuffle(get_data())
    # Chargement des données d'entraînement et de test
    # Loading training and testing set
    train_set = MusicDataset(sets[:300])
    test_set = MusicDataset(sets[300:])

    # Création du sampler avec les classes balancées
    # Create the sampler with balanced classes
    balanced_train_sampler = create_balanced_sampler(train_set)
    balanced_test_sampler = create_balanced_sampler(test_set)

    # Création du dataloader d'entraînement
    # Create training dataloader
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=balanced_train_sampler)
    test_loader = DataLoader(test_set, batch_size=batch_size, sampler=balanced_test_sampler)

    model = SimpleSound()

    scores = train_model(model, train_loader, test_loader, DEVICE, nb_epoch, learning_rate, momentum, )
    print(' [-] test acc. {:.6f}%'.format(scores[-1, 2] * 100))

    # Affichage de la matrice de confusion / Display confusion matrix
    matrix = compute_confusion_matrix(model, test_loader, DEVICE)
    print(matrix)

    # Libère la cache sur le GPU *important sur un cluster de GPU*
    # Free GPU cache *important on a GPU cluster*
    torch.cuda.empty_cache()

    scores = np.array(scores)
    plt.plot(scores[:, 0], scores[:, 1], color="blue", label="Score en entrainement")
    plt.plot(scores[:, 0], scores[:, 2], color="red", label="Score en validation")
    plt.ylabel("Scores")
    plt.xlabel("Number of epoch")
    plt.legend()
    plt.title("Scores for SoundNet")
    plt.show()
