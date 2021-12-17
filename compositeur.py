import os
import time

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD

from matplotlib import pyplot as plt
import matplotlib

from MusicDataset import MusicDataset
from Models import SoundNet5000, SoundNet50000

os.environ["OMP_NUM_THREADS"] = "1"
DEVICE = str(torch.device('cuda')) if torch.cuda.is_available() else str(torch.device('cpu'))
matplotlib.rcParams['figure.figsize'] = (14.0, 7.0)


def create_balanced_sampler(dataset):
    # Cette fonction est tirée directement de la question 1 du devoir 4
    def make_weights_for_balanced_classes(images, n_classes):
        count = [0] * n_classes
        for item in images:
            count[item[1]] += 1
        weight_per_class = [0.] * n_classes
        n = float(sum(count))
        for i in range(n_classes):
            weight_per_class[i] = n / float(count[i])
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
    path = "musicnet_metadata.csv"
    raw_data = np.asarray(pd.read_csv(path))
    raw_data = raw_data[:, [0, 1]]
    for i in range(raw_data.shape[0]):
        if raw_data[i, 1] == "Beethoven":
            raw_data[i, 1] = 1
        else:
            raw_data[i, 1] = 0
    return raw_data


def make_bar_plots():
    # This function shows plots of the sources and the transcribers of the data
    path = "musicnet_metadata.csv"
    raw_data = np.asarray(pd.read_csv(path))
    raw_data = raw_data[:, [0, 1, 5, 6]]

    authors = np.unique(raw_data[:, 1])
    n_authors = np.zeros(authors.shape)
    for i in range(authors.shape[0]):
        for j in range(raw_data.shape[0]):
            if raw_data[j, 1] == authors[i]:
                n_authors[i] += 1
    fig, subfig = plt.subplots(1, 1, tight_layout=True)
    plt.bar(np.arange(authors.shape[0]), n_authors, color='b', width=0.25)
    subfig.xaxis.set_ticks(np.arange(authors.shape[0]), authors)
    subfig.set_title("Nombre de données par compositeur")
    plt.setp(subfig.xaxis.get_ticklabels(), rotation="vertical")
    plt.show()

    list_source = np.unique(raw_data[:, 2])
    list_transcriber = np.unique(raw_data[:, 3])
    sources_beethoven = np.zeros(list_source.shape[0])
    sources_other = np.zeros(list_source.shape[0])
    transcriber_beethoven = np.zeros(list_transcriber.shape[0])
    transcriber_other = np.zeros(list_transcriber.shape[0])
    for i in range(list_source.shape[0]):
        for j in range(raw_data.shape[0]):
            if (raw_data[j, 1] == "Beethoven") and (raw_data[j, 2] == list_source[i]):
                sources_beethoven[i] += 1
            elif (raw_data[j, 1] != "Beethoven") and (raw_data[j, 2] == list_source[i]):
                sources_other[i] += 1
    for i in range(list_transcriber.shape[0]):
        for j in range(raw_data.shape[0]):
            if (raw_data[j, 1] == "Beethoven") and (raw_data[j, 3] == list_transcriber[i]):
                transcriber_beethoven[i] += 1
            elif (raw_data[j, 1] != "Beethoven") and (raw_data[j, 3] == list_transcriber[i]):
                transcriber_other[i] += 1

    fig, subfigs = plt.subplots(1, 2, tight_layout=True)
    subfig = subfigs[0]
    nb_sources = np.arange(list_source.shape[0])
    subfig.bar(nb_sources, sources_beethoven, color='b', width=0.25, label="Beethoven")
    subfig.bar(nb_sources + 0.25, sources_other, color='g', width=0.25, label="other")
    subfig.legend()
    subfig.set_title("Nombre de données par source pour chaque classe")
    subfig.xaxis.set_ticks(nb_sources, list_source)
    plt.setp(subfig.xaxis.get_ticklabels(), rotation="vertical")

    subfig = subfigs[1]
    nb_transcribers = np.arange(list_transcriber.shape[0])
    subfig.bar(nb_transcribers, transcriber_beethoven, color='b', width=0.25, label="Beethoven")
    subfig.bar(nb_transcribers + 0.25, transcriber_other, color='g', width=0.25, label="other")
    subfig.legend()
    subfig.set_title("Nombre de données par transcripteur pour chaque classe")
    subfig.xaxis.set_ticks(nb_transcribers, list_transcriber)
    plt.setp(subfig.xaxis.get_ticklabels(), rotation="vertical")
    plt.show()


def train(model, train_set, validation_set, device, nb_epoch=1000, learning_rate=0.01, momentum=0.9):
    # Cette fonction est grandement inspirée de la question 1 du devoir 4
    nb_epoch = 3000
    learning_rate = 0.01
    momentum = 0.9
    batch_size = 32

    # Création du sampler avec les classes balancées
    # Create the sampler with balanced classes
    balanced_train_sampler = create_balanced_sampler(train_set)
    balanced_validation_sampler = create_balanced_sampler(validation_set)

    # Création du dataloader d'entraînement
    # Create training dataloader
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=balanced_train_sampler)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, sampler=balanced_validation_sampler)

    def train_model():
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

    scores = train_model()
    print(' [-] test acc. {:.6f}%'.format(scores[-1, 2] * 100))

    # Libère la cache sur le GPU *important sur un cluster de GPU*
    # Free GPU cache *important on a GPU cluster*
    torch.cuda.empty_cache()

    scores = np.array(scores)
    plt.plot(scores[:, 0], scores[:, 1], color="blue", label="Score en entrainement")
    plt.plot(scores[:, 0], scores[:, 2], color="red", label="Score en validation")
    plt.ylabel("Scores")
    plt.xlabel("Nombres d'époque")
    plt.legend()
    plt.title("Scores pour " + model.name)
    plt.show()


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
    make_bar_plots()

    sets = get_data()
    rng = np.random.default_rng(1)
    rng.shuffle(get_data())

    # Chargement des données d'entraînement et de test
    # Loading training and testing set
    print("Chargement des données")
    train_set_50000 = MusicDataset(sets[:300], 50000)
    validation_set_50000 = MusicDataset(sets[300:], 50000)
    train_set_5000 = MusicDataset(sets[:300], 5000)
    validation_set_5000 = MusicDataset(sets[300:], 5000)

    model_50000 = SoundNet50000()
    model_5000 = SoundNet5000()

    train(model_50000, train_set_50000, validation_set_50000, DEVICE, 3000)
    train(model_5000, train_set_5000, validation_set_5000, DEVICE, 3000)
