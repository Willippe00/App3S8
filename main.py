# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import *
from models import trajectory2seq
from dataset import *
from metrics import *

from dataset import HandwrittenWords

if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False           # Forcer a utiliser le cpu?
    trainning = True           # Entrainement?
    test = False                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)

    # À compléter
    n_epochs = 4
    batch_size = 64
    hidden_dim = 3
    n_layers = 3

    best_val = np.inf

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    # À compléter
    dataset = HandwrittenWords(filename="./data_trainval.p")

    
    # Séparation de l'ensemble de données (entraînement et validation)
    # modifier depuis code app2
    train_split = 0.70
    val_split = 0.15
    n_train_samples = int(len(dataset) * train_split)
    n_val_samples = int(len(dataset) * val_split)
    n_test_samples = len(dataset) - n_train_samples - n_val_samples
    generator1 = torch.Generator().manual_seed(42)
    dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(dataset,[n_train_samples, n_val_samples, n_test_samples],generator=generator1)
   

    # Instanciation des dataloaders
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=n_workers)#[None]
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=n_workers)#[None]
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, num_workers=n_workers)#[None]


    # Instanciation du model
    model = trajectory2seq(hidden_dim=hidden_dim, n_layers=n_layers, int2symb = dataset.int2symb, symb2int = dataset.symb2int, dict_size = dataset.dict_size, device=device, maxlen=dataset.maxTexte).to(device)


    # Initialisation des variables
    epochs_train_losses = []
    epochs_val_losses = []

    if trainning:

        # Fonction de coût et optimizateur
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        loss_criterion = nn.CrossEntropyLoss()

        if learning_curves:
            plt.ion()
            plt.figure(figsize=(8, 5))


        for epoch in range(1, n_epochs + 1):
            # Entraînement
            model.train()
            running_loss = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output, _, _ = model(data)
                # CrossEntropyLoss utilise les indices de classes, donc on
                # reconvertit la cible one-hot en indices avant le calcul.
                target_indices = target.argmax(dim=-1).long()
                loss = loss_criterion(output.transpose(1, 2), target_indices)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(
                        f"Train - époque {epoch}/{n_epochs} - "
                        f"batch {batch_idx + 1}/{len(train_loader)} - "
                        f"loss: {loss.item():.6f}",
                        flush=True,
                    )

            
            # Validation
            model.eval()
            val_loss = 0
            accuracy = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output, _, _ = model(data)
                    target_indices = target.argmax(dim=-1).long()
                    loss = loss_criterion(output.transpose(1, 2), target_indices)
                    val_loss += loss.item()
                    prediction = output.argmax(dim=-1)
                    accuracy += (prediction == target_indices).sum().item()


            # Ajouter les loss aux listes
            epochs_train_losses.append(running_loss / len(train_loader))
            epochs_val_losses.append(val_loss/ len(val_loader))

            print(
                f"Époque {epoch}/{n_epochs} terminée - "
                f"train loss: {epochs_train_losses[-1]:.6f} - "
                f"validation loss: {epochs_val_losses[-1]:.6f}",
                flush=True,
            )


            # Enregistrer les poids
            if val_loss < best_val:
                best_val = val_loss
                print('Saving new best model\n')
                torch.save(model, './poid.pt')


            # Affichage
            if learning_curves:
                # visualization
                # Affichage des courbes d'apprentissages
                ##lab app2
                plt.clf()
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.plot(range(1, epoch + 1), epochs_train_losses, color='blue', label='Training loss', linestyle=':')
                plt.plot(range(1, epoch + 1), epochs_val_losses, color='red', label='Validation loss', linestyle='-.')
                plt.legend()
                plt.draw()
                plt.pause(0.001)

        if learning_curves:
            plt.ioff()
            plt.show()

    if test:
        # Évaluation
        model = torch.load('./poid.pt', weights_only=False)

        # Charger les données de tests
        model.eval()
        test_loss = 0
        accuracy = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

        # Affichage de l'attention
        # À compléter (si nécessaire)

        # Affichage des résultats de test
        # À compléter
        
        # Affichage de la matrice de confusion
        # À compléter

        pass
