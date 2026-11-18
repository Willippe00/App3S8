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
    force_cpu = False            # Le backend Triton/CUDA local n'a pas de compilateur C.
    trainning = True            # Entrainement? Requis après une modification du modèle.
    test = True                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)

    # À compléter
    typeRNN = "GRU" #"ELMAN", "GRU", "LSTM"
    bidirectionnel = True 
    n_epochs = 30
    batch_size = 128 #128 
    hidden_dim = 11
    n_layers = 3
    lr = 0.03  #0.005

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
    dataset_test =  HandwrittenWords(filename="./data_test.p")

    
    # Séparation de l'ensemble train/validation.
    # Le jeu de test provient séparément de data_test.p et ne doit pas
    # être remplacé par une troisième portion de data_trainval.p.
    train_split = 0.85
    n_train_samples = int(len(dataset) * train_split)
    n_val_samples = len(dataset) - n_train_samples
    generator1 = torch.Generator().manual_seed(42)
    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset,
        [n_train_samples, n_val_samples],
        generator=generator1,
    )
   

    # Instanciation des dataloaders
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=n_workers)#[None]
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=n_workers)#[None]
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, num_workers=n_workers)#[None]


    # Instanciation du model
    model = trajectory2seq(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        int2symb=dataset.int2symb,
        symb2int=dataset.symb2int,
        dict_size=dataset.dict_size,
        device=device,
        maxlen=dataset.maxTexte,
        rnn_type=typeRNN,
        bidirectional=bidirectionnel,
    ).to(device)

    nb_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre total de paramètres : {nb_params:,}")
    # Initialisation des variables
    epochs_train_losses = []
    epochs_val_losses = []

    distance_train = []
    distance_val = []

    def indices_to_word(indices):
        """Convertit une séquence d'indices en chaîne sans les tokens spéciaux."""
        word = []
        for index in indices:
            symbol = dataset.int2symb[int(index)]
            if symbol in ('<eos>', '<pad>'):
                break
            if symbol != '<sos>':
                word.append(symbol)
        return ''.join(word)

    if trainning:

        # Fonction de coût et optimizateur
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_criterion = nn.CrossEntropyLoss(
            ignore_index=dataset.symb2int['<pad>']
        )

        if learning_curves:
            plt.ion()
            figure, axes = plt.subplots(1, 2, figsize=(12, 5))
            loss_axis, distance_axis = axes


        for epoch in range(1, n_epochs + 1):
            # Entraînement
            model.train()
            running_loss = 0
            train_edit_distances = []

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                target_indices = target.argmax(dim=-1).long()
                output, _, _ = model(data, target_indices)
                loss = loss_criterion(output.transpose(1, 2), target_indices)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                prediction = output.argmax(dim=-1)
                for predicted_indices, expected_indices in zip(
                    prediction.detach().cpu(), target_indices.detach().cpu()
                ):
                    train_edit_distances.append(
                        edit_distance(
                            indices_to_word(predicted_indices),
                            indices_to_word(expected_indices),
                        )
                    )

            
            # Validation
            model.eval()
            val_loss = 0
            accuracy = 0
            val_total_characters = 0
            val_edit_distances = []
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output, _, _ = model(data)
                    target_indices = target.argmax(dim=-1).long()
                    loss = loss_criterion(output.transpose(1, 2), target_indices)
                    val_loss += loss.item()
                    prediction = output.argmax(dim=-1)
                    valid = target_indices != dataset.symb2int['<pad>']


                    accuracy += ((prediction == target_indices) & valid).sum().item()
                    val_total_characters += valid.sum().item()

                    for predicted_indices, expected_indices in zip(
                        prediction.cpu(), target_indices.cpu()
                    ):
                        val_edit_distances.append(
                            edit_distance(
                                indices_to_word(predicted_indices),
                                indices_to_word(expected_indices),
                            )
                        )


            # Ajouter les loss aux listes
            epochs_train_losses.append(running_loss / len(train_loader))
            epochs_val_losses.append(val_loss/ len(val_loader))
            distance_train.append(np.mean(train_edit_distances))
            distance_val.append(np.mean(val_edit_distances))

            print(
                f"Époque {epoch}/{n_epochs} terminée - "
                f"train loss: {epochs_train_losses[-1]:.6f} - "
                f"validation loss: {epochs_val_losses[-1]:.6f} - "
                f"distance train: {distance_train[-1]:.4f} - "
                f"distance validation: {distance_val[-1]:.4f}",
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
                epochs = range(1, epoch + 1)
                loss_axis.clear()
                loss_axis.set_xlabel('Epochs')
                loss_axis.set_ylabel('Loss')
                loss_axis.plot(epochs, epochs_train_losses, color='blue', label='Training loss', linestyle=':')
                loss_axis.plot(epochs, epochs_val_losses, color='red', label='Validation loss', linestyle='-.')
                loss_axis.legend()

                distance_axis.clear()
                distance_axis.set_xlabel('Epochs')
                distance_axis.set_ylabel("Distance d'édition moyenne")
                distance_axis.plot(epochs, distance_train, color='blue', label='Training distance', linestyle=':')
                distance_axis.plot(epochs, distance_val, color='red', label='Validation distance', linestyle='-.')
                distance_axis.legend()

                figure.tight_layout()
                figure.canvas.draw()
                figure.canvas.flush_events()
                plt.pause(0.001)
                figure.savefig('courbe.png')

        if learning_curves:
            plt.ioff()
            plt.show()

    if test:
        # Évaluation
        model = torch.load('./poid.pt', map_location=device, weights_only=False)

        # Charger les données de tests
        model.eval()
        test_loss = 0
        accuracy = 0
        total_characters = 0
        edit_distances = []
        loss_criterion = nn.CrossEntropyLoss(
            ignore_index=dataset.symb2int['<pad>']
        )
        example_position = int(np.random.default_rng().integers(len(dataset_test)))
        example_prediction = None
        example_expected = None
        example_attention = None
        example_data = None
        example_predicted_indices = None
        sample_offset = 0

        true_labels = []
        predicted_labels = []

        def indices_to_display(indices):
            """Convertit une séquence en texte visible, tokens spéciaux compris."""
            word = indices_to_word(indices)
            if word:
                return word

            first_symbol = dataset.int2symb[int(indices[0])]
            return first_symbol if first_symbol in ('<eos>', '<pad>') else '<vide>'

        

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output, _, attention = model(data)

                target_indices = target.argmax(dim=-1).long()
                prediction = output.argmax(dim=-1)
                test_loss += loss_criterion(
                    output.transpose(1, 2), target_indices
                ).item()
                valid = target_indices != dataset.symb2int['<pad>']

                true_labels.extend(target_indices[valid].cpu().numpy())
                predicted_labels.extend(prediction[valid].cpu().numpy())

                accuracy += ((prediction == target_indices) & valid).sum().item()
                total_characters += valid.sum().item()

                for batch_position, (predicted_indices, expected_indices) in enumerate(zip(
                    prediction.cpu(), target_indices.cpu()
                )):
                    predicted_word = indices_to_word(predicted_indices)
                    expected_word = indices_to_word(expected_indices)
                    if sample_offset + batch_position == example_position:
                        example_prediction = indices_to_display(predicted_indices)
                        example_expected = indices_to_display(expected_indices)
                        example_attention = attention[batch_position].cpu()
                        example_data = data[batch_position].cpu()
                        example_predicted_indices = predicted_indices.cpu()
                    edit_distances.append(
                        edit_distance(predicted_word, expected_word)
                    )
                sample_offset += prediction.size(0)


        mean_edit_distance = np.mean(edit_distances)
        mean_test_loss = test_loss / len(test_loader)
        test_accuracy = accuracy / total_characters

        print(f"Test loss: {mean_test_loss:.6f}")
        print(f"Test accuracy: {test_accuracy:.2%}")
        print(f"Distance d'édition moyenne: {mean_edit_distance:.4f}")

        # Affichage d'un exemple choisi au hasard dans le dataset de test.
        if example_prediction is not None:
            fig = dataset_test.visualisation(
                example_position,
                predicted_word=example_prediction,
                expected_word=example_expected,
            )
            fig.savefig('resultat_test.png', dpi=150, bbox_inches='tight')
            plt.close(fig)

        # Affichage de l'attention
        if example_attention is not None:
            plot_attention_trajectory(
                example_data,
                example_attention,
                example_predicted_indices,
                'example_test_avec_attention.png',
            )

        # Affichage de la matrice de confusion
        true_labels = [dataset.int2symb[int(label)] for label in true_labels]
        predicted_labels = [dataset.int2symb[int(label)] for label in predicted_labels]
        confusion_matrix(true_labels, predicted_labels, [dataset.stop_symbol])
