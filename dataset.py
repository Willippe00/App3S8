import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import re
import pickle
import string

class HandwrittenWords(Dataset):
    """Ensemble de donnees de mots ecrits a la main."""

    def __init__(self, filename):
        # Lecture du text
        self.pad_symbol     = pad_symbol = '<pad>'
        self.start_symbol   = start_symbol = '<sos>'
        self.stop_symbol    = stop_symbol = '<eos>'

        self.data = dict()
        with open(filename, 'rb') as fp:
            self.data = pickle.load(fp)

        #print(self.data[0])

        #26 chart + 3 spécial
        # Extraction des symboles
        self.symbols = list(string.ascii_lowercase) + [
            self.pad_symbol,
            self.start_symbol,
            self.stop_symbol,
        ]
        self.symb2int = {
            symbol: index for index, symbol in enumerate(self.symbols)
        }
        self.int2symb = {
            index: symbol for symbol, index in self.symb2int.items()
        }
        self.dict_size = len(self.symbols)  # 29


       

        ###stuct 
        ## clef : string
        ## valuer : tableau 2d float

        
        
        # Ajout du padding aux séquences
        ###plus 1 a cause du end of sequence
        maxTexte = 5 +1 
        self.maxTexte = maxTexte
        maxSEQ = max(element[1].shape[1] for element in self.data) +1 
        print(maxSEQ)
        self.maxSEQ = maxSEQ

        for elem in self.data:
            # <eos> et <pad> sont des tokens, pas plusieurs caractères.
            target_symbols = list(elem[0]) + [self.stop_symbol]
            target_symbols += [self.pad_symbol] * (maxTexte - len(target_symbols))
            if len(target_symbols) != maxTexte:
                raise ValueError(
                    f"La cible doit contenir {maxTexte} symboles, "
                    f"mais en contient {len(target_symbols)}."
                )
            # Les coordonnées doivent rester numériques. Le symbole <eos>
            # n'est pas ajouté aux entrées; on ajoute seulement un point nul
            # pour conserver la même longueur que la cible temporelle.
            ligne0 = np.append(elem[1][0], 0.0)  ## pour rester un tensor
            ligne1 = np.append(elem[1][1], 0.0)

            elem[1] = np.array([ligne0, ligne1], dtype=np.float32)

            # Encodage one-hot de chaque mot : un vecteur de longueur 29
            # pour chaque caractère, y compris <eos> et <pad>.
            # Les tokens spéciaux doivent rester entiers : <eos> et <pad>
            # ne doivent pas être découpés en caractères ('<', 'e', ...).
            target_indices = np.array(
                [self.symb2int[symbol] for symbol in target_symbols],
                dtype=np.int64,
            )
            elem[0] = np.eye(self.dict_size, dtype=np.float32)[target_indices]

           
            for j in range(maxSEQ - elem[1].shape[1]):
                ligne0 = np.append(elem[1][0], 0.0)
                ligne1 = np.append(elem[1][1], 0.0)
    
                elem[1] = np.array([ligne0, ligne1], dtype=np.float32)


        #print(self.data[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # À compléter
        elem = self.data[idx]
        data = torch.from_numpy(elem[1].T.astype(np.float32))
        target = torch.from_numpy(elem[0].astype(np.float32))
        return data, target

    def visualisation(self, idx, predicted_word=None, expected_word=None):
        """Affiche une trajectoire et, dessous, le résultat de prédiction."""

        elem = self.data[idx]

        points = elem[1]

        # points contient deux lignes : coordonnées x puis coordonnées y
        x = np.asarray(points[0], dtype=float)
        y = np.asarray(points[1], dtype=float)

        # Retirer les points de padding ajoutés en fin de séquence.
        while len(x) > 1 and x[-1] == 0 and y[-1] == 0:
            x = x[:-1]
            y = y[:-1]

        fig, (ax_trajectory, ax_result) = plt.subplots(
            2, 1, figsize=(6, 5), gridspec_kw={'height_ratios': [4, 1]}
        )
        ax_trajectory.plot(x, y, marker='.', linewidth=1.5)
        ax_trajectory.set_title(f"Trajectoire du dataset (exemple {idx})")
        ax_trajectory.set_xlabel('x')
        ax_trajectory.set_ylabel('y')
        ax_trajectory.axis('equal')
        ax_trajectory.grid(True, alpha=0.3)

        ax_result.axis('off')
        if expected_word is not None or predicted_word is not None:
            ax_result.text(
                0.5,
                0.65,
                f"Attendu : {expected_word or ''}",
                ha='center',
                va='center',
                fontsize=12,
            )
            ax_result.text(
                0.5,
                0.25,
                f"Prédit : {predicted_word or ''}",
                ha='center',
                va='center',
                fontsize=12,
                color='tab:green' if predicted_word == expected_word else 'tab:red',
            )

        fig.tight_layout()
        return fig
        

if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('data_trainval.p')
    for i in range(1):
        a.visualisation(np.random.randint(0, len(a)))
