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

    # Points particuliers pour SOS, EOS et PAD
    SOS_POINT = (2.0, 0.0)
    EOS_POINT = (-2.0, 0.0)
    PAD_POINT = (0.0, 0.0)

    def __init__(self, filename):
        # Lecture du texte
        self.pad_symbol = '<pad>'
        self.start_symbol = '<sos>'
        self.stop_symbol = '<eos>'

        # self.data = dict()
        with open(filename, 'rb') as fp:
            raw = pickle.load(fp)

        #print(self.data[0])

        # 26 charactères + 3 spécial
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

        ###plus 1 a cause du end of sequence
        self.maxTexte = 5 + 1 

        # Trajectoires -> vecteurs non normalisés
        vectors = [self._trajectory_to_vectors(elem[1]) for elem in raw]

        # Normalisation
        vectors = [self._normalize(v) for v in vectors]

        # Ajout des symboles SOS / EOS
        sequences = [
            np.vstack([
                np.array([self.SOS_POINT]),
                v.reshape(-1, 2),
                np.array([self.EOS_POINT])
            ])
            for v in vectors
        ]
        
        # Ajout du padding aux séquences
        
        self.maxSEQ = max(len(s) for s in sequences)
        print(self.maxSEQ)

        self.data = []
        for elem, seq in zip(raw, sequences):
            inputs = np.tile(
                np.array(self.PAD_POINT, dtype=np.float32), (self.maxSEQ, 1)
            )
            inputs[:len(seq)] = seq

            # Cible : mot + EOS + PAD en one-hot
            target_symbols = list(elem[0]) + [self.stop_symbol]
            target_symbols += [self.pad_symbol] * (self.maxTexte - len(target_symbols))
            if len(target_symbols) != self.maxTexte:
                raise ValueError(
                    f"La cible doit contenir {self.maxTexte} symboles, "
                    f"mais en contient {len(target_symbols)}"
                )
            target_indices = np.array(
                [self.symb2int[s] for s in target_symbols], dtype=np.int64
            )
            target = np.eye(self.dict_size, dtype=np.float32)[target_indices]

            self.data.append([inputs, target])


    def _trajectory_to_vectors(self, points):
        """points : (2, N) -> vecteurs (M, 2), M <= N-1

        On retire les vecteurs de norme nulle pour éviter confusion avec PAD
        """
        pts = np.asarray(points, dtype=np.float64).T #(N, 2)
        vec = np.diff(pts, axis=0)
        norms = np.linalg.norm(vec, axis=1)
        return vec[norms > 0]

    def _normalize(self, vec):
        if len(vec) == 0:
            return vec
        norms = np.linalg.norm(vec, axis=1)
        scale = norms.max()
        return vec / scale
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs, target = self.data[idx]
        return torch.from_numpy(inputs), torch.from_numpy(target)

    def visualisation(self, idx, predicted_word=None, expected_word=None):
        """Affiche une trajectoire et, dessous, le résultat de prédiction."""

        seq = self.data[idx][0] #(T, 2)

        # On retire PAD, EOS, SOS
        norms = np.linalg.norm(seq, axis=1)
        traits = seq[(norms > 0) & (norms < 1.5)]

        pts = np.vstack([[0.0, 0.0], np.cumsum(traits, axis=0)])
        x, y = pts[:, 0], pts[:, 1]

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
        plt.show()
