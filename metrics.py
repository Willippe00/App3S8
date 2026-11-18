# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import ConfusionMatrixDisplay
from dataset import *


dataset = HandwrittenWords(filename="./data_trainval.p")
def edit_distance(a,b):
    
    #print(a, b)

    # création de la matrice
    mat = np.zeros((len(a) + 1, len(b) + 1), dtype=int)

    ##init première ligne horizntale et vertical
    for i in range(len(a) + 1):
        mat[i, 0] = i
    for j in range(len(b) + 1):
        mat[0, j] = j

    # Remplissage de la matrice.
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cout_substitution = 0 if a[i - 1] == b[j - 1] else 1
            mat[i, j] = min(mat[i - 1, j] + 1, mat[i, j - 1] + 1, mat[i - 1, j - 1] + cout_substitution)

    #print(mat)
    return mat[len(a), len(b)] #retourne l'élément en bas a droit

def confusion_matrix(true, pred, ignore=None):
    # Calcul de la matrice de confusion

    labels = sorted(set(true) | set(pred))
    if ignore:
        labels = [label for label in labels if label not in ignore]

    confusion = cm(true, pred, labels=labels) #matrice de confusion sktlearn
    display = ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=labels,)
    display.plot(cmap="Blues")

    plt.title("Matrice de confusion")
    plt.savefig("matrice_confusion.png")

    return confusion



def plot_attention_trajectory(seq, attention, predicted_indices, filename):
        """Dessine la trajectoire en gris et l'attention de chaque symbole en noir."""
        seq = seq.detach().cpu().numpy()
        attention = attention.detach().cpu().numpy()
        predicted_indices = predicted_indices.detach().cpu().numpy()

        positions = np.zeros((len(seq), 2), dtype=np.float32)
        current = np.zeros(2, dtype=np.float32)
        for timestep, vector in enumerate(seq):
            norm = np.linalg.norm(vector)
            if 0 < norm < 1.5:  # ignore SOS, EOS et PAD
                current = current + vector
            positions[timestep] = current

        norms = np.linalg.norm(seq, axis=1)
        valid_steps = (norms > 0) & (norms < 1.5)
        stroke_positions = positions[valid_steps]
        if len(stroke_positions) == 0:
            return

        labels = [dataset.int2symb[int(index)] for index in predicted_indices]
        x_min, x_max = stroke_positions[:, 0].min(), stroke_positions[:, 0].max()
        y_min, y_max = stroke_positions[:, 1].min(), stroke_positions[:, 1].max()
        margin = 0.05 * max(x_max - x_min, y_max - y_min, 1.0)

        fig, axes = plt.subplots(
            len(labels), 1, figsize=(4.5, 1.15 * len(labels)), squeeze=False
        )
        for char_index, (ax, label) in enumerate(zip(axes[:, 0], labels)):
            ax.plot(stroke_positions[:, 0], stroke_positions[:, 1],
                    color='lightgray', linewidth=2.0)
            weights = np.maximum(attention[char_index, :len(seq)], 0)
            if weights.max() > 0:
                weights = weights / weights.max()
            focused = valid_steps & (weights > 0.02)
            ax.scatter(positions[focused, 0], positions[focused, 1],
                        c=weights[focused], cmap='gray_r', vmin=0, vmax=1,
                        s=8 + 28 * weights[focused], zorder=3)
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(label, rotation=0, labelpad=12, va='center')
            for spine in ax.spines.values():
                spine.set_visible(True)

        fig.tight_layout(h_pad=0.25)
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)