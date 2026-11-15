# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import ConfusionMatrixDisplay


def edit_distance(a,b):
    # Calcul de la distance d'édition
    
    print(a, b)

    # La ligne et la colonne 0 représentent la chaîne vide (#).
    mat = np.zeros((len(a) + 1, len(b) + 1), dtype=int)

    # D[i, 0] = i : supprimer les i premiers caractères de a.
    for i in range(len(a) + 1):
        mat[i, 0] = i

    # D[0, j] = j : insérer les j premiers caractères de b.
    for j in range(len(b) + 1):
        mat[0, j] = j

    # Remplissage de la matrice.
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cout_substitution = 0 if a[i - 1] == b[j - 1] else 1

            mat[i, j] = min(
                mat[i - 1, j] + 1,                    # suppression
                mat[i, j - 1] + 1,                    # insertion
                mat[i - 1, j - 1] + cout_substitution # substitution
            )

    print(mat)
    return mat[len(a), len(b)]

def confusion_matrix(true, pred, ignore=None):
    # Calcul de la matrice de confusion

    labels = sorted(set(true) | set(pred))
    if ignore:
        labels = [label for label in labels if label not in ignore]

    confusion = cm(true, pred, labels=labels)

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion,
        display_labels=labels,
    )
    display.plot(cmap="Blues")

    plt.title("Matrice de confusion")
    plt.savefig("matrice_confusion.png")


    return confusion
