# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import ConfusionMatrixDisplay


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
