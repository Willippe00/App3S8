# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class trajectory2seq(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen):
        super(trajectory2seq, self).__init__()
        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.maxlen = maxlen

        # Definition des couches
        # Couches pour rnn
        # comme traductrion ENcodeur décodeur
        # Les entrées sont des coordonnées réelles (x, y), pas des indices.
        self.input_projection = nn.Linear(2, self.hidden_dim)
        self.en_embedding = nn.Embedding(self.dict_size, self.hidden_dim)

        self.encode = nn.GRU(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.decoder_rnn = nn.GRU(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True)

        # Couches pour attention
        # À compléter

        # Couche dense pour la sortie
        self.fc = nn.Linear(self.hidden_dim, self.dict_size)

        self.to(device)


    def encoder(self, x):
        x = self.input_projection(x)
        out, hidden = self.encode(x)
        return out, hidden

    def decode_sequence(self, encoder_outs, hidden, target=None):
        batch_size = hidden.shape[1]
        sos_index = self.symb2int['<sos>']
        vec_in = torch.full(
            (batch_size, 1), sos_index, device=self.device, dtype=torch.long
        )
        outputs = []

        for i in range(self.maxlen):
            embedding = self.en_embedding(vec_in)
            decoder_out, hidden = self.decoder_rnn(embedding, hidden)
            logits = self.fc(decoder_out)
            probabilities = torch.softmax(logits, dim=-1)
            outputs.append(probabilities)

            # Pendant l'entraînement, utiliser le caractère attendu pour
            # éviter qu'une mauvaise prédiction ne se propage à toute la suite.
            if target is not None and i < self.maxlen - 1:
                vec_in = target[:, i:i + 1]
            else:
                vec_in = probabilities.argmax(dim=-1)

        return torch.cat(outputs, dim=1), hidden, None
        

    def forward(self, x, target=None):
        # À compléter
        out, h = self.encoder(x)
        out, hidden, attn = self.decode_sequence(out, h, target)
        return out, hidden, attn
    

