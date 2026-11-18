# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class trajectory2seq(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen, rnn_type='GRU', bidirectional=False):
        super(trajectory2seq, self).__init__()
        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.maxlen = maxlen
        self.rnn_type = rnn_type.upper()
        self.bidirectional = bidirectional


        self.en_embedding = nn.Embedding(self.dict_size, self.hidden_dim)

        rnn_class = {
            'ELMAN': nn.RNN,
            'GRU': nn.GRU,
            'LSTM': nn.LSTM,
        }[self.rnn_type]
        self.encode = rnn_class(
            2, self.hidden_dim, self.n_layers,
            batch_first=True, bidirectional=self.bidirectional,
        )
        
        # Le décodeur reste unidirectionnel pour générer les lettres dans
        self.decoder_rnn = rnn_class(
            self.hidden_dim, self.hidden_dim, self.n_layers,
            batch_first=True,
        )

        # Couches pour attention
        self.hidden2query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.att_combine = nn.Linear(2 * self.hidden_dim, self.hidden_dim)

        # Couche dense pour la sortie
        self.fc = nn.Linear(self.hidden_dim, self.dict_size)

        self.to(device)


    def encoder(self, x):

        lengths = (x.abs().sum(dim=2) > 0).sum(dim=1).clamp_min(1)
        encoder_outs, hidden = self.encode(x)

        if self.bidirectional:
            encoder_outs = encoder_outs.view(
                x.size(0), x.size(1), 2, self.hidden_dim
            ).sum(dim=2)

            if self.rnn_type == 'LSTM':
                h_n, c_n = hidden
                h_n = h_n.view(
                    self.n_layers, 2, h_n.size(1), self.hidden_dim
                ).sum(dim=1)
                c_n = c_n.view(
                    self.n_layers, 2, c_n.size(1), self.hidden_dim
                ).sum(dim=1)
                hidden = (h_n, c_n)
            else:
                hidden = hidden.view(
                    self.n_layers, 2, hidden.size(1), self.hidden_dim
                ).sum(dim=1)

        encoder_mask = (
            torch.arange(x.size(1), device=x.device)[None, :]
            < lengths.to(x.device)[:, None]
        )

        return encoder_outs, hidden, encoder_mask

    def decode_sequence(self, encoder_outs, hidden, target=None):

        batch_size = hidden[0].shape[1] if self.rnn_type == 'LSTM' else hidden.shape[1]
        sos_index = self.symb2int['<sos>']
        vec_in = torch.full(
            (batch_size, 1), sos_index, device=self.device, dtype=torch.long
        )
        outputs = []

        for i in range(self.maxlen):
            embedding = self.en_embedding(vec_in)
            decoder_out, hidden = self.decoder_rnn(embedding, hidden)
            logits = self.fc(decoder_out)
            outputs.append(logits)

        return torch.cat(outputs, dim=1), hidden, None

    def attentionModule(self, query, values, mask):
        q = self.hidden2query(query)
        scores = torch.bmm(q, values.transpose(1, 2))
        scores = scores.masked_fill(~mask[:, None, :], float('-inf'))
        weights = torch.softmax(scores, dim=2)
        context = torch.bmm(weights, values)
        out = torch.tanh(
            self.att_combine(torch.cat((query, context), dim=2))
        )
        return out, weights

    def decode_sequence_Attn(self, encoder_outs, hidden, encoder_mask, target=None):
        
        batch_size = hidden[0].shape[1] if self.rnn_type == 'LSTM' else hidden.shape[1]
        sos_index = self.symb2int['<sos>']
        vec_in = torch.full(
            (batch_size, 1),
            sos_index,
            device=self.device,
            dtype=torch.long,
        )
        outputs = []
        attentions = []

        for i in range(self.maxlen):
            embedding = self.en_embedding(vec_in)
            decoder_out, hidden = self.decoder_rnn(embedding, hidden)

            attention_output, weights = self.attentionModule(
                decoder_out,
                encoder_outs,
                encoder_mask,
            )

            logits = self.fc(attention_output)

            outputs.append(logits)
            attentions.append(weights)

        return (
            torch.cat(outputs, dim=1),
            hidden,
            torch.cat(attentions, dim=1),
        )
        

    def forward(self, x, target=None):

        out, h, encoder_mask = self.encoder(x)
        #out, hidden, attn = self.decode_sequence(out, h, target)
        out, hidden, attn = self.decode_sequence_Attn(
            out,
            h,
            encoder_mask,
            target,
        )
        return out, hidden, attn
    

