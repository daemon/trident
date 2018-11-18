from copy import deepcopy
from functools import lru_cache
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from deeplm.weight_drop import WeightDrop


@lru_cache(maxsize=100)
def cache_eye(length, device):
    return torch.eye(length).to(device)


def to_ohv(indices, length):
    return cache_eye(length, str(indices.device))[indices]


class EncoderRNN(nn.Module):

    def __init__(self, vocab, config):
        super().__init__()
        self.hid_size = config["encoder"]["hidden_size"]
        self.n_layers = config["encoder"]["n_layers"]
        self.emb_size = config["encoder"]["emb_size"]
        self.vocab = vocab
        self.embedding = nn.Embedding(len(vocab), self.emb_size)
        self.bi = config["encoder"]["bidirectional"]
        if config["encoder"]["type"].upper() == "GRU":
            self.encoder = nn.GRU(self.emb_size, self.hid_size, self.n_layers, batch_first=True, bidirectional=self.bi)
        elif config["encoder"]["type"].upper() == "LSTM":
            self.encoder = nn.LSTM(self.emb_size, self.hid_size, self.n_layers, batch_first=True, bidirectional=self.bi)
        wdrop = config["encoder"]["weight_drop"]
        if wdrop:
            self.encoder = WeightDrop(self.encoder, ["weight_hh_l0"], dropout=wdrop)

    def forward(self, x, lengths):
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        rnn_seq, rnn_out  = self.encoder(x)
        rnn_seq, _ = nn.utils.rnn.pad_packed_sequence(rnn_seq)
        return rnn_seq, rnn_out


class GeneralAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = nn.Linear(config["decoder"]["hidden_size"], config["encoder"]["hidden_size"])
        self.fc = nn.Linear(config["encoder"]["hidden_size"] + config["decoder"]["hidden_size"], config["dnn_size"])

    def forward(self, x, rnn_seq):
        rnn_out = x
        x = self.attn(x).unsqueeze(-1).expand(-1, -1, rnn_seq.size(2))
        x = x.unsqueeze(-2).matmul(rnn_seq.unsqueeze(-1)).squeeze(-1)
        scores = F.softmax(x, 1).expand_as(rnn_seq)
        c = scores * rnn_seq
        c = c.sum(-1)
        return torch.tanh(self.fc(torch.cat((c, rnn_out), 1)))


class DecoderRNN(nn.Module):

    def __init__(self, vocab, config):
        super().__init__()
        dec_hid_size = config["decoder"]["hidden_size"]
        self.hid_size = config["decoder"]["hidden_size"]
        self.n_layers = config["decoder"]["n_layers"]
        if config["decoder"]["type"].upper() == "GRU":
            self.decoder = nn.GRUCell(dec_hid_size, self.hid_size)
        elif config["decoder"]["type"].upper() == "LSTM":
            self.decoder = nn.LSTMCell(dec_hid_size, self.hid_size)

    def forward(self, x, hidden):
        rnn_out  = self.decoder(x, hidden)
        if isinstance(self.decoder, nn.LSTMCell):
            rnn_out = rnn_out[0]
        return rnn_out


class InterstitialModel(nn.Module):

    def __init__(self, vocab, config):
        super().__init__()
        self.vocab = vocab
        self.config = config
        self.encoder = EncoderRNN(vocab, config)
        self.cnn_size = config["cnn_size"]
        self.dnn_size = config["dnn_size"]
        self.cnn = nn.Conv2d(1, self.cnn_size, (2 * self.encoder.hid_size, 1))
        self.dnn = nn.Linear(self.cnn_size, self.dnn_size)
        self.linear = nn.Linear(self.cnn_size, 3)

        self.steps_ema = 0
        self.beta_ema = 0.99
        self.avg_param = deepcopy(list(p.data for p in self.parameters()))
        if torch.cuda.is_available():
            self.avg_param = [a.cuda() for a in self.avg_param]

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema**self.steps_ema))

    def forward(self, x, lengths, gt=None):
        rnn_seq, rnn_out = self.encoder(x, lengths)
        rnn_seq = rnn_seq.permute(1, 2, 0).contiguous()
        x = F.relu(self.cnn(rnn_seq.unsqueeze(1)), inplace=True)
        x = x.squeeze(-2).permute(0, 2, 1).contiguous()
        bsz, l, _ = x.size()
        # x = F.relu(self.dnn(x).view(bsz * l, -1), inplace=True)
        x = self.linear(x).view(bsz, l, -1).permute(0, 2, 1)
        return x


class Seq2SeqModel(nn.Module):

    def __init__(self, vocab, config):
        super().__init__()
        self.vocab = vocab
        self.config = config
        self.encoder = EncoderRNN(vocab, config)
        self.attn = GeneralAttention(config)
        self.decoder = DecoderRNN(vocab, config)
        self.teacher_forcing = config["teacher_forcing"]
        self.rand = random.Random(config["seed"])
        self.dnn_size = config["dnn_size"]
        self.linear = nn.Linear(len(vocab) + self.dnn_size, self.decoder.hid_size)
        self.fc = nn.Linear(config["dnn_size"], len(vocab))

    def forward(self, x, lengths, gt=None):
        rnn_seq, rnn_out = self.encoder(x, lengths)
        rnn_seq = rnn_seq.permute(1, 2, 0).contiguous()
        rnn_out = rnn_out[-1]
        outputs = []
        last_attn = torch.zeros(x.size(0), self.dnn_size).to(x.device)
        gt = to_ohv(gt, len(self.vocab)).permute(1, 0, 2)
        for idx in range(len(gt) - 1):
            if self.training and ((self.teacher_forcing and self.rand.random() < self.teacher_forcing) or idx == 0):
                y_in = gt[idx]
            y_in = self.linear(torch.cat((y_in, last_attn), 1))
            dec_out = self.decoder(y_in, rnn_out)
            last_attn = self.attn(dec_out, rnn_seq)
            x = self.fc(last_attn)
            outputs.append(x)
            y_in = to_ohv(x.max(1)[1], len(self.vocab))
        return torch.stack(outputs, -1)

