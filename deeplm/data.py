from collections import defaultdict
import enum 
import datetime
import hashlib


import boto3
import pandas as pd
import numpy as np
import torch
import torch.utils.data as td


PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
NP_TOKEN = "<no_program>"
CT_TOKEN = "<correct>"
OOV_TOKEN = "<oov>"


class DatasetType(enum.Enum):
    TRAINING = 0
    DEV = 1
    TEST = 2


def fetch_relative_date(n_days):
    td = datetime.timedelta(n_days)
    return datetime.date.today() + td


def strftime_s3(date):
    return date.strftime("%Y/%m/%d")


def range_relative_date(date1, date2):
    for x in range(date1, date2):
        yield fetch_relative_date(x)


def fetch_s3(s3_path, out_file):
    client = boto3.client("s3")
    key, path = s3_path.split("/", 1)
    try:
        client.download_file(key, path, out_file)
    except:
        pass


class WordVocab(object):

    def __init__(self, corpus):
        words = set()
        transcripts = list(corpus["transcript"])
        for transcript in transcripts:
            words.update(set(transcript))
        transcripts = list(corpus["transcript_final"])
        for transcript in transcripts:
            if transcript in (NP_TOKEN, CT_TOKEN):
                words.add(CT_TOKEN)
            else:
                words.update(set(transcript.split()))
        words.add(PAD_TOKEN)
        words.add(EOS_TOKEN)
        words.add(OOV_TOKEN)
        self.idx2tok = sorted(list(words))
        tok2idx = {c: idx for idx, c in enumerate(self.idx2tok)}
        self._oov_idx = tok2idx[OOV_TOKEN]
        self.tok2idx = defaultdict(lambda: self._oov_idx)
        self.tok2idx.update(tok2idx)

    def __len__(self):
        return len(self.idx2tok)


class CharacterVocab(object):

    def __init__(self, corpus):
        chars = set()
        transcripts = list(corpus["transcript"])
        for transcript in transcripts:
            chars.update(set(transcript))
        transcripts = list(corpus["transcript_final"])
        for transcript in transcripts:
            if transcript in (NP_TOKEN, CT_TOKEN):
                chars.add(CT_TOKEN)
            else:
                chars.update(set(transcript))
        chars.add(PAD_TOKEN)
        chars.add(EOS_TOKEN)
        self.idx2tok = sorted(list(chars))
        self.tok2idx = {c: idx for idx, c in enumerate(self.idx2tok)}

    def __len__(self):
        return len(self.idx2tok)


class Seq2SeqDataset(td.Dataset):

    def __init__(self, vocab, dataset, interstitial=True):
        super().__init__()
        self.transcripts = list(dataset["transcript"])
        self.targets = list(dataset["transcript_final"])
        self.vocab = vocab
        self.interstitial = interstitial
        self.transcripts = list(map(lambda x: [self.vocab.tok2idx[c] for c in x.split()], self.transcripts))
        self.targets = list(map(self.encode_line, self.targets))

    def encode_line(self, line):
        if line in (CT_TOKEN, NP_TOKEN):
            return [self.vocab.tok2idx[EOS_TOKEN], self.vocab.tok2idx[CT_TOKEN], self.vocab.tok2idx[EOS_TOKEN]]
        if self.interstitial:
            ints = {"X": 0, "-": 1}
            return [ints[c] for c in line.split()]
        else:
            return [self.vocab.tok2idx[c] for c in line.split()]

    @classmethod
    def iters(cls, config):
        def compute_bin(x):
            bucket = int(hashlib.sha256(str(x).encode()).hexdigest(), 16) % (training_pct + dev_pct + test_pct)
            if bucket < training_pct:
                return DatasetType.TRAINING.value
            elif bucket < training_pct + dev_pct:
                return DatasetType.DEV.value
            return DatasetType.TEST.value
        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]
        training_pct = config["training_pct"]
        dataset = pd.read_csv(config["dataset_file"], sep="\t")
        vocab = CharacterVocab(dataset)
        dataset["bin"] = np.vectorize(compute_bin)(dataset["trx"])
        datasets = [cls(vocab, dataset[dataset["bin"] == ds_type.value]) for ds_type in DatasetType]
        return vocab, datasets

    def collate(self, seq):
        seq = sorted(seq, key=lambda x: len(x[0]), reverse=True)
        transcripts, targets = zip(*seq)
        lengths = [len(t) for t in transcripts]
        max_length = max(lengths)
        transcripts = [t + [self.vocab.tok2idx[PAD_TOKEN]] * (max_length - len(t)) for t in transcripts]
        max_length = max([len(t) for t in targets])
        targets = [t + [self.vocab.tok2idx[PAD_TOKEN]] * (max_length - len(t)) for t in targets]
        return torch.LongTensor(transcripts), torch.LongTensor(targets), torch.LongTensor(lengths)

    def __getitem__(self, idx):
        transcript = self.transcripts[idx]
        target = self.targets[idx]
        return transcript, target

    def __len__(self):
        return len(self.transcripts)
