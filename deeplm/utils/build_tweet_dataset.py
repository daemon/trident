from collections import defaultdict
import argparse
import enum
import json
import re
import os
import sys
import unicodedata

from tqdm import tqdm
import pandas as pd
import torch


def deaccent(str):
   return "".join(c for c in unicodedata.normalize("NFD", str) if unicodedata.category(c) != "Mn")


class TokenEnum(enum.Enum):

    HASHTAG = enum.auto()
    USER = enum.auto()
    URL = enum.auto()
    RETWEET = enum.auto()
    TEXT = enum.auto()


def categorize(token):
    if token[0] == "#":
        return TokenEnum.HASHTAG
    elif token[0] == "@":
        return TokenEnum.USER
    elif token[:4].lower() == "http":
        return TokenEnum.URL
    elif token == "RT":
        return TokenEnum.RETWEET
    else:
        return TokenEnum.TEXT


def clean_tokens(tokens, pattern=re.compile(r"[^A-z0-9]"), type="text"):
    if len(tokens) == 0:
        return
    if categorize(tokens[0]) == TokenEnum.RETWEET:
        tokens = tokens[1:]
    tokens = filter(lambda x: categorize(x) == getattr(TokenEnum, type.upper()), tokens)
    clean_toks = []
    for tok in tokens:
        tok = re.sub(pattern, "", deaccent(tok))
        if len(tok) == 0:
            continue
        clean_toks.append(tok)
    return clean_toks


def n_gramify(tokens, n_grams, skip=None):
    if skip is None:
        skip = 1
    if isinstance(skip, int):
        skip = [skip] * n_grams
    for n_gram, s in zip(range(1, min(n_grams + 1, len(tokens))), skip):
        for idx in range(0, len(tokens) - n_gram + 1, s):
            yield tokens[idx:idx + n_gram]


def build_tweet_dataset(file_in, file_out, n_grams=5, type="text", words=None):
    df = defaultdict(list)
    pbar = tqdm(file_in)
    pbar.set_description("Crunching tweets")
    count = 0
    check_set = set()
    ws_patt = re.compile(r"\s+")
    ws_patt_inv = re.compile(r"[^\s]")
    for line in pbar:
        data = json.loads(line)
        if "text" not in data:
            continue
        twid = data["id"]
        text = data["text"]
        toks = clean_tokens(re.split(ws_patt, text), type=type)
        if not toks:
            continue
        if type == "text":
            for n_gram in n_gramify(toks, n_grams, skip=[1, 2, 2, 3, 3]):
                orig = " ".join(list("".join(n_gram))).lower()
                segmented = re.sub(ws_patt_inv, "X", " ".join(n_gram).lower())
                segmented = " ".join(list(segmented.replace(" ", "-").replace("X-", "-")))

                chk_key = " ".join(n_gram)
                if chk_key in check_set:
                    continue
                check_set.add(chk_key)
                
                df["idx"].append(0)
                df["transcript"].append(orig)
                df["transcript_final"].append(segmented)
                df["trx"].append(twid)
                count += 1
        elif type == "hashtag":
            for hashtag in toks:
                hashtag = hashtag.lower()
                if hashtag in check_set or hashtag in words:
                    continue
                check_set.add(hashtag)
                print(hashtag, file=file_out)
                count += 1
        pbar.set_postfix(dict(generated_lines=f"{count}"))
    if type == "text":
        pd.DataFrame(df).to_csv(file_out, index=False, sep="\t")


def main():
    description = "Builds a tweet dataset for domain parsing."
    epilog = "Usage:\ncat tuna_file| python -m deeplm.utils.build_tweet_dataset > tweets.csv"
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument("--out_file", type=str)
    parser.add_argument("--castor_data", type=str, default="../Castor-data")
    parser.add_argument("--in_file", type=str)
    parser.add_argument("--n_grams", type=int, default=5)
    parser.add_argument("--type", type=str, choices=["text", "hashtag"], default="text")
    args = parser.parse_args()
    out_file = open(args.out_file, "w") if args.out_file else sys.stdout
    in_file = open(args.in_file) if args.in_file else sys.stdin
    wordvecs = None
    if args.type == "hashtag":
        wordvecs = torch.load(os.path.join(args.castor_data, "embeddings", "word2vec", "GoogleNews-vectors-negative300.txt.pt"))[0]
        wordvecs = set(list(map(str.lower, wordvecs)))
    build_tweet_dataset(in_file, out_file, n_grams=args.n_grams, type=args.type, words=wordvecs)


if __name__ == "__main__":
    main()