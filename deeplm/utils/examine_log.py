import argparse
import sys

import pandas as pd


def examine_log(log_file, ds_set="dev", epoch_idx=None):
    df = pd.read_csv(log_file)
    if not epoch_idx:
        epoch_idx = df[df["type"] == ds_set]["epoch"].max()
    df = df[(df["epoch"] == epoch_idx) & (df["type"] == ds_set)]
    recall = df["pos_acc"].mean()
    precision = df["neg_acc"].mean()
    f1 = 2 * recall * precision / (recall + precision)
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1: {f1}")


def main():
    description = "Examines a log."
    epilog = "Usage:\ncat the_log | python -m deeplm.utils.examine_log"
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument("--in_file", type=str)
    parser.add_argument("--set", type=str, choices=["dev", "test"], default="dev")
    parser.add_argument("--epoch_idx", type=int)
    args = parser.parse_args()
    args.in_file = sys.stdin if args.in_file is None else open(args.in_file)
    examine_log(args.in_file, ds_set=args.set, epoch_idx=args.epoch_idx)


if __name__ == "__main__":
    main()