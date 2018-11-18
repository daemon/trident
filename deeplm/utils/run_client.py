import argparse

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    args = parser.parse_args()
    text = requests.post("http://127.0.0.1:8080/", json=dict(text=args.text)).json()["text"]
    print(f"{args.text:<20} {text:<20}")


if __name__ == "__main__":
    main()
