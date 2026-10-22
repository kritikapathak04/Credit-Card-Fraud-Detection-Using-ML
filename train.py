"""Train and evaluate all models.   Usage:  python train.py [--data data/creditcard.csv] [--fast]"""
import argparse

from src.data_loader import load_data
from src.training import run_training


def main():
    ap = argparse.ArgumentParser(description="Credit card fraud detection - training")
    ap.add_argument("--data", default=None, help="path to creditcard.csv (default: data/creditcard.csv, else synthetic)")
    ap.add_argument("--fast", action="store_true", help="smaller models for a quick run")
    ap.add_argument("--synthetic", action="store_true", help="force the synthetic demo dataset")
    args = ap.parse_args()
    df, source = load_data(args.data, force_synthetic=args.synthetic)
    run_training(df, fast=args.fast, source=source)


if __name__ == "__main__":
    main()
