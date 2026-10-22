"""Create a synthetic demo dataset (same schema as the Kaggle file) at data/sample_creditcard.csv."""
from src.config import SYNTH_DATA
from src.data_loader import generate_synthetic

if __name__ == "__main__":
    SYNTH_DATA.parent.mkdir(exist_ok=True)
    df = generate_synthetic()
    df.to_csv(SYNTH_DATA, index=False)
    print(f"Wrote {len(df):,} rows ({int(df.Class.sum())} fraud) -> {SYNTH_DATA}")
