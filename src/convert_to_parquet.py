
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "merged_dataset": "data/processed/merged_dataset.csv",
    "train": "data/splits/train.csv",
    "validation": "data/splits/validation.csv",
    "test": "data/splits/test.csv",
}


def convert_csv_to_parquet(name, relative_path):
    csv_path = PROJECT_ROOT / relative_path
    parquet_path = csv_path.with_suffix(".parquet")

    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Optimize suitable columns without changing labels.
    for column in df.select_dtypes(include="object").columns:
        if df[column].nunique(dropna=False) / max(len(df), 1) < 0.5:
            df[column] = df[column].astype("category")

    df.to_parquet(
        parquet_path,
        engine="pyarrow",
        index=False,
        compression="snappy"
    )

    # Verify the saved file.
    restored = pd.read_parquet(parquet_path)

    assert len(df) == len(restored)
    assert list(df.columns) == list(restored.columns)

    csv_size = csv_path.stat().st_size / 1024
    parquet_size = parquet_path.stat().st_size / 1024

    print(f"\n{name}")
    print(f"Rows: {len(df)}")
    print(f"CSV: {csv_size:.2f} KB")
    print(f"Parquet: {parquet_size:.2f} KB")
    print(f"Parquet saved: {parquet_path}")


def main():
    for name, path in FILES.items():
        convert_csv_to_parquet(name, path)


if __name__ == "__main__":
    main()
