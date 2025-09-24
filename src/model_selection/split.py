import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def split_dataset(metadata_path, out_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Divide el dataset en train, val y test.
    Genera CSVs separados en `out_dir`.
    """
    metadata_path = Path(metadata_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_path)

    # Primero train y temp
    train_df, temp_df = train_test_split(
        df, train_size=train_ratio, random_state=random_state
    )

    # Luego val y test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / (test_ratio + val_ratio),
        random_state=random_state
    )

    # Guardar CSVs
    train_df.to_csv(out_dir / "train_metadata.csv", index=False)
    val_df.to_csv(out_dir / "val_metadata.csv", index=False)
    test_df.to_csv(out_dir / "test_metadata.csv", index=False)

    print(f"✅ Split completado. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
