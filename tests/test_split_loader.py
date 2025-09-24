# tests/test_split_loader.py
# Ejecutar desde root del proyecto con python -m tests.test_split_loader

import torch
from torch.utils.data import DataLoader
from src.model_selection.dataset_loader import TTSDataset, collate_fn
from src.model_selection.split import split_dataset

def main():
    metadata_path = "data/processed/css10/metadata.csv"
    split_dir = "data/processed/css10/splits"
    data_dir = "data/processed/css10"

    # Dividir dataset si aún no se hizo
    split_dataset(metadata_path, split_dir)

    # Cargar datasets
    train_dataset = TTSDataset(split_dir + "/train_metadata.csv", data_dir)
    val_dataset = TTSDataset(split_dir + "/val_metadata.csv", data_dir)
    test_dataset = TTSDataset(split_dir + "/test_metadata.csv", data_dir)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # Mini prueba de batch
    batch = next(iter(train_loader))
    print("=== Batch de prueba ===")
    print(f"text_ids: {batch['text_ids'].shape}")
    print(f"mel: {batch['mel'].shape}")
    print(f"texts: {batch['texts'][:2]} ...")

if __name__ == "__main__":
    main()
