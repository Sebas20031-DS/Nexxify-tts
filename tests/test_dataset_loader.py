# python -m tests.test_dataset_loader

import torch
from torch.utils.data import DataLoader
from src.model_selection.dataset_loader import TTSDataset, collate_fn

def main():
    metadata_path = "data/processed/css10/metadata.csv"
    data_dir = "data/processed/css10"

    dataset = TTSDataset(metadata_path, data_dir)
    print(f"✅ Dataset cargado con {len(dataset)} ejemplos")

    # Crear DataLoader con collate_fn
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    batch = next(iter(loader))

    print("=== Batch de prueba ===")
    print(f"text_ids: {batch['text_ids'].shape}")
    print(f"text_lens: {batch['text_lens']}")
    print(f"mel: {batch['mel'].shape}")
    print(f"mel_lens: {batch['mel_lens']}")
    print(f"wav_paths: {batch['wav_paths'][:2]} ...")  # muestra solo 2
    print(f"texts: {batch['texts'][:2]} ...")

if __name__ == "__main__":
    main()
