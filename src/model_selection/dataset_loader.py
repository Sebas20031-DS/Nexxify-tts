import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path

class TTSDataset(Dataset):
    """
    Dataset para TTS basado en metadata.csv.
    Devuelve pares (text_ids, mel) junto con metadatos útiles.
    """
    def __init__(self, metadata_path, data_dir):
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_path)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Cargar texto (ids)
        ids = np.load(self.data_dir / row["text_ids_path"])
        ids = torch.tensor(ids, dtype=torch.long)

        # Cargar mel
        mel = np.load(self.data_dir / row["mel_path"])
        mel = torch.tensor(mel, dtype=torch.float)

        return {
            "text_ids": ids,
            "mel": mel,
            "wav_path": str(self.data_dir / row["wav_path"]),
            "duration": float(row["duration"]),
            "text": row["text"],
        }


def collate_fn(batch):
    """
    Collate function para Tacotron2.
    Aplica padding dinámico a text_ids y mels.
    """
    # Ordenar batch por longitud de texto (recomendado en Tacotron2)
    batch.sort(key=lambda x: len(x["text_ids"]), reverse=True)

    # Textos
    text_lens = [len(x["text_ids"]) for x in batch]
    max_text_len = max(text_lens)
    padded_texts = torch.zeros(len(batch), max_text_len, dtype=torch.long)

    for i, x in enumerate(batch):
        padded_texts[i, :len(x["text_ids"])] = x["text_ids"]

    # Mels
    mel_lens = [x["mel"].shape[1] for x in batch]  # eje 1 = tiempo
    max_mel_len = max(mel_lens)
    n_mels = batch[0]["mel"].shape[0]

    padded_mels = torch.zeros(len(batch), n_mels, max_mel_len, dtype=torch.float)

    for i, x in enumerate(batch):
        mel = x["mel"]
        padded_mels[i, :, :mel.shape[1]] = mel

    # Empaquetar
    return {
        "text_ids": padded_texts,
        "text_lens": torch.tensor(text_lens, dtype=torch.long),
        "mel": padded_mels,
        "mel_lens": torch.tensor(mel_lens, dtype=torch.long),
        "wav_paths": [x["wav_path"] for x in batch],
        "texts": [x["text"] for x in batch],
    }
