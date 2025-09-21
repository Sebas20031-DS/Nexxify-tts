# src/prepare_dataset_css10.py
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.preprocess_text import normalize_text, tokenize, text_to_ids
from src.preprocess_audio import load_audio, audio_to_mel

def prepare_css10(dataset_path: str, out_path: str, vocab: dict, sr=22050):
    dataset_path = Path(dataset_path)
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    transcript = pd.read_csv(dataset_path / "transcript.txt", sep="|", header=None)
    transcript.columns = ["id", "text"]

    for _, row in tqdm(transcript.iterrows(), total=len(transcript)):
        wav_path = dataset_path / "wav" / f"{row['id']}.wav"

        # --- Texto ---
        text = normalize_text(row["text"])
        tokens = tokenize(text)
        ids = text_to_ids(tokens, vocab)
        np.save(out_path / f"{row['id']}_ids.npy", ids)

        # --- Audio ---
        audio = load_audio(str(wav_path), sr)
        mel = audio_to_mel(audio, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
        np.save(out_path / f"{row['id']}_mel.npy", mel)

    print(f"✅ Dataset CSS10 procesado y guardado en {out_path}")
