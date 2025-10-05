# tests/test_dataset.py
# Ejecutar desde root del proyecto con python -m tests.check_preprocessing
import csv
from pathlib import Path
import numpy as np
from src.preprocessing.text import normalize_text, text_to_phonemes, tokenize_phonemes, text_to_ids
from src.preprocessing.audio import load_audio, audio_to_mel

# ===== CONFIGURACIÓN =====
RAW_PATH = Path("data/raw/CSS10_spanish")
VOCAB = {ch: i for i, ch in enumerate("abcdefghijklmnopqrstuvwxyzáéíóúñü,.!? ")}
SR = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80

# Número de archivos a revisar
NUM_SAMPLES = 10

# ===== LEER TRANSCRIPT =====
transcript_file = RAW_PATH / "transcript.txt"
with open(transcript_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="|")
    lines = list(reader)[:NUM_SAMPLES]  # solo primeros NUM_SAMPLES

for line in lines:
    wav_rel, text, *_ = line
    wav_path = RAW_PATH / wav_rel

    # ===== Preprocesar texto =====
    print("\n--- Preprocesando archivo ---")
    print(f"Archivo de audio: {wav_path}")
    print(f"Texto original: {text}")

    norm_text = normalize_text(text)
    print(f"Texto normalizado: {norm_text}")

    tokens = tokenize(norm_text)
    print(f"Tokens: {tokens}")

    try:
        ids = text_to_ids(tokens, VOCAB)
        print(f"IDs: {ids}")
    except ValueError as e:
        print(f"⚠️ Error al mapear a IDs: {e}")
        continue

    # ===== Preprocesar audio =====
    audio = load_audio(str(wav_path), sr=SR)
    print(f"Audio cargado, {len(audio)} samples")

    mel = audio_to_mel(audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    print(f"Mel shape: {mel.shape}, dtype: {mel.dtype}")
