import os
import csv
from pathlib import Path
import numpy as np
from utils.text import normalize_text, tokenize, text_to_ids
from utils.audio import load_audio, audio_to_mel

def prepare_css10(raw_path, out_path, vocab, sr=22050, n_fft=1024, hop_length=256, n_mels=80):
    raw_path = Path(raw_path)
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    transcript_file = raw_path / "transcript.txt"
    metadata_path = out_path / "metadata.csv"

    with open(transcript_file, "r", encoding="utf-8") as f_in, \
         open(metadata_path, "w", newline="", encoding="utf-8") as f_out:

        reader = csv.reader(f_in, delimiter="|")
        writer = csv.writer(f_out)
        writer.writerow(["wav_path", "text", "text_ids_path", "mel_path", "duration"])

        for line in reader:
            wav_rel, text, _, duration = line
            wav_path = raw_path / wav_rel

            # Normalizar y tokenizar
            norm_text = normalize_text(text)
            tokens = tokenize(norm_text)
            try:
                ids = text_to_ids(tokens, vocab)
            except ValueError as e:
                print(f"⚠️ Error en {wav_path}: {e}")
                continue  # saltar archivos problemáticos

            # Guardar text_ids
            text_ids_path = out_path / f"{Path(wav_rel).stem}_ids.npy"
            np.save(text_ids_path, np.array(ids, dtype=np.int64))

            # Procesar audio -> mel
            audio = load_audio(str(wav_path), sr=sr)
            mel = audio_to_mel(audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            if mel.shape[1] < 5:  # audio demasiado corto
                print(f"⚠️ Mel muy corto para {wav_path}, saltando")
                continue

            mel_path = out_path / f"{Path(wav_rel).stem}_mel.npy"
            np.save(mel_path, mel)

            # Registrar en metadata
            writer.writerow([wav_rel, norm_text, text_ids_path.name, mel_path.name, duration])

    print(f"✅ Dataset procesado. Archivos guardados en {out_path}")
    print(f"📄 Metadata: {metadata_path}")
