import json
from pathlib import Path
from src.preprocessing.text import normalize_text, text_to_phonemes, tokenize_phonemes
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

# ===== CONFIG =====
RAW_PATH = Path("data/raw/CSS10_spanish")
TRANSCRIPT_FILE = RAW_PATH / "transcript.txt"
OUTPUT_FILE = Path("data/artifacts/vocab_phonemes.json")

# Inicializa el backend de espeak una vez, para ser reutilizado en todo el proceso.
backend = EspeakBackend("es", preserve_punctuation=True, with_stress=True)

def build_vocab(transcript_file, output_file, num_samples=None):
    phoneme_set = set()

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Leer las filas del archivo transcript.txt
    with open(transcript_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

        # Limitar número de muestras si se especifica
        if num_samples:
            lines = lines[:num_samples]

        for line in lines:
            # Saltar líneas vacías o sin transcripción
            row = line.strip().split("|")
            if len(row) < 2 or not row[1].strip():
                continue

            _, text, *_ = row

            # Normalizar texto
            norm_text = normalize_text(text)

            # Fonemizar el texto normalizado usando el backend de eSpeak (pasando el backend como parámetro)
            phoneme_text = text_to_phonemes(norm_text, backend)

            # Tokenizar la secuencia fonética en fonemas individuales
            tokens = tokenize_phonemes(phoneme_text)

            # Actualizar el conjunto de fonemas
            phoneme_set.update(tokens)

    # Crear el vocabulario de fonemas con índices únicos
    vocab = {ph: i for i, ph in enumerate(sorted(phoneme_set))}

    # Guardar el vocabulario en un archivo JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"✅ Vocab construido con {len(vocab)} símbolos y guardado en {output_file}")

if __name__ == "__main__":
    build_vocab(TRANSCRIPT_FILE, OUTPUT_FILE, num_samples=500)  # Puedes limitar a 500 para test
