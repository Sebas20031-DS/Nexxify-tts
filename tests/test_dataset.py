# tests/test_dataset.py
# Ejecutar desde root del proyecto con python -m tests.test_dataset

import numpy as np
import pandas as pd
from pathlib import Path

processed_path = Path("data/processed/css10")
metadata_file = processed_path / "metadata.csv"

metadata = pd.read_csv(metadata_file)

errors = 0

for idx, row in metadata.iterrows():
    text_ids_path = processed_path / row["text_ids_path"]
    mel_path = processed_path / row["mel_path"]

    # Verificar existencia
    if not text_ids_path.exists():
        print(f"❌ No existe {text_ids_path}")
        errors += 1
    if not mel_path.exists():
        print(f"❌ No existe {mel_path}")
        errors += 1

    # Verificar contenido
    text_ids = np.load(text_ids_path)
    mel = np.load(mel_path)

    if text_ids.size == 0:
        print(f"❌ Text IDs vacíos en {text_ids_path}")
        errors += 1
    if mel.size == 0:
        print(f"❌ Mel vacío en {mel_path}")
        errors += 1
    if mel.shape[0] != 80:  # n_mels
        print(f"❌ Mel con n_mels incorrecto en {mel_path}: {mel.shape[0]}")
        errors += 1
    if not np.isfinite(mel).all():
        print(f"❌ Mel contiene NaN o Inf en {mel_path}")
        errors += 1

if errors == 0:
    print("✅ Dataset validado correctamente. Todo listo para modelado.")
else:
    print(f"⚠️ Dataset contiene {errors} errores. Revisar preprocesamiento.")
