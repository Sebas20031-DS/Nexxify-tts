import mlflow
import shutil
from pathlib import Path
from src.prepare_dataset import prepare_css10
from src.mlflow_setup import mlflow_config

# ================================
# Configuración de MLflow
# ================================
mlflow_config(
    exp_name="tts_preprocessing",
)

# ================================
# Definición de vocabulario (incluye caracteres en español)
# ================================
VOCAB = {ch: i for i, ch in enumerate("abcdefghijklmnopqrstuvwxyzáéíóúñü,.!? ")} 

# ================================
# Ejecución de preprocessing
# ================================
with mlflow.start_run(run_name="css10_preprocess"):
    mlflow.log_params({
        "sample_rate": 22050,
        "n_fft": 1024,
        "hop_length": 256,
        "n_mels": 80,
        "tokenization": "characters",
        "vocab_size": len(VOCAB),
        "mel_normalization": "[-1,1]"
    })
    
    mlflow.set_tag("dataset", "CSS10 Spanish")

    # Rutas de entrada/salida
    raw_path = "data/raw/CSS10_spanish"
    out_path = "data/processed/css10"

    # Preprocesar dataset completo
    prepare_css10(raw_path, out_path, VOCAB, sr=22050, n_fft=1024, hop_length=256, n_mels=80)

    # Guardar metadata en MLflow
    mlflow.log_param("output_path", out_path)

    print(f"✅ Preprocesamiento de CSS10 completo. Archivos guardados en: {out_path}")


# Eliminar mlruns (vacio) automáticamente tras la ejecución
mlruns_path = Path("mlruns")
if mlruns_path.exists() and mlruns_path.is_dir():
    shutil.rmtree(mlruns_path)
    print("Directorio 'mlruns' eliminado automáticamente 🧹")
    