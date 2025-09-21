import os
import shutil
import mlflow
from pathlib import Path
import numpy as np
from src.preprocess_text import normalize_text, tokenize, text_to_ids
from src.preprocess_audio import load_audio, audio_to_mel
from src.mlflow_setup import create_experiment_with_postgres_and_artifact


# Configurar experimento en MLflow
tracking_uri = "postgresql://postgres:2724@localhost:5432/mlflow_db" 
mlflow.set_tracking_uri(tracking_uri) 

artifact_root = Path(r"C:/Users/sebas/Documents/Projects/TTS/tts_project/mlflow/artifacts").as_uri()
exp_name = "tts_preprocessing"

# Carpeta temporal para guardar archivos antes de loggearlos
temp_dir = Path(r"C:/Users/sebas/Documents/Projects/TTS/tts_project/mlflow/temp")
temp_dir.mkdir(parents=True, exist_ok=True)

exp_id = create_experiment_with_postgres_and_artifact(exp_name, tracking_uri, artifact_root)
mlflow.set_experiment(exp_name)


# Simulamos vocabulario simple
VOCAB = {ch: i for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz,.!? ")} 

with mlflow.start_run(run_name="preprocess_sample"):

    # Parámetros
    sr = 22050
    n_fft = 1024
    hop_length = 256
    n_mels = 80

    mlflow.log_params({
        "sample_rate": sr,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "n_mels": n_mels,
        "tokenization": "characters",
        "vocab_size": len(VOCAB)
    })
    
    mlflow.set_tag("tokenization", "characters")

    # Procesar texto
    text = "¡Hola, este es un ejemplo de TTS!"
    norm_text = normalize_text(text)
    tokens = tokenize(norm_text)
    ids = text_to_ids(tokens, VOCAB)
    mlflow.log_param("text_length", len(tokens))
    
    # Guardar IDs temporalmente y como artefacto
    text_ids_path = temp_dir / "text_ids.npy"
    np.save(text_ids_path, ids)
    mlflow.log_artifact(str(text_ids_path))


    # Procesar audio
    audio_path = "data/raw/example.wav"
    audio = load_audio(audio_path, sr=sr)
    mel = audio_to_mel(audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    # Guardar espectrograma como artefacto
    mel_path = temp_dir / "mel.npy"
    np.save(mel_path, mel)
    mlflow.log_artifact(str(mel_path))

    print("Preprocesamiento completo ✅")

# Eliminar mlruns (vacio) automáticamente tras la ejecución
mlruns_path = Path("mlruns")
if mlruns_path.exists() and mlruns_path.is_dir():
    shutil.rmtree(mlruns_path)
    print("Directorio 'mlruns' eliminado automáticamente 🧹")