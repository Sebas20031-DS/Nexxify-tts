import mlflow
import shutil
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Utils y módulos propios
from src.preprocessing.datasets import prepare_css10
from src.model_selection.dataset_loader import TTSDataset, collate_fn
from src.model_selection.split import split_dataset
from utils.mlflow import mlflow_config

def main():
    # ================================
    # Configuración de MLflow
    # ================================
    mlflow_config(exp_name="tts_pipeline")

    # Definición del vocabulario
    VOCAB = {ch: i for i, ch in enumerate("abcdefghijklmnopqrstuvwxyzáéíóúñü,.!? ")}

    # Parámetros comunes
    raw_path = "data/raw/CSS10_spanish"
    out_path = "data/processed/css10"
    split_dir = f"{out_path}/splits"

    # ================================
    # 1. Preprocesamiento
    # ================================
    with mlflow.start_run(run_name="css10_preprocess_and_split"):
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

        # Preprocesar dataset
        prepare_css10(raw_path, out_path, VOCAB, sr=22050, n_fft=1024, hop_length=256, n_mels=80)
        mlflow.log_param("output_path", out_path)

        # ================================
        # 2. Split
        # ================================
        split_dataset(out_path + "/metadata.csv", split_dir)

        # ================================
        # 3. DataLoader
        # ================================
        train_dataset = TTSDataset(f"{split_dir}/train_metadata.csv", out_path)
        val_dataset = TTSDataset(f"{split_dir}/val_metadata.csv", out_path)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

        # Prueba de batch
        batch = next(iter(train_loader))
        print("=== Batch de prueba ===")
        print(f"text_ids: {batch['text_ids'].shape}")
        print(f"text_lens: {batch['text_lens']}")
        print(f"mel: {batch['mel'].shape}")
        print(f"mel_lens: {batch['mel_lens']}")
        print(f"texts: {batch['texts'][:2]} ...")

        # Log en MLflow
        mlflow.log_param("train_size", len(train_dataset))
        mlflow.log_param("val_size", len(val_dataset))

    # ================================
    # 4. Cleanup (solo para evitar mlruns local vacío)
    # ================================
    mlruns_path = Path("mlruns")
    if mlruns_path.exists() and mlruns_path.is_dir():
        shutil.rmtree(mlruns_path)
        print("Directorio 'mlruns' eliminado automáticamente 🧹")


if __name__ == "__main__":
    main()