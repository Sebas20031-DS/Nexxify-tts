import mlflow
import shutil
import uuid
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Módulos propios
from src.model_selection.dataset_loader import TTSDataset, collate_fn
from src.model_selection.split import split_dataset
from src.training.tacotron2 import Tacotron2
from utils.mlflow import mlflow_config


def train_one_epoch(model, loader, optimizer, criterion_mel, criterion_stop, device, teacher_forcing_ratio=0.9):
    model.train()
    total_loss, total_mel_loss, total_stop_loss = 0, 0, 0
    for batch in loader:
        text_ids = batch["text_ids"].to(device)
        text_lens = batch["text_lens"].to(device)
        mels = batch["mel"].to(device)

        optimizer.zero_grad()
        mel_out, stop_out, _ = model(text_ids, text_lens, mels=mels,
                                     teacher_forcing_ratio=teacher_forcing_ratio)

        # Ajustar longitudes temporales
        T_target = mels.size(2)
        T_out = mel_out.size(2)
        if T_out > T_target:
            mel_out = mel_out[:, :, :T_target]
            stop_out = stop_out[:, :T_target]
        elif T_out < T_target:
            mels = mels[:, :, :T_out]
            stop_out = stop_out[:, :T_out]

        # losses
        mel_loss = criterion_mel(mel_out, mels)
        stop_targets = torch.zeros_like(stop_out)  # supervisión de stop opcional
        stop_loss = criterion_stop(stop_out, stop_targets)

        loss = mel_loss + stop_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mel_loss += mel_loss.item()
        total_stop_loss += stop_loss.item()

    n_batches = len(loader)
    return total_loss / n_batches, total_mel_loss / n_batches, total_stop_loss / n_batches


def validate(model, loader, criterion_mel, criterion_stop, device):
    model.eval()
    total_loss, total_mel_loss, total_stop_loss = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            text_ids = batch["text_ids"].to(device)
            text_lens = batch["text_lens"].to(device)
            mels = batch["mel"].to(device)

            mel_out, stop_out, _ = model(text_ids, text_lens, mels=mels,
                                         teacher_forcing_ratio=0.0)

            # Ajustar longitudes temporales
            T_target = mels.size(2)
            T_out = mel_out.size(2)
            if T_out > T_target:
                mel_out = mel_out[:, :, :T_target]
                stop_out = stop_out[:, :T_target]
            elif T_out < T_target:
                mels = mels[:, :, :T_out]
                stop_out = stop_out[:, :T_out]

            mel_loss = criterion_mel(mel_out, mels)
            stop_targets = torch.zeros_like(stop_out)
            stop_loss = criterion_stop(stop_out, stop_targets)
            loss = mel_loss + stop_loss

            total_loss += loss.item()
            total_mel_loss += mel_loss.item()
            total_stop_loss += stop_loss.item()

    n_batches = len(loader)
    return total_loss / n_batches, total_mel_loss / n_batches, total_stop_loss / n_batches

def main():
    # ================================
    # Configuración
    # ================================
    
    mlflow_config(exp_name="tacotron2_training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VOCAB = {ch: i for i, ch in enumerate("abcdefghijklmnopqrstuvwxyzáéíóúñü,.!? ")}
    out_path = "data/processed/css10"
    split_dir = f"{out_path}/splits"

    # ================================
    # Dataset y DataLoader
    # ================================
    split_dataset(out_path + "/metadata.csv", split_dir)
    train_dataset = TTSDataset(f"{split_dir}/train_metadata.csv", out_path)
    val_dataset = TTSDataset(f"{split_dir}/val_metadata.csv", out_path)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # ================================
    # Modelo y optimización
    # ================================
    model = Tacotron2(vocab_size=len(VOCAB), n_mels=80).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion_mel = nn.L1Loss()
    criterion_stop = nn.BCEWithLogitsLoss()

    mlflow.log_params({
        "epochs": 5,
        "batch_size": 4,
        "lr": 1e-3,
        "vocab_size": len(VOCAB)
    })

    # ================================
    # Entrenamiento
    # ================================
    
    if mlflow.active_run() is not None:
        mlflow.end_run()
    
    run_name = f"tacotron2_train_{uuid.uuid4().hex[:8]}"
    with mlflow.start_run(run_name=run_name):
        for epoch in range(1, 6):
            train_loss, train_mel, train_stop = train_one_epoch(
                model, train_loader, optimizer, criterion_mel, criterion_stop, device
            )
            val_loss, val_mel, val_stop = validate(
                model, val_loader, criterion_mel, criterion_stop, device
            )

            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_mel_loss": train_mel,
                "train_stop_loss": train_stop,
                "val_loss": val_loss,
                "val_mel_loss": val_mel,
                "val_stop_loss": val_stop
            }, step=epoch)

        # guardar modelo
        torch.save(model.state_dict(), "tacotron2_checkpoint.pt")
        mlflow.log_artifact("tacotron2_checkpoint.pt")

    # ================================
    # Cleanup
    # ================================
    mlruns_path = Path("mlruns")
    if mlruns_path.exists() and mlruns_path.is_dir():
        shutil.rmtree(mlruns_path)
        print("Directorio 'mlruns' eliminado automáticamente 🧹")


if __name__ == "__main__":
    main()
