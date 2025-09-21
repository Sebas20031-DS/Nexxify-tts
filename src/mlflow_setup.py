import mlflow
from pathlib import Path

def create_experiment_with_postgres_and_artifact(exp_name: str, tracking_uri: str, artifact_root: str ) -> str:
    """
    Crea un experimento en MLflow con backend y una ubicación específica para los artefactos.

    Args:
        exp_name (str): Nombre del experimento.
        tracking_uri (str): URI de seguimiento para MLflow (debe ser una URI válida para Mlflow).
        artifact_root (str): Ruta del artifact root (debe ser una URI válida).

    Returns:
        str: ID del experimento creado.
    """
    # Configura MLflow para usar PostgreSQL como backend y Artifacts como ubicación de artefactos
    mlflow.set_tracking_uri(tracking_uri)


    existing = mlflow.get_experiment_by_name(exp_name)
    if existing is not None:
        print(f"El experimento '{exp_name}' ya existe con ID {existing.experiment_id}.")
        return existing.experiment_id
    else:
        # Crear nuevo experimento
        new_exp_id = mlflow.create_experiment(
            name=exp_name,
            artifact_location=artifact_root
        )
        return new_exp_id
