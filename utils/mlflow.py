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


def mlflow_config(exp_name: str = "default_experiment",
                      tracking_uri: str = "postgresql://postgres:2724@localhost:5432/mlflow_db",
                      artifact_root: str = r"C:/Users/sebas/Documents/Projects/TTS/tts_project/mlflow/artifacts",
                      temp_dir: str = r"C:/Users/sebas/Documents/Projects/TTS/tts_project/mlflow/temp") -> str:
    """
    Configura el entorno de MLflow para un proyecto específico, crea un experimento y configura los parámetros necesarios.
    Los valores predeterminados pueden ser reemplazados con los valores proporcionados.

    Args:
        exp_name (str): Nombre del experimento. Valor por defecto: "default_experiment".
        tracking_uri (str): URI de seguimiento para MLflow. Valor por defecto: URI de PostgreSQL.
        artifact_root (str): Ruta de almacenamiento de artefactos. Valor por defecto: ruta local de artefactos.
        temp_dir (str): Ruta de la carpeta temporal para archivos antes de loggearlos. Valor por defecto: ruta local.

    Returns:
        str: ID del experimento creado o existente.
    """
    # ================================
    # Configuración de MLflow
    # ================================
    # Configura URI de tracking
    mlflow.set_tracking_uri(tracking_uri)

    # Asegura que la carpeta temporal exista
    temp_dir_path = Path(temp_dir)
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    # Crear el experimento usando la función que definimos anteriormente
    create_experiment_with_postgres_and_artifact(exp_name, tracking_uri, artifact_root)
    
    # Establecer el experimento actual en MLflow
    mlflow.set_experiment(exp_name)