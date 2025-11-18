from pathlib import Path

import joblib
import pandas as pd
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from prefect.events import emit_event
from prefect.logging import get_run_logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from flows.storage_utils import upload_to_remote_storage

MODELS_DIR = Path("data/models")
THRESHOLD_RMSE = 2.0  # soglia arbitraria per dire "modello buono/brutto"


@task
def train_with_params(data_path: str, n_estimators: int) -> dict:
    """
    Allena un RFRegressor su un singolo data_path con i parametri dati.
    """
    df = pd.read_parquet(data_path)

    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    X = df[feature_cols]
    y = df["target"]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
    )
    model.fit(X, y)
    preds = model.predict(X)
    rmse = mean_squared_error(y, preds) ** 0.5  # compatibile anche con versioni sklearn vecchie

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"rf_{n_estimators}.joblib"
    joblib.dump(model, model_path)

    return {
        "n_estimators": n_estimators,
        "rmse": rmse,
        "model_path": str(model_path),
    }


@task
def select_best_model(results: list[dict]) -> dict:
    """
    Seleziona il modello con RMSE più basso.
    """
    return sorted(results, key=lambda r: r["rmse"])[0]


@task
def emit_model_quality_event(rmse: float) -> None:
    """
    Emette un evento Prefect:
    - model.good se rmse < soglia
    - model.bad altrimenti
    """
    event_name = "model.good" if rmse < THRESHOLD_RMSE else "model.bad"
    emit_event(
        event=event_name,
        resource={
            "prefect.resource.id": "ml-model-quality",
            "prefect.resource.name": "ML model quality",
        },
        payload={
            "rmse": rmse,
            "threshold": THRESHOLD_RMSE,
        },
    )


@flow(name="training_flow")
def training_flow(clean_data_paths: list[str]):
    """
    Subflow di training:
    - prende il primo dataset
    - prova diversi n_estimators
    - sceglie il migliore
    - emette un evento di qualità modello
    """
    logger = get_run_logger()

    if not clean_data_paths:
        raise ValueError("Nessun dataset passato a training_flow")

    data_path = clean_data_paths[0]
    logger.info(f"[TRAINING] Uso dataset: {data_path}")

    params = [50, 100, 200]

    results_futures = [
        train_with_params.submit(data_path, n) for n in params
    ]
    # Risolviamo i future per poter confrontare le metriche
    results = [future.result() for future in results_futures]

    best = select_best_model(results)
    rmse = best["rmse"]
    best_model_path = best["model_path"]

    logger.info(f"[TRAINING] Best model n_estimators={best['n_estimators']} rmse={rmse:.4f}")

    remote_uri = upload_to_remote_storage(Path(best_model_path), "data/models")
    markdown = (
        f"**Modello RandomForest**\n\n"
        f"- file locale: `{best_model_path}`\n"
        f"- n_estimators: {best['n_estimators']}\n"
        f"- rmse: {rmse:.4f}"
    )
    if remote_uri:
        markdown += f"\n- download: [rf_{best['n_estimators']}.joblib]({remote_uri})"
    create_markdown_artifact(
        key=f"best-model-{best['n_estimators']}",
        markdown=markdown,
        description="Modello RandomForest addestrato dal training flow.",
    )

    # Evento di qualità
    emit_model_quality_event.submit(rmse)

    metrics = {"rmse": rmse, "n_estimators": best["n_estimators"]}
    return metrics, best_model_path
