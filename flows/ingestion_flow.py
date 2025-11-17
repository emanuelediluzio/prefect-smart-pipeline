from prefect import flow, task
from prefect.logging import get_run_logger
from pathlib import Path
import pandas as pd
import numpy as np

DATA_RAW_DIR = Path("data/raw")
DATA_PROCESSED_DIR = Path("data/processed")


@task
def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    """
    Se il file CSV esiste, lo legge e lo restituisce.
    Altrimenti ritorna None.
    """
    if path.exists():
        return pd.read_csv(path)
    return None


@task
def save_parquet(df: pd.DataFrame, filename: str) -> str:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / filename
    df.to_parquet(out_path, index=False)
    return str(out_path)


@flow(name="ingestion_flow")
def ingestion_flow(run_date: str) -> list[str]:
    """
    Subflow di ingestion.

    Logica:
    - cerca un CSV in data/raw/dataset_<run_date>.csv
    - se c'è, lo usa come input reale
    - se NON c'è, genera un dataset sintetico
    - in ogni caso salva un parquet in data/processed/
    """
    logger = get_run_logger()

    raw_csv_path = DATA_RAW_DIR / f"dataset_{run_date}.csv"
    logger.info(f"[INGESTION] Cerco CSV reale in: {raw_csv_path}")

    df = read_csv_if_exists(raw_csv_path)
    logger.info(f"[INGESTION] Trovato CSV reale con shape={df.shape}")

    out_path = save_parquet(df, f"dataset_{run_date}.parquet")
    logger.info(f"[INGESTION] Dataset (reale o sintetico) salvato in {out_path}")

    return [out_path]
