import pandas as pd
from prefect import flow, task
from prefect.logging import get_run_logger


@task
def validate_df(path: str) -> dict:
    """
    Controlla:
    - numero di righe
    - nulli per colonna
    - flag ok/not ok (solo se row_count > 0)
    """
    df = pd.read_parquet(path)
    row_count = len(df)
    nulls_per_col = df.isna().sum().to_dict()

    checks = {
        "path": path,
        "row_count": row_count,
        "nulls_per_col": nulls_per_col,
        "ok": row_count > 0,
    }
    return checks


@flow(name="validation_flow")
def validation_flow(processed_paths: list[str]):
    """
    Subflow di validazione: esegue validate_df su ogni file.
    """
    logger = get_run_logger()
    logger.info(f"[VALIDATION] Valido {len(processed_paths)} file processed")

    results_futures = [validate_df.submit(p) for p in processed_paths]
    return results_futures, processed_paths
