from prefect import flow, task
from prefect.events import emit_event
from prefect.logging import get_run_logger

from flows.ingestion_flow import ingestion_flow
from flows.validation_flow import validation_flow
from flows.training_flow import training_flow
from flows.reporting_flow import reporting_flow


@task
def emit_pipeline_finished_event(run_date: str):
    """
    Emette un evento Prefect a fine pipeline.
    """
    emit_event(
        event="pipeline.finished",
        resource={
            "prefect.resource.id": f"pipeline-run-{run_date}",
            "prefect.resource.name": "Smart Prefect Pipeline",
        },
        payload={"run_date": run_date},
    )


@flow(name="main_orchestrator_flow")
def main_orchestrator_flow(run_date: str = "2025-11-17"):
    """
    Flow principale:
    1. ingestion
    2. validation
    3. training
    4. reporting (LLM)
    5. evento finale
    """
    logger = get_run_logger()
    logger.info(f"[ORCHESTRATOR] Avvio pipeline per run_date={run_date}")

    # 1) Ingestion
    processed_paths = ingestion_flow(run_date=run_date)

    # 2) Validation
    validation_results_futures, clean_data_paths = validation_flow(processed_paths)
    validation_results = [f.result() for f in validation_results_futures]

    # 3) Training
    metrics, best_model_path = training_flow(clean_data_paths)

    # 4) Reporting (LLM)
    reporting_flow(
        run_date=run_date,
        validation_results=validation_results,
        metrics=metrics,
        best_model_path=best_model_path,
    )

    # 5) Evento di fine pipeline
    emit_pipeline_finished_event.submit(run_date)

    logger.info("[ORCHESTRATOR] Pipeline completata con successo")


if __name__ == "__main__":
    main_orchestrator_flow()
