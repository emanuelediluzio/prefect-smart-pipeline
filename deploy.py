from prefect import flow


if __name__ == "__main__":
    # Carica il flow dal repo GitHub
    remote_flow = flow.from_source(
        source="https://github.com/emanuelediluzio/prefect-smart-pipeline.git",
        entrypoint="flows/orchestrator.py:main_orchestrator_flow",
    )

    # Crea un deployment su Prefect Cloud usando il work pool gestito
    remote_flow.deploy(
        name="smart-pipeline-managed",
        work_pool_name="default-work-pool",
        job_variables={
            # Pacchetti che Prefect deve installare nel container
            "pip_packages": [
                "prefect>=3.0.0",
                "prefect-email",
                "pandas",
                "scikit-learn",
                "pyarrow",
                "fastparquet",
                "joblib",
                "python-dotenv",
                "requests",
            ]
        },
    )
