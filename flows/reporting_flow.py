import json
from pathlib import Path

import requests
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact
from prefect.blocks.system import Secret

from flows.storage_utils import upload_to_remote_storage

REPORTS_DIR = Path("reports/llm")


@task
def build_prompt(run_date: str, validation_results, metrics: dict, best_model_path: str) -> str:
    """
    Costruisce il prompt da mandare al modello LLM.
    """
    return f"""
Ho eseguito una pipeline dati con Prefect nel run_date: {run_date}.

RISULTATI VALIDAZIONE (estratto grezzo):
{validation_results}

METRICHE MODELLO:
{metrics}

MODELLO SALVATO IN:
{best_model_path}

Scrivi un report tecnico IN ITALIANO, massimo ~250 parole, che:
- riassuma la qualita dei dati (spiega i controlli in modo comprensibile)
- interpreti la metrica RMSE e il numero di alberi
- suggerisca 2-3 possibili miglioramenti per dati e modello
- stile sintetico ma chiaro, a bullet point dove utile.
"""


@task(retries=3, retry_delay_seconds=10)
def call_llm_openrouter(prompt: str) -> str:
    """
    Chiama l'API OpenRouter usando un Secret Block Prefect per la API key.
    """
    secret_block = Secret.load("openrouter-api-key")
    api_key = secret_block.get()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY non presente nel Secret Block")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "openai/gpt-oss-20b:free",  # modello LLM selezionato
        "messages": [
            {
                "role": "system",
                "content": "Sei un data engineer che spiega in modo chiaro e sintetico.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload),
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    # schema compatibile OpenAI: choices[0].message.content
    return data["choices"][0]["message"]["content"]


@task
def save_report(run_date: str, content: str) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"report_{run_date}.md"
    out_path.write_text(content, encoding="utf-8")
    remote_uri = upload_to_remote_storage(out_path, "reports/llm")
    markdown = (
        f"**Report LLM**\n\n"
        f"- file locale: `{out_path}`\n"
        f"- run_date: {run_date}"
    )
    if remote_uri:
        markdown += f"\n- download: [report_{run_date}.md]({remote_uri})"
    create_markdown_artifact(
        key=f"llm-report-{run_date}",
        markdown=markdown,
        description="Report LLM generato dal flow di reporting.",
    )
    return str(out_path)


@flow(name="reporting_flow")
def reporting_flow(run_date: str, validation_results, metrics: dict, best_model_path: str):
    """
    Subflow di reporting: prepara il prompt, chiama LLM, salva report .md.
    """
    logger = get_run_logger()
    prompt = build_prompt(run_date, validation_results, metrics, best_model_path)
    report_text = call_llm_openrouter(prompt)
    path = save_report(run_date, report_text)
    logger.info(f"[REPORTING] Report LLM salvato in {path}")
