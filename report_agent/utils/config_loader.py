import os
from pathlib import Path

from dotenv import load_dotenv
import yaml

load_dotenv()

def load_configs() -> dict:
    """
    Loads:
      • ClickHouse connection (read/write DBs)
      • Gemini API key
      • DBT docs location (URL or local path)
      • Metric definitions from configs/metrics.yml
    """
    pkg_root = Path(__file__).resolve().parent.parent

    clickhouse = {
        "host":     os.getenv("CLICKHOUSE_HOST"),
        #"port":     int(os.getenv("CLICKHOUSE_PORT", 8443)),
        "user":     os.getenv("CLICKHOUSE_USER"),
        "password": os.getenv("CLICKHOUSE_PASSWORD"),
        "db_read":  os.getenv("CLICKHOUSE_DB_READ", "dbt"),
        "db_write": os.getenv("CLICKHOUSE_DB_WRITE", "playground_max"),
        "secure": os.getenv("CLICKHOUSE_SECURE", "false").lower() in ("1","true")
    }

    llm = {
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
    }

    dbt_docs = {
        "base_url":   os.getenv("DBT_DOCS_BASE_URL")
    }

    metrics_path = pkg_root / "configs" / "metrics.yml"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics YAML at {metrics_path}")
    with metrics_path.open() as f:
        metrics_cfg = yaml.safe_load(f)
    metrics = metrics_cfg.get("metrics", [])

    return {
        "clickhouse": clickhouse,
        "llm": llm,
        "dbt_docs": dbt_docs,
        "metrics": metrics,
    }
