import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def load_configs() -> dict:
    """
    Loads:
      • ClickHouse connection (read/write DBs)
      • LLM settings (provider, model, API keys)
      • dbt docs location (URL or local manifest path)
    """
    pkg_root = Path(__file__).resolve().parent.parent

    clickhouse = {
        "host": os.getenv("CLICKHOUSE_HOST"),
        "user": os.getenv("CLICKHOUSE_USER"),
        "password": os.getenv("CLICKHOUSE_PASSWORD"),
        "db_read": os.getenv("CLICKHOUSE_DB_READ", "dbt"),
        "db_write": os.getenv("CLICKHOUSE_DB_WRITE", "playground_max"),
        "secure": os.getenv("CLICKHOUSE_SECURE", "false").lower() in ("1", "true"),
    }

    llm = {
        "provider": os.getenv("LLM_PROVIDER", "openai"),
        "model": os.getenv("OPENAI_MODEL", "gpt-4.1"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
    }

    dbt_docs = {
        "base_url": os.getenv("DBT_DOCS_BASE_URL"),
        "manifest_path": os.getenv("DBT_MANIFEST_PATH")
        or str(pkg_root / "dbt_context" / "manifest.json"),
    }

    return {
        "clickhouse": clickhouse,
        "llm": llm,
        "dbt_docs": dbt_docs,
    }
