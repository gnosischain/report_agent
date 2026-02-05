"""
DEPRECATED: This module is deprecated in favor of report_agent.config.

Use the new typed configuration instead:
    from report_agent.config import get_config, AppConfig
    
    config = get_config()
    config.validate(require_llm=True, require_db=True)
    
    # Access typed config
    host = config.clickhouse.host
    api_key = config.llm.api_key

This module is kept for backward compatibility but will be removed in a future version.
"""
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def load_configs() -> dict:
    """
    DEPRECATED: Use report_agent.config.get_config() instead.
    
    Loads:
      • ClickHouse connection (read/write DBs)
      • LLM settings (provider, model, API keys)
      • dbt docs location (URL or local manifest path)
    """
    warnings.warn(
        "load_configs() is deprecated. Use report_agent.config.get_config() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    pkg_root = Path(__file__).resolve().parent.parent

    clickhouse = {
        "host": os.getenv("CLICKHOUSE_HOST"),
        "port": os.getenv("CLICKHOUSE_PORT"),  # Optional; clickhouse-connect uses defaults if not provided
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


def validate_config(cfg: dict, require_llm: bool = True, require_db: bool = True) -> None:
    """
    DEPRECATED: Use AppConfig.validate() instead.
    
    Validate that required configuration values are present.
    
    Args:
        cfg: Configuration dict from load_configs()
        require_llm: If True, validate that OpenAI API key is present
        require_db: If True, validate that ClickHouse credentials are present
        
    Raises:
        ValueError: If required configuration is missing
    """
    warnings.warn(
        "validate_config() is deprecated. Use AppConfig.validate() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    errors = []
    
    if require_llm:
        api_key = os.getenv("OPENAI_API_KEY") or cfg["llm"]["api_key"]
        if not api_key:
            errors.append("OPENAI_API_KEY not found in environment or config")
        if not cfg["llm"]["model"]:
            errors.append("OPENAI_MODEL not configured")
    
    if require_db:
        if not cfg["clickhouse"]["host"]:
            errors.append("CLICKHOUSE_HOST not found in environment")
        if not cfg["clickhouse"]["user"]:
            errors.append("CLICKHOUSE_USER not found in environment")
        if not cfg["clickhouse"]["password"]:
            errors.append("CLICKHOUSE_PASSWORD not found in environment")
    
    if errors:
        error_msg = "Configuration validation failed:\n  - " + "\n  - ".join(errors)
        error_msg += "\n\nPlease check your .env file or environment variables."
        raise ValueError(error_msg)
