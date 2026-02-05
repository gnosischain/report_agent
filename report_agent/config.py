# report_agent/config.py
"""
Centralized application configuration.

This module provides typed, validated, immutable configuration that loads once
and can be passed explicitly to components that need it.

Usage:
    from report_agent.config import get_config, AppConfig
    
    # Get singleton config (loads from environment)
    config = get_config()
    
    # Or create explicitly for testing
    config = AppConfig(
        clickhouse=ClickHouseConfig(host="localhost", ...),
        llm=LLMConfig(api_key="test-key"),
        dbt_docs=DbtDocsConfig(),
    )
    
    # Validate before use
    config.validate(require_llm=True, require_db=True)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class ClickHouseConfig:
    """ClickHouse database connection configuration."""
    host: str
    user: str
    password: str
    db_read: str = "dbt"
    db_write: str = "playground_max"
    port: Optional[int] = None
    secure: bool = False
    verify: bool = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        return {
            "host": self.host,
            "user": self.user,
            "password": self.password,
            "db_read": self.db_read,
            "db_write": self.db_write,
            "port": self.port,
            "secure": self.secure,
            "verify": self.verify,
        }


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider configuration."""
    provider: str = "openai"
    model: str = "gpt-4.1"
    api_key: str = ""
    gemini_api_key: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key": self.api_key,
            "openai_api_key": self.api_key,  # Alias for backward compat
            "gemini_api_key": self.gemini_api_key,
        }


@dataclass(frozen=True)
class DbtDocsConfig:
    """dbt documentation configuration."""
    base_url: Optional[str] = None
    manifest_path: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        return {
            "base_url": self.base_url,
            "manifest_path": self.manifest_path,
        }


@dataclass(frozen=True)
class AppConfig:
    """
    Application configuration - loaded once, passed everywhere.
    
    This is an immutable (frozen) dataclass that holds all application
    configuration. Use `from_env()` to load from environment variables,
    or construct directly for testing.
    """
    clickhouse: ClickHouseConfig
    llm: LLMConfig
    dbt_docs: DbtDocsConfig
    
    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> AppConfig:
        """
        Load configuration from environment variables.
        
        Args:
            env_file: Optional path to .env file. If not provided,
                     uses python-dotenv's default discovery.
        
        Returns:
            AppConfig instance with values from environment.
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        # Package root for default paths
        pkg_root = Path(__file__).resolve().parent
        
        # Parse port if provided
        port_str = os.getenv("CLICKHOUSE_PORT")
        port = int(port_str) if port_str else None
        
        return cls(
            clickhouse=ClickHouseConfig(
                host=os.getenv("CLICKHOUSE_HOST", ""),
                user=os.getenv("CLICKHOUSE_USER", ""),
                password=os.getenv("CLICKHOUSE_PASSWORD", ""),
                db_read=os.getenv("CLICKHOUSE_DB_READ", "dbt"),
                db_write=os.getenv("CLICKHOUSE_DB_WRITE", "playground_max"),
                port=port,
                secure=os.getenv("CLICKHOUSE_SECURE", "false").lower() in ("1", "true"),
                verify=os.getenv("CLICKHOUSE_VERIFY", "true").lower() in ("1", "true"),
            ),
            llm=LLMConfig(
                provider=os.getenv("LLM_PROVIDER", "openai"),
                model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
                api_key=os.getenv("OPENAI_API_KEY", ""),
                gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            ),
            dbt_docs=DbtDocsConfig(
                base_url=os.getenv("DBT_DOCS_BASE_URL"),
                manifest_path=os.getenv("DBT_MANIFEST_PATH")
                or str(pkg_root / "dbt_context" / "manifest.json"),
            ),
        )
    
    def validate(self, require_llm: bool = True, require_db: bool = True) -> None:
        """
        Validate that required configuration values are present.
        
        Args:
            require_llm: If True, validate that LLM API key is present
            require_db: If True, validate that ClickHouse credentials are present
        
        Raises:
            ValueError: If required configuration is missing
        """
        errors = []
        
        if require_llm:
            if not self.llm.api_key:
                errors.append("OPENAI_API_KEY not found in environment")
            if not self.llm.model:
                errors.append("OPENAI_MODEL not configured")
        
        if require_db:
            if not self.clickhouse.host:
                errors.append("CLICKHOUSE_HOST not found in environment")
            if not self.clickhouse.user:
                errors.append("CLICKHOUSE_USER not found in environment")
            if not self.clickhouse.password:
                errors.append("CLICKHOUSE_PASSWORD not found in environment")
        
        if errors:
            error_msg = "Configuration validation failed:\n  - " + "\n  - ".join(errors)
            error_msg += "\n\nPlease check your .env file or environment variables."
            raise ValueError(error_msg)
    
    def to_dict(self) -> dict:
        """
        Convert to nested dictionary for backward compatibility.
        
        This matches the structure returned by the old load_configs() function.
        """
        return {
            "clickhouse": self.clickhouse.to_dict(),
            "llm": self.llm.to_dict(),
            "dbt_docs": self.dbt_docs.to_dict(),
        }


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """
    Get the singleton application config.
    
    This function is cached, so it only loads from environment once per process.
    For testing, construct AppConfig directly instead of using this function.
    
    Returns:
        AppConfig instance loaded from environment variables.
    """
    return AppConfig.from_env()


def clear_config_cache() -> None:
    """
    Clear the cached config.
    
    Useful for testing when you need to reload config from changed environment.
    """
    get_config.cache_clear()
