import logging
import clickhouse_connect
from clickhouse_connect.driver.exceptions import DatabaseError, OperationalError, ProgrammingError
import pandas as pd

from report_agent.utils.config_loader import load_configs

log = logging.getLogger(__name__)

class ClickHouseConnector:
    def __init__(self):
        cfg = load_configs()["clickhouse"]

        common = dict(
            host    = cfg["host"],
            #port    = cfg.get("port"),
            username= cfg["user"],
            password= cfg["password"],
            secure  = cfg["secure"],
            verify  = cfg.get("verify", True),
        )
        # read client
        self.read = clickhouse_connect.get_client(
            **common,
            database=cfg["db_read"],
        )
        # write client
        self.write = clickhouse_connect.get_client(
            **common,
            database=cfg["db_write"],
        )

    def _ensure_read_only(self, sql: str):
        stmt = sql.strip().lower()
        if not (stmt.startswith("select") or stmt.startswith("with")):
            raise ValueError("Only read-only queries allowed on the read client.")

    def execute_read(self, sql: str, params: dict = None):
        """
        Executes a read-only query and returns the raw QueryResult.
        Use .result_set or .column_names on it.
        """
        self._ensure_read_only(sql)
        log.debug("ClickHouse READ query: %s | params: %r", sql, params)
        try:
            return self.read.query(sql, params or {})
        except (DatabaseError, OperationalError, ProgrammingError) as e:
            log.error("ClickHouse error running SQL: %s — %s", sql, e)
            raise

    def execute_write(self, sql: str, params: dict = None):
        """
        Executes any query on the write client (e.g. inserts to playground_max).
        """
        log.debug("ClickHouse WRITE query: %s | params: %r", sql, params)
        try:
            return self.write.query(sql, params or {})
        except (DatabaseError, OperationalError, ProgrammingError) as e:
            log.error("ClickHouse error running SQL: %s — %s", sql, e)
            raise

    def fetch_df(self, sql: str, params: dict = None) -> pd.DataFrame:
        """
        Read-only convenience: runs sql, returns a DataFrame.
        """
        qr = self.execute_read(sql, params)
        rows = qr.result_set
        cols = qr.column_names
        return pd.DataFrame(rows, columns=cols)