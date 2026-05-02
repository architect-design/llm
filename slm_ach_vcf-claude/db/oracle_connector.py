"""
Oracle Database Connector for FinSLM
Uses oracledb (thin driver — no Oracle Client install needed).
Falls back gracefully to MOCK mode when DB is unavailable so the app
always runs even without a live Oracle instance.

Supported drivers (in preference order):
  1. oracledb  (python-oracledb >= 1.0, thin mode — recommended)
  2. cx_Oracle (legacy thick-mode driver)
  3. MOCK      (synthetic data, no driver needed)
"""

import os
import json
import time
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple

log = logging.getLogger(__name__)

# ── Try to import the best available Oracle driver ────────────────────────────
try:
    import oracledb
    oracledb.init_oracle_client()          # thick mode if client found
    _DRIVER = "oracledb-thick"
except Exception:
    try:
        import oracledb                    # thin mode (no client)
        _DRIVER = "oracledb-thin"
    except ImportError:
        try:
            import cx_Oracle as oracledb   # legacy fallback
            _DRIVER = "cx_Oracle"
        except ImportError:
            oracledb = None
            _DRIVER = "mock"

log.info("Oracle driver: %s", _DRIVER)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OracleConfig:
    """
    All connection parameters.  Values are read from environment variables
    so nothing sensitive lives in source code.

    Environment variables (all optional — defaults shown):
      ORACLE_HOST          localhost
      ORACLE_PORT          1521
      ORACLE_SERVICE       ORCL
      ORACLE_USER          finslm
      ORACLE_PASSWORD      (empty)
      ORACLE_WALLET_DIR    (empty — mTLS wallet path)
      ORACLE_WALLET_PWD    (empty)
      ORACLE_POOL_MIN      2
      ORACLE_POOL_MAX      10
      ORACLE_POOL_INC      1
      ORACLE_TIMEOUT_S     30
      ORACLE_SCHEMA        FINSLM   (default schema/owner prefix)
      ORACLE_MOCK          false    (set 'true' to force mock mode)
    """
    host:        str = os.getenv("ORACLE_HOST",       "localhost")
    port:        int = int(os.getenv("ORACLE_PORT",   "1521"))
    service:     str = os.getenv("ORACLE_SERVICE",    "ORCL")
    user:        str = os.getenv("ORACLE_USER",       "finslm")
    password:    str = os.getenv("ORACLE_PASSWORD",   "")
    wallet_dir:  str = os.getenv("ORACLE_WALLET_DIR", "")
    wallet_pwd:  str = os.getenv("ORACLE_WALLET_PWD", "")
    pool_min:    int = int(os.getenv("ORACLE_POOL_MIN", "2"))
    pool_max:    int = int(os.getenv("ORACLE_POOL_MAX", "10"))
    pool_inc:    int = int(os.getenv("ORACLE_POOL_INC", "1"))
    timeout_s:   int = int(os.getenv("ORACLE_TIMEOUT_S", "30"))
    schema:      str = os.getenv("ORACLE_SCHEMA",    "FINSLM")
    mock:        bool = os.getenv("ORACLE_MOCK", "false").lower() == "true"

    @property
    def dsn(self) -> str:
        return f"{self.host}:{self.port}/{self.service}"

    def to_safe_dict(self) -> dict:
        """Return config dict with password masked."""
        d = asdict(self)
        d["password"] = "***" if self.password else "(empty)"
        d["wallet_pwd"] = "***" if self.wallet_pwd else "(empty)"
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Connection Pool
# ─────────────────────────────────────────────────────────────────────────────

class OracleConnectionPool:
    """
    Thread-safe Oracle connection pool.
    Automatically degrades to MOCK mode if the driver is missing or the
    database is unreachable.
    """

    _instance: Optional["OracleConnectionPool"] = None
    _lock = threading.Lock()

    def __init__(self, config: OracleConfig):
        self.config = config
        self._pool = None
        self._mock = config.mock or (_DRIVER == "mock")
        self._connected = False
        self._last_error: Optional[str] = None
        self._connect_time: Optional[float] = None

        if not self._mock:
            self._init_pool()

    # ── Singleton factory ────────────────────────────────────────────────────
    @classmethod
    def get_instance(cls, config: Optional[OracleConfig] = None) -> "OracleConnectionPool":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config or OracleConfig())
            return cls._instance

    @classmethod
    def reset(cls):
        with cls._lock:
            if cls._instance and cls._instance._pool:
                try:
                    cls._instance._pool.close()
                except Exception:
                    pass
            cls._instance = None

    # ── Pool initialisation ──────────────────────────────────────────────────
    def _init_pool(self):
        cfg = self.config
        try:
            kwargs: Dict[str, Any] = dict(
                user=cfg.user,
                password=cfg.password,
                dsn=cfg.dsn,
                min=cfg.pool_min,
                max=cfg.pool_max,
                increment=cfg.pool_inc,
                getmode=oracledb.POOL_GETMODE_WAIT,
                wait_timeout=cfg.timeout_s * 1000,
                timeout=cfg.timeout_s,
                encoding="UTF-8",
            )
            if cfg.wallet_dir:
                kwargs["wallet_location"] = cfg.wallet_dir
                kwargs["wallet_password"] = cfg.wallet_pwd

            self._pool = oracledb.create_pool(**kwargs)
            self._connected = True
            self._connect_time = time.time()
            log.info("Oracle pool created: %s (driver=%s)", cfg.dsn, _DRIVER)
        except Exception as exc:
            self._last_error = str(exc)
            self._mock = True
            log.warning(
                "Oracle pool failed (%s) — falling back to MOCK mode. Error: %s",
                cfg.dsn, exc
            )

    # ── Context manager ──────────────────────────────────────────────────────
    @contextmanager
    def connection(self):
        """
        Yields a database connection.
        In MOCK mode yields a MockConnection instead.
        """
        if self._mock:
            yield MockConnection()
            return

        conn = None
        try:
            conn = self._pool.acquire()
            yield conn
        except Exception as exc:
            log.error("Oracle connection error: %s", exc)
            self._last_error = str(exc)
            raise
        finally:
            if conn:
                try:
                    self._pool.release(conn)
                except Exception:
                    pass

    # ── Health check ────────────────────────────────────────────────────────
    def health(self) -> dict:
        status = {
            "driver":       _DRIVER,
            "mode":         "mock" if self._mock else "live",
            "connected":    self._connected,
            "dsn":          self.config.dsn,
            "schema":       self.config.schema,
            "pool_min":     self.config.pool_min,
            "pool_max":     self.config.pool_max,
            "last_error":   self._last_error,
            "connect_time": self._connect_time,
        }
        if not self._mock and self._pool:
            try:
                status["pool_open"] = self._pool.opened
                status["pool_busy"] = self._pool.busy
            except Exception:
                pass
        return status

    def test_connection(self) -> Tuple[bool, str]:
        """Ping the database. Returns (success, message)."""
        if self._mock:
            return True, "MOCK mode — no live DB connection"
        try:
            with self.connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT 1 FROM DUAL")
                cur.fetchone()
            return True, "Oracle connection successful"
        except Exception as exc:
            return False, f"Connection failed: {exc}"

    @property
    def is_mock(self) -> bool:
        return self._mock


# ─────────────────────────────────────────────────────────────────────────────
# Mock connection (used when Oracle is unavailable)
# ─────────────────────────────────────────────────────────────────────────────

class MockCursor:
    """Returns empty result sets; callers handle gracefully."""
    def __init__(self):
        self.description = []
        self._rows: List = []

    def execute(self, sql: str, params=None):
        self._rows = []

    def executemany(self, sql: str, params=None):
        pass

    def fetchall(self) -> List:
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchmany(self, n=100):
        return self._rows[:n]

    def close(self):
        pass

    def __enter__(self): return self
    def __exit__(self, *a): self.close()


class MockConnection:
    def cursor(self): return MockCursor()
    def commit(self): pass
    def rollback(self): pass
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helper
# ─────────────────────────────────────────────────────────────────────────────

def get_pool(config: Optional[OracleConfig] = None) -> OracleConnectionPool:
    """Return the singleton pool (creates it on first call)."""
    return OracleConnectionPool.get_instance(config)


def execute_query(
    sql: str,
    params=None,
    config: Optional[OracleConfig] = None,
    fetchall: bool = True,
) -> List[Dict[str, Any]]:
    """
    Execute a SELECT and return a list of dicts (column_name → value).
    Handles both mock and live connections transparently.
    """
    pool = get_pool(config)
    with pool.connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, params or {})
        if not cur.description:
            return []
        cols = [d[0].lower() for d in cur.description]
        rows = cur.fetchall() if fetchall else cur.fetchmany(1000)
        return [dict(zip(cols, row)) for row in rows]


def execute_dml(
    sql: str,
    params=None,
    config: Optional[OracleConfig] = None,
) -> int:
    """Execute INSERT/UPDATE/DELETE and commit. Returns rowcount."""
    pool = get_pool(config)
    with pool.connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, params or {})
        rowcount = getattr(cur, "rowcount", 0)
        conn.commit()
        return rowcount
