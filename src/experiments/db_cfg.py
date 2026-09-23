import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# optional dependency dotenv
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs) -> bool:
        logger.info("python-dotenv is not installed; skipping loading of .env files.")
        return False

BASE_DIR = Path(__file__).resolve().parents[2]
dotenv_path = BASE_DIR / ".env"

if dotenv_path.exists():
    loaded = load_dotenv(dotenv_path)
    if loaded:
        logger.info("Loaded environment variables from %s", dotenv_path)
    else:
        logger.warning(
            "Attempted to load .env from %s but load_dotenv returned False.",
            dotenv_path,
        )
else:
    logger.info("No .env file found at %s.", dotenv_path)


@dataclass
class DBConfig:
    user: Optional[str] = None
    password: Optional[str] = None
    host: str = "localhost"
    port: int = 5432

    @property
    def is_configured(self) -> bool:
        """Return True if we have enough info to connect."""
        return bool(self.user and self.password)

    def url(self, db_name: str) -> str:
        if not self.is_configured:
            logger.error("Database configuration incomplete (PGUSER or PGPASSWORD missing).")
            raise RuntimeError(
                "Database configuration incomplete (PGUSER or PGPASSWORD missing)."
            )

        logger.info(
            "Building DB URL for db_name=%s (host=%s, port=%s, user=%s)",
            db_name,
            self.host,
            self.port,
            self.user,
        )
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{db_name}"


DB_SETTINGS = DBConfig(
    user=os.getenv("PGUSER"),
    password=os.getenv("PGPASSWORD"),
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
)

if DB_SETTINGS.is_configured:
    logger.info(
        "DB_SETTINGS initialized with host=%s, port=%s, user=%s",
        DB_SETTINGS.host,
        DB_SETTINGS.port,
        DB_SETTINGS.user,
    )
else:
    logger.warning(
        "DB_SETTINGS is incomplete (PGUSER or PGPASSWORD missing). "
        "DB-dependent functions will raise until credentials are provided."
    )


def db_env(*, base_env: Optional[dict] = None, db_name: Optional[str] = None) -> dict:
    """
    Return an environment dict with PG* variables set from DB_SETTINGS.
    """
    if base_env is None:
        base_env = os.environ

    if not DB_SETTINGS.is_configured:
        logger.error("Cannot construct DB env: DB_SETTINGS is incomplete.")
        raise RuntimeError("Cannot construct DB env: DB_SETTINGS is incomplete.")

    env = dict(base_env)

    env.update(
        PGHOST=DB_SETTINGS.host,
        PGPORT=str(DB_SETTINGS.port),
        PGUSER=DB_SETTINGS.user,
        PGPASSWORD=DB_SETTINGS.password,
    )

    if db_name is not None:
        env["PGDATABASE"] = db_name
        logger.debug("Setting PGDATABASE=%s in DB env.", db_name)
    else:
        logger.debug("No db_name provided to db_env().")

    return env


def get_optuna_db(db_name: Optional[str] = None) -> str:
    """
    Build a PostgreSQL URL for Optuna.

    db_name:
        - If provided, use this as the DB name.
        - Otherwise, use $PGDATABASE from the environment.
    """
    if db_name is None:
        db_name = os.getenv("PGDATABASE")

    if not db_name:
        logger.error(
            "No database name specified: PGDATABASE not set and no db_name "
            "argument passed to get_optuna_db()."
        )
        raise RuntimeError(
            "No database name specified: PGDATABASE not set and no db_name "
            "argument passed to get_optuna_db()."
        )

    if not DB_SETTINGS.is_configured:
        logger.error("Database configuration incomplete (PGUSER or PGPASSWORD missing).")
        raise RuntimeError(
            "Database configuration incomplete (PGUSER or PGPASSWORD missing)."
        )

    url = DB_SETTINGS.url(db_name)
    logger.info(
        "Using Optuna DB (host=%s, port=%s, user=%s, db=%s)",
        DB_SETTINGS.host,
        DB_SETTINGS.port,
        DB_SETTINGS.user,
        db_name,
    )
    return url