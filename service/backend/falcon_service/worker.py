"""Separate worker process. A lease makes interrupted work recoverable."""

import logging
import signal
import sys
import threading
import time
from pathlib import Path

from sqlalchemy import text

from .config import Settings
from .db import Database
from .jobs import run_once
from .logging_setup import configure_logging
from .runtime import EncoderRuntime
from .storage import ArtifactStore

HEARTBEAT_FILE = Path("/tmp/falcon-worker-heartbeat")
logger = logging.getLogger(__name__)


def healthcheck() -> int:
    """Container probe: active worker heartbeat and current dependencies."""
    try:
        if time.time() - HEARTBEAT_FILE.stat().st_mtime > 35:
            return 1
        settings = Settings()
        database = Database(settings.database_url)
        try:
            with database.transaction() as session:
                session.execute(text("SELECT 1"))
        finally:
            database.close()
        runtime = EncoderRuntime(settings)
        runtime.initialize()
        return 0 if runtime.encoder is not None else 1
    except Exception:
        return 1


def main():
    if "--healthcheck" in sys.argv:
        raise SystemExit(healthcheck())
    configure_logging()
    settings = Settings()
    database = Database(settings.database_url)
    database.initialize()
    store = ArtifactStore(settings.storage_root)
    runtime = EncoderRuntime(settings)
    runtime.initialize()
    stopped = threading.Event()

    def worker_heartbeat():
        while not stopped.is_set():
            HEARTBEAT_FILE.touch()
            stopped.wait(10)

    heartbeat_thread = threading.Thread(target=worker_heartbeat, name="worker-health", daemon=True)
    heartbeat_thread.start()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stopped.set())
    try:
        while not stopped.is_set():
            if runtime.encoder is None:
                runtime.initialize()
            try:
                processed = run_once(database, store, runtime, settings)
            except Exception:
                logger.exception("Unable to claim a durable job")
                processed = False
            if not processed:
                stopped.wait(settings.worker_poll_seconds)
    finally:
        stopped.set()
        heartbeat_thread.join(timeout=2)
        database.close()


if __name__ == "__main__":
    main()
