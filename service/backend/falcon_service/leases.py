"""Heartbeats keep long inference calls recoverable without duplicate publishing."""

import threading
from contextlib import contextmanager
from datetime import timedelta

from sqlalchemy import update

from .db import now


@contextmanager
def heartbeat(db, model, identifier: str, claim_token: str, lease_seconds: int):
    stop = threading.Event()

    def renew():
        while not stop.wait(max(1, lease_seconds / 3)):
            try:
                with db.transaction() as session:
                    changed = session.execute(
                        update(model)
                        .where(
                            model.id == identifier,
                            model.claim_token == claim_token,
                        )
                        .values(lease_until=now() + timedelta(seconds=lease_seconds))
                    )
                    if changed.rowcount != 1:
                        return
            except Exception:
                # Fencing on final commit prevents this worker publishing after a
                # lease takeover even if the database was temporarily unavailable.
                continue

    thread = threading.Thread(target=renew, name="falcon-lease", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)
