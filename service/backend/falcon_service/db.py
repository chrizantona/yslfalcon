"""Relational persistence and explicit transaction boundaries."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool


def now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def timestamp(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds") + "Z"


class Base(DeclarativeBase):
    pass


class Database:
    def __init__(self, url: str):
        kwargs = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
            if ":memory:" in url:
                kwargs["poolclass"] = StaticPool
        self.engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):

            @event.listens_for(self.engine, "connect")
            def configure_sqlite(connection, _):
                cursor = connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def initialize(self) -> None:
        from . import models  # noqa: F401 — registers schema once at startup

        with self.engine.begin() as connection:
            if connection.dialect.name == "postgresql":
                # API and worker may boot together. Serialize initial schema DDL.
                connection.execute(text("SELECT pg_advisory_xact_lock(6842417041)"))
            Base.metadata.create_all(connection)

    @contextmanager
    def transaction(self) -> Iterator[Session]:
        with self.sessions.begin() as session:
            yield session

    def close(self) -> None:
        self.engine.dispose()
