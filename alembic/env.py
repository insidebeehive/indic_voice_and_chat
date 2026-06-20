"""Alembic env — async-aware, sources URL from app config / env."""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy.schema import CreateSchema

# Import models so their metadata is registered against Base.
from src.config import load_settings, normalize_db_url
from src.models import Base  # noqa: F401  (side-effect: registers all tables)
from src.models.database import search_path_connect_args

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Resolve DB URL: explicit -x url=..., else app settings.
_settings = load_settings()
_x = context.get_x_argument(as_dictionary=True)
db_url = _x.get("url") or config.get_main_option("sqlalchemy.url")
if not db_url:
    db_url = _settings.database.url
db_url = normalize_db_url(db_url)
config.set_main_option("sqlalchemy.url", db_url)

target_metadata = Base.metadata

# All tables (and alembic_version) live under our schema, except on SQLite
# which has no schemas. The translate map rewrites the default schema to ours.
DB_SCHEMA = None if db_url.startswith("sqlite") else (
    getattr(_settings.database, "db_schema", None) or None)


def run_migrations_offline() -> None:
    context.configure(
        url=db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table_schema=DB_SCHEMA,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    # search_path is pinned at connect time (server_settings), so the
    # migrations' unqualified table names — CREATE *and* ALTER — resolve to our
    # schema. version_table_schema keeps alembic_version there too.
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table_schema=DB_SCHEMA,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connect_args = search_path_connect_args(db_url)
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )

    # Ensure the schema exists, then run the migrations. ``CREATE SCHEMA IF NOT
    # EXISTS`` still needs database-level CREATE in Postgres even when the schema
    # is already there, so we check existence first and only attempt the CREATE
    # when it's actually missing. This lets a limited deploy role (with rights on
    # an already-provisioned schema, but not database-level CREATE) run migrations;
    # first-time provisioning still needs a privileged role.
    if DB_SCHEMA:
        async with connectable.begin() as conn:
            exists = await conn.run_sync(
                lambda c: c.execute(
                    text("SELECT 1 FROM information_schema.schemata WHERE schema_name = :s"),
                    {"s": DB_SCHEMA},
                ).first() is not None)
            if not exists:
                await conn.run_sync(
                    lambda c: c.execute(CreateSchema(DB_SCHEMA, if_not_exists=True)))

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
