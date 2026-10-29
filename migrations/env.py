from alembic import context
from sqlalchemy import MetaData, create_engine

from causal_agent.common.config import ingestion
from causal_agent.server.context.schema import metadata as context_metadata
from causal_agent.server.conversations.schema import metadata as chat_metadata
from causal_agent.server.ingestion.schema import metadata as ingestion_metadata
from causal_agent.server.profiling.schema import metadata as profiling_metadata

metadata = MetaData()
for source in (ingestion_metadata, profiling_metadata, context_metadata, chat_metadata):
    for table in source.tables.values():
        if table.key not in metadata.tables:
            table.to_metadata(metadata)

with create_engine(ingestion().database_url).connect() as connection:
    context.configure(connection=connection, target_metadata=metadata, include_schemas=True)
    with context.begin_transaction():
        context.run_migrations()
