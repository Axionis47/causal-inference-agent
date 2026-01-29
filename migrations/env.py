from alembic import context

from backend.db import engine
from backend.models import Base

with engine().connect() as connection:
    context.configure(
        connection=connection,
        target_metadata=Base.metadata,
        compare_type=True,
        include_object=lambda obj, name, kind, reflected, compare_to: (
            not (name.startswith("procrastinate_") or name.startswith("checkpoint")) if name else True
        ),
    )
    with context.begin_transaction():
        context.run_migrations()
