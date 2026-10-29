"""Deployment operations: migrate, initialize private bucket, or reconcile uploads."""

import argparse
from pathlib import Path

from alembic import command
from alembic.config import Config
from botocore.exceptions import ClientError

from causal_agent.common.config import ingestion
from causal_agent.server.ingestion.service import Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["init", "cleanup"])
    args = parser.parse_args()
    settings = ingestion()
    storage = Store(settings)
    if args.operation == "cleanup":
        from causal_agent.server.conversations.publication import cleanup

        print(f"Cleaned {cleanup(storage)} abandoned exploration attempts")
        print(f"Cleaned {storage.cleanup()} abandoned uploads")
        return
    root = Path(__file__).resolve().parents[3]
    config = Config(str(root / "alembic.ini"))
    command.upgrade(config, "head")
    try:
        storage.objects.client.head_bucket(Bucket=settings.bucket)
    except ClientError as exc:
        if str(exc.response["Error"]["Code"]) not in {"404", "NoSuchBucket"}:
            raise
        storage.objects.client.create_bucket(Bucket=settings.bucket)
    print("Ingestion schema and private bucket initialized")


if __name__ == "__main__":
    main()
