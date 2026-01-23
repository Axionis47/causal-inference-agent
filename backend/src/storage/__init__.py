"""Storage module."""

from .firestore import FirestoreClient, get_firestore_client

__all__ = ["FirestoreClient", "get_firestore_client"]
