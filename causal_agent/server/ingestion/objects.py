"""Private object storage. SDK retries are bounded; no public object URLs are returned."""

import hashlib
import mimetypes

import boto3
from botocore.config import Config


class Objects:
    def __init__(self, settings):
        self.bucket = settings.bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.endpoint,
            aws_access_key_id=settings.access_key,
            aws_secret_access_key=settings.secret_key,
            region_name="us-east-1",
            config=Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 2}, s3={"addressing_style": "path"}),
        )

    def put(self, key, stream, size, digest):
        stream.seek(0)
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=stream,
            ContentLength=size,
            ContentType=mimetypes.guess_type(key)[0] or "application/octet-stream",
            Metadata={"sha256": digest},
        )
        # Verify actual returned bytes, not merely the metadata we supplied.
        body = self.open(key)
        actual, length = hashlib.sha256(), 0
        try:
            for chunk in body.iter_chunks(64 * 1024):
                actual.update(chunk)
                length += len(chunk)
        finally:
            body.close()
        if length != size or actual.hexdigest() != digest:
            raise OSError("Stored CSV verification failed")

    def open(self, key):
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"]

    def delete(self, key):
        self.client.delete_object(Bucket=self.bucket, Key=key)
