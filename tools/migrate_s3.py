"""One-off copy of every object from the legacy MinIO store into SeaweedFS.

Run through the compose "migrate" profile (see docker-compose.unified.yml):

    docker compose -f docker-compose.unified.yml --profile migrate run --rm storage-migrate

Idempotent: objects already present in the destination with the same size are
skipped, so it is safe to re-run after an interruption. User metadata and
content type are carried over. The source is only read, never modified.
"""

import os
import sys

import boto3
from botocore.exceptions import ClientError

SRC = os.environ.get("SRC_ENDPOINT_URL", "http://minio-legacy:9000")
DST = os.environ.get("DST_ENDPOINT_URL", "http://seaweedfs:8333")
ACCESS = os.environ["S3_ACCESS_KEY"]
SECRET = os.environ["S3_SECRET_KEY"]
BUCKETS = [b.strip() for b in os.environ.get("MIGRATE_BUCKETS", "documents,openwebui").split(",") if b.strip()]


def client(endpoint):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=ACCESS,
        aws_secret_access_key=SECRET,
        region_name="us-east-1",
    )


def dest_size(dst, bucket, key):
    try:
        return dst.head_object(Bucket=bucket, Key=key)["ContentLength"]
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def migrate_bucket(src, dst, bucket):
    try:
        src.head_bucket(Bucket=bucket)
    except ClientError:
        print(f"[migrate] source bucket {bucket!r} not found, skipping")
        return 0, 0, 0
    try:
        dst.head_bucket(Bucket=bucket)
    except ClientError:
        dst.create_bucket(Bucket=bucket)

    copied = skipped = failed = 0
    for page in src.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            key, size = obj["Key"], obj["Size"]
            if dest_size(dst, bucket, key) == size:
                skipped += 1
                continue
            try:
                got = src.get_object(Bucket=bucket, Key=key)
                extra = {"Metadata": got.get("Metadata", {})}
                if got.get("ContentType"):
                    extra["ContentType"] = got["ContentType"]
                dst.upload_fileobj(got["Body"], bucket, key, ExtraArgs=extra)
                copied += 1
            except Exception as exc:  # keep going; report at the end
                failed += 1
                print(f"[migrate] FAILED {bucket}/{key}: {exc}", file=sys.stderr)
    print(f"[migrate] {bucket}: copied={copied} skipped={skipped} failed={failed}")
    return copied, skipped, failed


def main():
    src, dst = client(SRC), client(DST)
    totals = [0, 0, 0]
    for bucket in BUCKETS:
        for i, n in enumerate(migrate_bucket(src, dst, bucket)):
            totals[i] += n
    print(f"[migrate] done: copied={totals[0]} skipped={totals[1]} failed={totals[2]}")
    sys.exit(1 if totals[2] else 0)


if __name__ == "__main__":
    main()
