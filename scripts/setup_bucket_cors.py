"""Script to configure CORS on the GCS bucket for direct client PUT uploads."""

import os

from google.cloud import storage


def configure_cors(bucket_name: str | None = None):
    bucket_name = bucket_name or os.getenv("GCS_BUCKET", "chat_attachment")
    print(f"Configuring CORS on GCS bucket: {bucket_name}")

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    cors_configuration = [
        {
            "origin": ["*"],
            "method": ["GET", "PUT", "POST", "HEAD", "OPTIONS", "DELETE"],
            "responseHeader": ["*"],
            "maxAgeSeconds": 3600,
        }
    ]

    bucket.cors = cors_configuration
    bucket.patch()
    print(f"Successfully applied CORS to bucket '{bucket_name}':")
    print(bucket.cors)


if __name__ == "__main__":
    configure_cors()
