"""S3-compatible media storage backed by aiobotocore."""

from __future__ import annotations

import aiobotocore.session

from src.interfaces.media_storage import IMediaStorage


class S3MediaStorage(IMediaStorage):
    def __init__(
        self,
        *,
        endpoint_url: str | None,
        access_key: str,
        secret_key: str,
        bucket: str,
        region: str = "auto",
    ) -> None:
        self._endpoint_url = endpoint_url
        self._access_key = access_key
        self._secret_key = secret_key
        self._bucket = bucket
        self._region = region

    def _client(self):
        session = aiobotocore.session.get_session()
        return session.create_client(
            "s3",
            region_name=self._region,
            endpoint_url=self._endpoint_url,
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
        )

    async def upload(self, data: bytes, key: str, content_type: str) -> None:
        async with self._client() as client:
            await client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )

    async def signed_url(self, key: str, ttl_seconds: int) -> str:
        async with self._client() as client:
            return client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=ttl_seconds,
            )
