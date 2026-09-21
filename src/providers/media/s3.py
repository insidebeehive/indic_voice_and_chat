"""S3-compatible media storage backed by aiobotocore."""

from __future__ import annotations

import logging

import aiobotocore.session
from botocore.exceptions import ClientError

from src.interfaces.media_storage import IMediaStorage
from src.utils.logging import debug_event

log = logging.getLogger(__name__)


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
        # The blob itself is never logged (binary, and can be large media) --
        # same rule as audio elsewhere in this package. Size/key/content_type
        # is the diagnostic value.
        debug_event(log, "s3 upload", bucket=self._bucket, key=key,
                    content_type=content_type, bytes=len(data))
        async with self._client() as client:
            await client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )

    async def signed_url(self, key: str, ttl_seconds: int) -> str:
        async with self._client() as client:
            url = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=ttl_seconds,
            )
            # The URL itself carries an AWS SigV4 signature (X-Amz-Signature/
            # X-Amz-Credential query params) that is a live, scoped credential
            # for `ttl_seconds` -- logging it would be exactly the thing the
            # credential rule in this pass exists to prevent. Log the request
            # that produced it, never the result.
            debug_event(log, "s3 signed_url", bucket=self._bucket, key=key,
                        ttl_seconds=ttl_seconds)
            return url

    async def download(self, key: str) -> tuple[bytes, str]:
        async with self._client() as client:
            try:
                response = await client.get_object(Bucket=self._bucket, Key=key)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code in ("NoSuchKey", "404"):
                    debug_event(log, "s3 download not found", bucket=self._bucket, key=key)
                    raise FileNotFoundError(key) from exc
                debug_event(log, "s3 download failed", bucket=self._bucket, key=key,
                            error=str(exc))
                raise
            body = await response["Body"].read()
            content_type = response.get("ContentType", "application/octet-stream")
            debug_event(log, "s3 download response", bucket=self._bucket, key=key,
                        content_type=content_type, bytes=len(body))
            return body, content_type
