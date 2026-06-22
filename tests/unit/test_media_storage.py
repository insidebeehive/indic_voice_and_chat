"""Unit tests for S3MediaStorage."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_s3_client():
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.put_object = AsyncMock(return_value={})
    client.generate_presigned_url = MagicMock(return_value="https://bucket.example.com/signed")
    return client


@pytest.fixture
def mock_session(mock_s3_client):
    session = MagicMock()
    session.create_client.return_value = mock_s3_client
    return session


@pytest.mark.asyncio
async def test_upload_calls_put_object(mock_session, mock_s3_client):
    with patch("aiobotocore.session.get_session", return_value=mock_session):
        from src.providers.media.s3 import S3MediaStorage
        store = S3MediaStorage(
            endpoint_url="https://r2.example.com",
            access_key="key",
            secret_key="secret",
            bucket="chat-media",
            region="auto",
        )
        await store.upload(b"audio_bytes", "chat/t1/s1/abc.webm", "audio/webm")

    mock_s3_client.put_object.assert_called_once_with(
        Bucket="chat-media",
        Key="chat/t1/s1/abc.webm",
        Body=b"audio_bytes",
        ContentType="audio/webm",
    )


@pytest.mark.asyncio
async def test_signed_url_returns_url(mock_session, mock_s3_client):
    with patch("aiobotocore.session.get_session", return_value=mock_session):
        from src.providers.media.s3 import S3MediaStorage
        store = S3MediaStorage(
            endpoint_url="https://r2.example.com",
            access_key="key",
            secret_key="secret",
            bucket="chat-media",
            region="auto",
        )
        url = await store.signed_url("chat/t1/s1/abc.webm", ttl_seconds=3600)

    assert url == "https://bucket.example.com/signed"
    mock_s3_client.generate_presigned_url.assert_called_once_with(
        "get_object",
        Params={"Bucket": "chat-media", "Key": "chat/t1/s1/abc.webm"},
        ExpiresIn=3600,
    )
