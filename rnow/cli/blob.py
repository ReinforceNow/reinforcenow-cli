# reinforcenow/cli/blob.py
"""Vercel Blob upload support for large files."""

from pathlib import Path

import requests

from rnow.cli import auth

# Size threshold for blob uploads (4MB to stay under 4.5MB limit)
MAX_INLINE_BYTES = 4 * 1024 * 1024

BLOB_API_URL = "https://blob.vercel-storage.com"
BLOB_API_VERSION = "7"


def request_blob_client_token(base_url: str, pathname: str) -> str:
    """
    Request a client upload token from the backend.
    This token allows direct upload to Vercel Blob.
    """
    headers = auth.get_auth_headers()
    headers["Content-Type"] = "application/json"

    payload = {
        "type": "blob.generate-client-token",
        "payload": {
            "pathname": pathname,
            "callbackUrl": f"{base_url}/dataset/upload",
        },
    }

    resp = requests.post(
        f"{base_url}/dataset/upload",
        headers=headers,
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("type") != "blob.generate-client-token":
        raise RuntimeError(f"Unexpected response from blob token endpoint: {data}")

    client_token = data.get("clientToken")
    if not client_token:
        raise RuntimeError("No clientToken returned from blob token endpoint")

    return client_token


def upload_file_to_blob(base_url: str, local_path: Path, blob_pathname: str) -> dict:
    """
    Upload a file directly to Vercel Blob using a client token.
    Returns the blob JSON (contains url, pathname, etc).
    """
    client_token = request_blob_client_token(base_url, blob_pathname)

    url = f"{BLOB_API_URL}/{blob_pathname.lstrip('/')}"
    headers = {
        "Authorization": f"Bearer {client_token}",
        "x-api-version": BLOB_API_VERSION,
        "x-content-type": "application/jsonl",
    }

    with open(local_path, "rb") as f:
        resp = requests.put(url, headers=headers, data=f, timeout=300)

    resp.raise_for_status()
    return resp.json()


def maybe_upload_to_blob(
    base_url: str,
    file_path: Path,
    dataset_id: str,
) -> tuple[str | None, dict | None]:
    """
    Check if file needs blob upload and handle it.

    Returns:
        (inline_contents, blob_info)
        - If small: inline_contents is file content, blob_info is None
        - If large: inline_contents is None, blob_info has url/pathname
    """
    size = file_path.stat().st_size

    if size <= MAX_INLINE_BYTES:
        # Small file - return contents for inline upload
        return None, None

    # Large file - upload to blob
    import uuid

    blob_pathname = f"datasets/{dataset_id}/{uuid.uuid4().hex[:8]}-{file_path.name}"

    blob = upload_file_to_blob(base_url, file_path, blob_pathname)
    return None, blob
