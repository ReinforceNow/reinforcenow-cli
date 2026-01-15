# reinforcenow/cli/upload.py
"""Direct S3 upload support via presigned URLs."""

import asyncio
import io
import math
import tarfile
from pathlib import Path

import httpx

from rnow.cli import auth

# Size threshold for multipart uploads (100MB)
MULTIPART_THRESHOLD = 100 * 1024 * 1024
# Part size for multipart uploads (16MB)
PART_SIZE = 16 * 1024 * 1024


def get_content_type(filename: str) -> str:
    """Get content type based on file extension."""
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    content_types = {
        "py": "text/x-python",
        "yml": "text/yaml",
        "yaml": "text/yaml",
        "json": "application/json",
        "jsonl": "application/jsonl",
        "txt": "text/plain",
        "tar": "application/x-tar",
        "gz": "application/gzip",
    }
    return content_types.get(ext, "application/octet-stream")


def upload_file_direct(
    base_url: str,
    file_path: Path,
    entity_id: str,
    version_id: str,
    filename: str | None = None,
    on_progress: callable = None,
    upload_type: str = "project",
) -> str:
    """
    Upload a file directly to S3 via presigned URL.

    For files > MULTIPART_THRESHOLD, uses multipart upload.
    upload_type: "project" or "dataset" to determine S3 path prefix.
    Returns the S3 key of the uploaded file.
    """
    filename = filename or file_path.name
    file_size = file_path.stat().st_size
    content_type = get_content_type(filename)

    if file_size > MULTIPART_THRESHOLD:
        return _multipart_upload(
            base_url,
            file_path,
            entity_id,
            version_id,
            filename,
            content_type,
            on_progress,
            upload_type,
        )
    else:
        return _single_put_upload(
            base_url,
            file_path,
            entity_id,
            version_id,
            filename,
            content_type,
            on_progress,
            upload_type,
        )


def _single_put_upload(
    base_url: str,
    file_path: Path,
    entity_id: str,
    version_id: str,
    filename: str,
    content_type: str,
    on_progress: callable = None,
    upload_type: str = "project",
) -> str:
    """Upload file using single presigned PUT."""
    headers = auth.get_auth_headers()

    # Get presigned URL
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{base_url}/uploads/presign-put",
            headers=headers,
            json={
                "projectId": entity_id,
                "versionId": version_id,
                "filename": filename,
                "contentType": content_type,
                "type": upload_type,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    presigned_url = data["url"]
    key = data["key"]

    # Upload directly to S3
    with open(file_path, "rb") as f:
        file_content = f.read()

    with httpx.Client(timeout=300) as client:
        resp = client.put(
            presigned_url,
            content=file_content,
            headers={"Content-Type": content_type},
        )
        resp.raise_for_status()

    if on_progress:
        on_progress(len(file_content), len(file_content))

    return key


def _multipart_upload(
    base_url: str,
    file_path: Path,
    entity_id: str,
    version_id: str,
    filename: str,
    content_type: str,
    on_progress: callable = None,
    upload_type: str = "project",
) -> str:
    """Upload large file using multipart upload."""
    headers = auth.get_auth_headers()
    file_size = file_path.stat().st_size
    part_count = math.ceil(file_size / PART_SIZE)

    with httpx.Client(timeout=30) as client:
        # Initialize multipart upload
        resp = client.post(
            f"{base_url}/uploads/multipart/init",
            headers=headers,
            json={
                "projectId": entity_id,
                "versionId": version_id,
                "filename": filename,
                "contentType": content_type,
                "type": upload_type,
            },
        )
        resp.raise_for_status()
        init_data = resp.json()

    upload_id = init_data["uploadId"]
    key = init_data["key"]

    parts = []
    uploaded_bytes = 0

    try:
        with open(file_path, "rb") as f:
            for part_number in range(1, part_count + 1):
                chunk = f.read(PART_SIZE)

                # Get presigned URL for this part
                with httpx.Client(timeout=30) as client:
                    resp = client.post(
                        f"{base_url}/uploads/multipart/sign",
                        headers=headers,
                        json={
                            "key": key,
                            "uploadId": upload_id,
                            "partNumber": part_number,
                        },
                    )
                    resp.raise_for_status()
                    sign_data = resp.json()

                part_url = sign_data["url"]

                # Upload part
                with httpx.Client(timeout=300) as client:
                    resp = client.put(part_url, content=chunk)
                    resp.raise_for_status()

                etag = resp.headers.get("ETag")
                if not etag:
                    raise RuntimeError("Missing ETag from S3 UploadPart response")

                parts.append({"PartNumber": part_number, "ETag": etag.strip('"')})

                uploaded_bytes += len(chunk)
                if on_progress:
                    on_progress(uploaded_bytes, file_size)

        # Complete multipart upload
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{base_url}/uploads/multipart/complete",
                headers=headers,
                json={
                    "key": key,
                    "uploadId": upload_id,
                    "parts": parts,
                },
            )
            resp.raise_for_status()

        return key

    except Exception as e:
        # Abort multipart upload on failure (cleanup)
        try:
            with httpx.Client(timeout=30) as client:
                client.post(
                    f"{base_url}/uploads/multipart/abort",
                    headers=headers,
                    json={"key": key, "uploadId": upload_id},
                )
        except Exception:
            pass  # Ignore abort errors
        raise e


def bundle_images_to_tar(images_dir: Path) -> tuple[io.BytesIO, int]:
    """
    Bundle an images/ directory into a tar archive (in memory).
    Returns (tar_buffer, file_count).
    """
    tar_buffer = io.BytesIO()
    file_count = 0

    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        for img_path in images_dir.rglob("*"):
            if img_path.is_file():
                # Use relative path within images/
                arcname = img_path.relative_to(images_dir.parent)
                tar.add(img_path, arcname=str(arcname))
                file_count += 1

    tar_buffer.seek(0)
    return tar_buffer, file_count


def upload_images_tar(
    base_url: str,
    images_dir: Path,
    project_id: str,
    version_id: str,
    on_progress: callable = None,
) -> tuple[str, int]:
    """
    Bundle images/ directory and upload as images.tar.
    Returns (s3_key, file_count).
    """
    import tempfile

    # Create tar in a temp file (for large image sets)
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        file_count = 0
        with tarfile.open(tmp_path, mode="w") as tar:
            for img_path in images_dir.rglob("*"):
                if img_path.is_file():
                    arcname = img_path.relative_to(images_dir.parent)
                    tar.add(img_path, arcname=str(arcname))
                    file_count += 1

        if file_count == 0:
            return None, 0

        # Upload the tar file
        key = upload_file_direct(
            base_url,
            tmp_path,
            project_id,
            version_id,
            filename="images.tar",
            on_progress=on_progress,
        )

        return key, file_count
    finally:
        tmp_path.unlink(missing_ok=True)


# =============================================================================
# Parallel Upload Support (async with bounded concurrency)
# =============================================================================

# Default concurrency limit for parallel uploads (balances speed vs S3 throttling)
DEFAULT_UPLOAD_CONCURRENCY = 16


async def _async_upload_single_file(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    base_url: str,
    headers: dict,
    filename: str,
    content: bytes,
    entity_id: str,
    version_id: str,
    upload_type: str,
    content_type: str,
) -> str:
    """Upload a single file with semaphore-bounded concurrency."""
    async with semaphore:
        # Get presigned URL
        resp = await client.post(
            f"{base_url}/uploads/presign-put",
            headers=headers,
            json={
                "projectId": entity_id,
                "versionId": version_id,
                "filename": filename,
                "contentType": content_type,
                "type": upload_type,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        # Upload to S3
        resp = await client.put(
            data["url"],
            content=content,
            headers={"Content-Type": content_type},
        )
        resp.raise_for_status()

        return data["key"]


async def upload_files_parallel(
    base_url: str,
    files: list[
        tuple[str, Path, str, str, str]
    ],  # (filename, path, entity_id, version_id, upload_type)
    concurrency: int = DEFAULT_UPLOAD_CONCURRENCY,
) -> list[str]:
    """
    Upload multiple files in parallel with bounded concurrency.

    Uses connection pooling and semaphore to limit concurrent uploads,
    avoiding S3 throttling while maximizing throughput.

    Args:
        base_url: API base URL
        files: List of (filename, file_path, entity_id, version_id, upload_type) tuples
        concurrency: Max concurrent uploads (default: 16)

    Returns:
        List of S3 keys for uploaded files
    """
    if not files:
        return []

    headers = auth.get_auth_headers()
    semaphore = asyncio.Semaphore(concurrency)

    # Read all file contents upfront
    file_data = []
    for filename, file_path, entity_id, version_id, upload_type in files:
        content_type = get_content_type(filename)
        with open(file_path, "rb") as f:
            content = f.read()
        file_data.append((filename, content, entity_id, version_id, upload_type, content_type))

    # Use a single client for all requests (connection pooling)
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=300.0)) as client:
        tasks = [
            _async_upload_single_file(
                client,
                semaphore,
                base_url,
                headers,
                filename,
                content,
                entity_id,
                version_id,
                upload_type,
                content_type,
            )
            for filename, content, entity_id, version_id, upload_type, content_type in file_data
        ]
        keys = await asyncio.gather(*tasks)

    return list(keys)


def upload_files_parallel_sync(
    base_url: str,
    files: list[tuple[str, Path, str, str, str]],
    concurrency: int = DEFAULT_UPLOAD_CONCURRENCY,
) -> list[str]:
    """
    Synchronous wrapper for parallel uploads.

    Args:
        base_url: API base URL
        files: List of (filename, file_path, entity_id, version_id, upload_type) tuples
        concurrency: Max concurrent uploads (default: 16)

    Returns:
        List of S3 keys for uploaded files
    """
    return asyncio.run(upload_files_parallel(base_url, files, concurrency))


async def upload_images_parallel(
    base_url: str,
    images_dir: Path,
    project_id: str,
    version_id: str,
    concurrency: int = DEFAULT_UPLOAD_CONCURRENCY,
) -> tuple[list[str], int]:
    """
    Upload all images from a directory in parallel.

    Images are uploaded to: projects/{id}/versions/{v}/images/{filename}

    Args:
        base_url: API base URL
        images_dir: Directory containing images
        project_id: Project ID
        version_id: Version ID
        concurrency: Max concurrent uploads

    Returns:
        (list of S3 keys, image count)
    """
    # Collect all image files
    image_files = []
    for img_path in images_dir.rglob("*"):
        if img_path.is_file():
            # Preserve relative path structure: images/subdir/file.png
            rel_path = img_path.relative_to(images_dir.parent)
            image_files.append((str(rel_path), img_path, project_id, version_id, "project"))

    if not image_files:
        return [], 0

    keys = await upload_files_parallel(base_url, image_files, concurrency)
    return keys, len(keys)


def upload_images_parallel_sync(
    base_url: str,
    images_dir: Path,
    project_id: str,
    version_id: str,
    concurrency: int = DEFAULT_UPLOAD_CONCURRENCY,
) -> tuple[list[str], int]:
    """Synchronous wrapper for parallel image uploads."""
    return asyncio.run(
        upload_images_parallel(base_url, images_dir, project_id, version_id, concurrency)
    )


# =============================================================================
# STS Session Upload (boto3 direct to S3)
# =============================================================================


def get_upload_session(
    base_url: str,
    project_id: str,
    project_version_id: str,
    dataset_id: str,
    dataset_version_id: str,
    duration: int = 900,
) -> dict | None:
    """
    Get temporary AWS credentials for direct S3 upload.

    Returns None if STS sessions are not configured (fallback to presigned URLs).
    """
    headers = auth.get_auth_headers()

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{base_url}/uploads/session",
                headers=headers,
                json={
                    "projectId": project_id,
                    "projectVersionId": project_version_id,
                    "datasetId": dataset_id,
                    "datasetVersionId": dataset_version_id,
                    "duration": duration,
                },
            )

            # 501 means STS not configured - fallback to presigned URLs
            if resp.status_code == 501:
                return None

            resp.raise_for_status()
            return resp.json()

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 501:
            return None
        raise


def upload_with_boto3(
    session: dict,
    files: list[tuple[str, Path, str]],  # (filename, path, prefix_type: "project" | "dataset")
    file_concurrency: int = 16,
    multipart_threshold: int = 64 * 1024 * 1024,  # 64MB
    multipart_chunksize: int = 64 * 1024 * 1024,  # 64MB
    max_concurrency: int = 32,
) -> list[str]:
    """
    Upload files directly to S3 using boto3 with temporary credentials.

    Uses boto3's TransferManager for automatic multipart uploads, retries,
    and optimal concurrency.

    Args:
        session: Upload session from get_upload_session()
        files: List of (filename, file_path, prefix_type) tuples
        file_concurrency: Max files to upload in parallel
        multipart_threshold: Size threshold for multipart uploads
        multipart_chunksize: Size of each multipart chunk
        max_concurrency: Max concurrent parts per multipart upload

    Returns:
        List of S3 keys for uploaded files
    """
    try:
        import boto3
        from boto3.s3.transfer import S3Transfer, TransferConfig
    except ImportError:
        raise ImportError("boto3 is required for STS uploads. Install with: pip install boto3")

    import concurrent.futures

    # Create S3 client with temporary credentials
    s3_client = boto3.client(
        "s3",
        region_name=session["region"],
        aws_access_key_id=session["credentials"]["accessKeyId"],
        aws_secret_access_key=session["credentials"]["secretAccessKey"],
        aws_session_token=session["credentials"]["sessionToken"],
    )

    # Configure transfer settings
    config = TransferConfig(
        multipart_threshold=multipart_threshold,
        multipart_chunksize=multipart_chunksize,
        max_concurrency=max_concurrency,
        use_threads=True,
    )
    transfer = S3Transfer(s3_client, config)

    bucket = session["bucket"]
    project_prefix = session["projectPrefix"]
    dataset_prefix = session["datasetPrefix"]

    def upload_one(item: tuple[str, Path, str]) -> str:
        filename, file_path, prefix_type = item
        if prefix_type == "dataset":
            key = f"{dataset_prefix}/{filename}"
        else:
            key = f"{project_prefix}/{filename}"

        transfer.upload_file(
            str(file_path),
            bucket,
            key,
            extra_args={"ContentType": get_content_type(filename)},
        )
        return key

    keys = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=file_concurrency) as executor:
        keys = list(executor.map(upload_one, files))

    return keys


def upload_directory_with_boto3(
    session: dict,
    directory: Path,
    prefix_type: str,  # "project" | "dataset"
    subdir: str = "",  # e.g., "images" to upload as prefix/images/*
    file_concurrency: int = 16,
) -> tuple[list[str], int]:
    """
    Upload an entire directory to S3 using boto3.

    Args:
        session: Upload session from get_upload_session()
        directory: Local directory to upload
        prefix_type: "project" or "dataset"
        subdir: Subdirectory name in S3 (e.g., "images")
        file_concurrency: Max files to upload in parallel

    Returns:
        (list of S3 keys, file count)
    """
    files = []
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(directory)
            filename = f"{subdir}/{rel_path.as_posix()}" if subdir else rel_path.as_posix()
            files.append((filename, file_path, prefix_type))

    if not files:
        return [], 0

    keys = upload_with_boto3(session, files, file_concurrency=file_concurrency)
    return keys, len(keys)
