# reinforcenow/cli/upload.py
"""Direct S3 upload support via STS temporary credentials."""

from pathlib import Path

import httpx

from rnow.cli import auth


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
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    return content_types.get(ext, "application/octet-stream")


def get_upload_session(
    base_url: str,
    project_id: str,
    project_version_id: str,
    dataset_id: str,
    dataset_version_id: str,
    duration: int = 900,
) -> dict:
    """
    Get temporary AWS credentials for direct S3 upload.

    Args:
        base_url: API base URL
        project_id: Project ID
        project_version_id: Project version ID
        dataset_id: Dataset ID
        dataset_version_id: Dataset version ID
        duration: Session duration in seconds (default 15 min)

    Returns:
        Upload session with credentials, bucket, region, and prefixes
    """
    headers = auth.get_auth_headers()

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
        resp.raise_for_status()
        return resp.json()


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
        raise ImportError("boto3 is required for uploads. Install with: pip install boto3")

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
