import base64
import json
import os
from datetime import datetime

import boto3
import click
import requests
from botocore.exceptions import ClientError, NoCredentialsError

from reinforcenow.auth import is_authenticated, get_auth_headers, login_flow, validate_token, TOKEN_FILE
from reinforcenow.utils import stream_sse_response

API_URL = "http://localhost:8000"

# S3 Configuration
S3_BUCKET = os.getenv("REINMAX_S3_BUCKET", "reinmax-project-templates")
S3_REGION = os.getenv("REINMAX_S3_REGION", "us-east-1")
S3_PREFIX = os.getenv("REINMAX_S3_PREFIX", "templates/")

def get_s3_client():
    """Initialize S3 client with AWS credentials"""
    try:
        return boto3.client(
            's3',
            region_name=S3_REGION,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN")  # Optional, for temporary credentials
        )
    except NoCredentialsError:
        click.echo("AWS credentials not found. Please configure your AWS credentials.")
        click.echo("You can set them via environment variables:")
        click.echo("  export AWS_ACCESS_KEY_ID=your_access_key")
        click.echo("  export AWS_SECRET_ACCESS_KEY=your_secret_key")
        click.echo("Or use AWS CLI: aws configure")
        return None

def download_file_from_s3(s3_client, bucket, key, local_path):
    """Download a file from S3 to local path"""
    try:
        s3_client.download_file(bucket, key, local_path)
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            click.echo(f" File not found in S3: {key}")
        elif error_code == 'NoSuchBucket':
            click.echo(f" S3 bucket not found: {bucket}")
        else:
            click.echo(f" Error downloading {key}: {e}")
        return False
    except Exception as e:
        click.echo(f" Unexpected error downloading {key}: {e}")
        return False

@click.group()
def cli():
    pass

@cli.command()
def login():
    login_flow()

@cli.command()
def status():
    """Check authentication status and validate token"""
    if TOKEN_FILE.exists():
        click.echo("🔍 Checking authentication status...")
        
        if validate_token():
            click.echo(" Authenticated - Token is valid")
            
            # Show token info
            try:
                with open(TOKEN_FILE) as f:
                    data = json.load(f)
                
                # Show basic token info (without exposing sensitive data)
                if 'access_token' in data:
                    click.echo(f" Token file: {TOKEN_FILE}")
                    
                    # Decode token expiry if available (basic JWT inspection)
                    access_token = data['access_token']
                    if '.' in access_token:
                        try:
                            # JWT tokens have 3 parts separated by dots
                            parts = access_token.split('.')
                            if len(parts) >= 2:
                                # Decode the payload (second part)
                                payload = parts[1]
                                # Add padding if needed for base64 decoding
                                payload += '=' * (4 - len(payload) % 4)
                                decoded = base64.b64decode(payload)
                                token_data = json.loads(decoded)

                                if 'exp' in token_data:
                                    exp_timestamp = token_data['exp']
                                    exp_date = datetime.fromtimestamp(exp_timestamp)
                                    click.echo(f" Token expires: {exp_date}")
                        except Exception:
                            # If JWT decoding fails, just skip showing expiry
                            pass
                            
            except Exception as e:
                click.echo(f"  Warning: Could not read token details: {e}")
        else:
            click.echo(" Not authenticated - Token is invalid or expired")
            click.echo(" Run 'reinmax login' to authenticate")
    else:
        click.echo(" Not authenticated - No token found")
        click.echo(" Run 'reinmax login' to authenticate")

@cli.command()
def logout():
    """Clear authentication token"""
    if TOKEN_FILE.exists():
        TOKEN_FILE.unlink()
        click.echo(" Successfully logged out")
        click.echo("  Authentication token removed")
    else:
        click.echo("  Already logged out - no token found")

@cli.command()
def start():
    """Pull project template files from S3 bucket"""
    # Create project directory
    os.makedirs("./project", exist_ok=True)
    
    # Initialize S3 client
    s3_client = get_s3_client()
    if not s3_client:
        return
    
    # List of files to download
    files_to_download = [
        "tools.py", 
        "env.py", 
        "reward.py", 
        "project.toml", 
        "config.json", 
        "dataset.json"
    ]
    
    click.echo(f"📥 Downloading project template files from S3 bucket: {S3_BUCKET}")
    
    success_count = 0
    failed_files = []
    
    for filename in files_to_download:
        s3_key = f"{S3_PREFIX}{filename}"
        local_path = f"./project/{filename}"
        
        click.echo(f"  Downloading {filename}...", nl=False)
        
        if download_file_from_s3(s3_client, S3_BUCKET, s3_key, local_path):
            click.echo(" ")
            success_count += 1
        else:
            click.echo(" ")
            failed_files.append(filename)
    
    # Summary
    click.echo(f"\n Download Summary:")
    click.echo(f"   Successfully downloaded: {success_count}/{len(files_to_download)} files")
    
    if failed_files:
        click.echo(f"   Failed to download: {', '.join(failed_files)}")
        click.echo(f"\n Tips:")
        click.echo(f"  • Check that files exist in S3 bucket: {S3_BUCKET}")
        click.echo(f"  • Verify S3 prefix: {S3_PREFIX}")
        click.echo(f"  • Ensure AWS credentials have read access to the bucket")
    else:
        click.echo(f"   All files downloaded successfully!")
        click.echo(f"   Files are available in: ./project/")
    
    # List downloaded files
    if success_count > 0:
        click.echo(f"\n Downloaded files:")
        for filename in files_to_download:
            if filename not in failed_files:
                local_path = f"./project/{filename}"
                if os.path.exists(local_path):
                    file_size = os.path.getsize(local_path)
                    click.echo(f"   {filename} ({file_size} bytes)")

@cli.command()
@click.option('--project_name', required=False)
@click.option('--project_id', required=False)
def pull(project_name, project_id):
    if not is_authenticated():
        click.echo("Please login first.")
        return

    payload = {"project_name": project_name, "project_id": project_id}
    response = requests.post(f"{API_URL}/pull", json=payload, headers=get_auth_headers(), stream=True)
    stream_sse_response(response)

@cli.command()
@click.argument('params', nargs=-1)
def run(params):
    if not is_authenticated():
        click.echo("Please login first.")
        return

    payload = {
        "files": {},
        "params": dict(p.split("=") for p in params)
    }
    
    # Read files from project directory
    for fname in ["tools.py", "env.py", "reward.py", "project.toml", "config.json", "dataset.json"]:
        file_path = f"./project/{fname}"
        try:
            with open(file_path, 'r') as f:
                payload["files"][fname] = f.read()
        except FileNotFoundError:
            click.echo(f" File not found: {file_path}")
            click.echo(" Run 'reinforcenow start' first to download project template files.")
            return
        except Exception as e:
            click.echo(f" Error reading {file_path}: {e}")
            return

    response = requests.post(f"{API_URL}/run", json=payload, headers=get_auth_headers(), stream=True)
    stream_sse_response(response)

@cli.command()
@click.option('--run_id', required=True)
def stop(run_id):
    if not is_authenticated():
        click.echo("Please login first.")
        return
    response = requests.post(f"{API_URL}/stop", json={"run_id": run_id}, headers=get_auth_headers())
    click.echo(response.text)

if __name__ == "__main__":
    cli()