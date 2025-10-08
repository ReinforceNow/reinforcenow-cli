import base64
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import click
import requests

from reinforcenow.auth import is_authenticated, get_auth_headers, login_flow, validate_token, TOKEN_FILE
from reinforcenow.utils import stream_sse_response

API_URL = "http://localhost:8000"

def get_template_dir():
    """Get the path to the bundled templates directory"""
    return Path(__file__).parent / "templates"

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
    """Initialize a new project with template files"""
    project_dir = Path("./project")
    template_dir = get_template_dir()

    # Create project directory
    project_dir.mkdir(exist_ok=True)

    # List of template files to copy
    template_files = [
        "tools.py",
        "env.py",
        "reward.py",
        "project.toml",
        "config.json",
        "dataset.json"
    ]

    click.echo("📦 Initializing new project with template files...")

    success_count = 0
    failed_files = []

    for filename in template_files:
        source = template_dir / filename
        destination = project_dir / filename

        try:
            shutil.copy2(source, destination)
            click.echo(f"  ✓ Created {filename}")
            success_count += 1
        except FileNotFoundError:
            click.echo(f"  ✗ Template not found: {filename}")
            failed_files.append(filename)
        except Exception as e:
            click.echo(f"  ✗ Error copying {filename}: {e}")
            failed_files.append(filename)

    # Summary
    click.echo(f"\n✨ Project initialized!")
    click.echo(f"   Successfully created: {success_count}/{len(template_files)} files")

    if failed_files:
        click.echo(f"   Failed to create: {', '.join(failed_files)}")
    else:
        click.echo(f"   All files created successfully!")
        click.echo(f"   Files are available in: ./project/")
        click.echo(f"\n📝 Next steps:")
        click.echo(f"   1. Edit the files in ./project/ to customize your RL environment")
        click.echo(f"   2. Run 'reinforcenow run' to start training")

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
            click.echo(" Run 'reinforcenow start' first to initialize project template files.")
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