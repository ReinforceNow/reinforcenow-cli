import click

def stream_sse_response(response):
    """Stream Server-Sent Events from a requests response object"""
    try:
        # Check if response is successful
        response.raise_for_status()
        
        # Stream the response line by line
        for line in response.iter_lines(decode_unicode=True):
            if line:
                # Parse SSE format manually
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data.strip():
                        click.echo(data)
                elif line.startswith('event: '):
                    # Handle event type if needed
                    pass
                elif line == '':
                    # Empty line indicates end of event
                    pass
                else:
                    # Just print the line as-is for debugging
                    click.echo(line)
                    
    except Exception as e:
        click.echo(f" Error streaming response: {e}")
        # Fall back to showing the full response
        try:
            click.echo(f"Response status: {response.status_code}")
            click.echo(f"Response content: {response.text}")
        except Exception as fallback_error:
            click.echo(f" Could not read response: {fallback_error}")
