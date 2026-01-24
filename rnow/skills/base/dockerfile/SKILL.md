---
name: dockerfile
description: Write Dockerfiles for sandbox environments. Triggers on "Dockerfile", "docker image", "FROM", "ENTRYPOINT", "docker build", "sandbox image".
allowed-tools: Read, Edit, Write, Bash, Grep, Glob
---

# Writing Dockerfiles for Sandbox Tools

## Local Dockerfile Convention

`"docker": "local/foo"` → looks for `Dockerfile.foo` in project root. No registry push needed.

## Critical: Modal Sandbox Behavior

**Modal sandboxes run `sleep infinity` and IGNORE Docker ENTRYPOINT/CMD.**

This means if your Dockerfile has:
```dockerfile
ENTRYPOINT ["node", "server.js", "--port", "8931"]
```

Modal will NOT run this. It runs `sleep infinity` instead, keeping the container alive but never starting your server.

### The Fix: Wrapper Script Pattern

Create an entrypoint script that starts your service in the background when ANY command runs:

```dockerfile
FROM some-image:latest

EXPOSE 8931

USER root

# This script runs BEFORE whatever command Modal passes (sleep infinity)
# 1. Starts your service in background
# 2. Waits for it to be ready
# 3. Runs the original command (sleep infinity)
RUN echo '#!/bin/bash' > /start.sh && \
    echo 'your-server-command --port 8931 &' >> /start.sh && \
    echo 'sleep 3' >> /start.sh && \
    echo 'exec "$@"' >> /start.sh && \
    chmod +x /start.sh

ENTRYPOINT ["/start.sh"]
```

When Modal runs `sleep infinity`:
1. `/start.sh` starts your server in background
2. Waits 3 seconds for server to be ready
3. Runs `sleep infinity` to keep container alive
4. Your server is now accessible on port 8931

## Key Rules

1. **No heredoc syntax** - Use `RUN echo` instead of `COPY <<'EOF'`
2. **Non-root images** - Use `USER root` to create files, then `USER <original>` after
3. **Platform** - Build with `--platform linux/amd64` for cloud

## Example: MCP Server (Playwright)

```dockerfile
FROM mcp/playwright

EXPOSE 8931

USER root

# Base image entrypoint: node cli.js --headless --browser chromium --no-sandbox
# --port 8931 --host 0.0.0.0: expose as HTTP/SSE server
# --allowed-hosts '*': allow any hostname (required for cloud tunnels, otherwise 403 Forbidden)
RUN echo '#!/bin/bash' > /start.sh && \
    echo 'node cli.js --headless --browser chromium --no-sandbox --port 8931 --host 0.0.0.0 --allowed-hosts "*" &' >> /start.sh && \
    echo 'sleep 5' >> /start.sh && \
    echo 'exec "$@"' >> /start.sh && \
    chmod +x /start.sh

ENTRYPOINT ["/start.sh"]
```

## Example: Background Service with Health Check

```dockerfile
FROM some-image:latest

USER root

RUN echo '#!/bin/bash' > /start.sh && \
    echo 'some-service &' >> /start.sh && \
    echo 'for i in {1..30}; do curl -s http://localhost:PORT/health && break; sleep 1; done' >> /start.sh && \
    echo 'exec "$@"' >> /start.sh && \
    chmod +x /start.sh

USER original-user  # switch back if needed

ENTRYPOINT ["/start.sh"]
```

## Debugging Methodology

```bash
# 1. Inspect the base image
docker pull IMAGE:TAG
docker inspect IMAGE:TAG --format='{{json .Config.Cmd}}'
docker run --rm IMAGE:TAG id  # check user
docker run --rm IMAGE:TAG cat supervisord.conf 2>/dev/null  # find startup commands

# 2. Build and test container starts
docker build --platform linux/amd64 -t test:latest -f Dockerfile.foo .
docker run --rm test:latest echo "SUCCESS"

# 3. Test with Modal's behavior (simulates how Modal runs it)
docker run --rm test:latest sleep 5  # Should start your service + sleep

# 4. TEST YOUR ACTUAL TOOL (critical!)
docker run --rm test:latest python -c "from tools import browse; print(browse('https://example.com'))"
```

**The most common mistake:** Testing that the container starts but not testing the actual tool function with real inputs.

## Common Issues

| Issue | Fix |
|-------|-----|
| `Server not starting` | Modal ignores ENTRYPOINT - use wrapper script pattern above |
| `Permission denied` | Add `USER root` before creating files |
| `Heredoc not supported` | Use `RUN echo '...' > file` |
| `Platform mismatch` | Build with `--platform linux/amd64` |
| `Server not responding` | Check `supervisord.conf` or similar for actual startup commands |
