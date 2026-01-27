---
name: dockerfile
description: Write Dockerfiles for sandbox environments. Triggers on "Dockerfile", "docker image", "FROM", "ENTRYPOINT", "docker build", "sandbox image".
allowed-tools: Read, Edit, Write, Bash, Grep, Glob
---

# Writing Dockerfiles for Sandbox Tools

## Local Dockerfile Convention

`"docker": "local/foo"` → looks for `Dockerfile.foo` in project root. No registry push needed.

## Critical: Modal Sandbox Behavior

**ReinforceNow runs `/start.sh sleep infinity` for all sandbox images.**

Modal ignores Docker ENTRYPOINT/CMD, so we explicitly run `/start.sh sleep infinity`. Your Dockerfile must define `/start.sh` to initialize the container.

At termination, ReinforceNow automatically runs `/terminate.sh` (if it exists) before killing the sandbox.

### Required Scripts

| Script | When Called | Purpose |
|--------|-------------|---------|
| `/start.sh` | Container startup | Initialize services, acquire resources |
| `/terminate.sh` | Before termination | Release resources, cleanup |

### The /start.sh Pattern

Create `/start.sh` that initializes your service then runs the passed command:

```dockerfile
FROM some-image:latest

EXPOSE 8931

USER root

# /start.sh: Called at container startup with "sleep infinity" as args
# 1. Initialize your service (acquire resources, start servers)
# 2. exec "$@" runs "sleep infinity" to keep container alive
RUN echo '#!/bin/bash' > /start.sh && \
    echo 'set -e' >> /start.sh && \
    echo 'your-server-command --port 8931 &' >> /start.sh && \
    echo 'sleep 3' >> /start.sh && \
    echo 'exec "$@"' >> /start.sh && \
    chmod +x /start.sh

# /terminate.sh: Called automatically before sandbox termination
RUN echo '#!/bin/bash' > /terminate.sh && \
    echo 'echo "Cleaning up..."' >> /terminate.sh && \
    echo '# YOUR CLEANUP CODE HERE' >> /terminate.sh && \
    chmod +x /terminate.sh

ENTRYPOINT ["/start.sh"]
CMD ["sleep", "infinity"]
```

When ReinforceNow runs `/start.sh sleep infinity`:
1. `/start.sh` initializes your service
2. `exec "$@"` runs `sleep infinity` to keep container alive
3. Tools execute against your service
4. On termination: `/terminate.sh` runs automatically, then container killed

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

## MCP Server Host Restrictions

**CRITICAL:** Many MCP servers (especially Playwright MCP) restrict connections to localhost by default, even when bound to `0.0.0.0`. When Modal's tunnel connects, the server sees a non-localhost hostname and returns **403 Forbidden**.

**Fix:** Add `--allowed-hosts '*'` (or `--allowed-hosts "*"` in shell scripts) to allow external connections.

### Example: Kernel Browser with Replay

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y curl jq && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir requests rnow

# Creates Kernel browser session and starts replay recording
RUN echo '#!/bin/bash' > /start.sh && \
    echo 'set -e' >> /start.sh && \
    echo 'RESP=$(curl -sS --max-time 60 -X POST "https://api.onkernel.com/browsers" -H "Authorization: Bearer $KERNEL_API_KEY" -H "Content-Type: application/json" -d "{\"timeout_seconds\": 300, \"headless\": false, \"viewport\": {\"width\": 1024, \"height\": 768}}")' >> /start.sh && \
    echo 'SESSION_ID=$(echo "$RESP" | jq -r ".session_id")' >> /start.sh && \
    echo 'echo "$SESSION_ID" > /tmp/kernel_session_id' >> /start.sh && \
    echo 'curl -sS -X POST "https://api.onkernel.com/browsers/$SESSION_ID/replays" -H "Authorization: Bearer $KERNEL_API_KEY" -H "Content-Type: application/json" -d "{}" || true' >> /start.sh && \
    echo 'exec "$@"' >> /start.sh && \
    chmod +x /start.sh

# Cleanup: delete browser session
RUN echo '#!/bin/bash' > /terminate.sh && \
    echo 'SESSION_ID=$(cat /tmp/kernel_session_id 2>/dev/null)' >> /terminate.sh && \
    echo '[ -n "$SESSION_ID" ] && curl -sS --max-time 10 -X DELETE "https://api.onkernel.com/browsers/$SESSION_ID" -H "Authorization: Bearer $KERNEL_API_KEY" || true' >> /terminate.sh && \
    chmod +x /terminate.sh

ENTRYPOINT ["/start.sh"]
CMD ["sleep", "infinity"]
```

## Cleanup with /terminate.sh

When a rollout completes, ReinforceNow automatically runs `/terminate.sh` before terminating the sandbox. Use this to release external resources (sessions, connections, etc.).

**The CLI shows cleanup status:**
```
Rollout 1: ✓ reward=2.000  [cleanup ✓]   # terminate.sh ran successfully
Rollout 1: ✓ reward=2.000  [cleanup ✗]   # terminate.sh not found or failed
```

### How It Works

```
Rollout completes:
1. ReinforceNow calls: sandbox.exec("/terminate.sh")  ← Your cleanup runs
2. ReinforceNow calls: sandbox.terminate()            ← Container killed
```

**No SIGTERM traps needed** - we explicitly call your script before killing the container.

### Creating /terminate.sh

Add your cleanup logic to `/terminate.sh`:

```dockerfile
# Option 1: Inline in Dockerfile
RUN echo '#!/bin/bash' > /terminate.sh && \
    echo '# YOUR CLEANUP CODE HERE' >> /terminate.sh && \
    echo 'echo "Cleaning up..."' >> /terminate.sh && \
    echo 'curl --max-time 5 -X POST "https://api.example.com/release" || true' >> /terminate.sh && \
    chmod +x /terminate.sh

# Option 2: Copy from file
COPY terminate.sh /terminate.sh
RUN chmod +x /terminate.sh
```

### Requirements

1. **Must be at `/terminate.sh`** (or `/app/terminate.sh`)
2. **Must be executable** (`chmod +x`)
3. **Use timeouts** on network calls (`--max-time 5`) to avoid hanging
4. **Handle errors gracefully** (`|| true`) - don't fail the cleanup

### Example: Delete Kernel Browser Session

```bash
#!/bin/bash
# /terminate.sh - deletes Kernel browser session

if [ -f /tmp/kernel_session_id ]; then
  SESSION_ID=$(cat /tmp/kernel_session_id)
  echo "Deleting Kernel browser $SESSION_ID..."
  curl -sS --max-time 10 -X DELETE \
    "https://api.onkernel.com/browsers/$SESSION_ID" \
    -H "Authorization: Bearer $KERNEL_API_KEY" || true
  echo "Done"
fi
```

### Example: Close Database Connection

```bash
#!/bin/bash
# /terminate.sh - close DB connection

if [ -n "$DB_CONNECTION_ID" ]; then
  echo "Closing database connection..."
  curl --max-time 5 -X DELETE \
    "https://db.example.com/connections/$DB_CONNECTION_ID" || true
fi
```

### Example: Generic Cleanup

```bash
#!/bin/bash
# /terminate.sh - generic cleanup

echo "Running cleanup..."

# Kill any background processes
pkill -f "my-server" || true

# Remove temp files
rm -rf /tmp/session_* || true

# Notify external service
curl --max-time 5 -X POST "https://api.example.com/cleanup" \
  -d '{"container_id": "'$HOSTNAME'"}' || true

echo "Cleanup complete"
```

### Why /terminate.sh Instead of SIGTERM Traps?

| | /terminate.sh | SIGTERM trap |
|---|---|---|
| Complexity | Simple script | Complex bash (traps, PID handling, wait loops) |
| Guaranteed to run | Yes (we call it explicitly) | No (race with SIGKILL) |
| PID 1 issues | None | Yes (signals may not reach your process) |
| User code | Standalone file | Buried in entrypoint |

For ReinforceNow/Modal, `/terminate.sh` is simpler and more reliable

## Common Issues

| Issue | Fix |
|-------|-----|
| `Server not starting` | Modal ignores ENTRYPOINT - use wrapper script pattern above |
| `Permission denied` | Add `USER root` before creating files |
| `Heredoc not supported` | Use `RUN echo '...' > file` |
| `Platform mismatch` | Build with `--platform linux/amd64` |
| `Server not responding` | Check `supervisord.conf` or similar for actual startup commands |
| `403 Forbidden` | Add `--allowed-hosts '*'` to MCP server command (localhost restriction) |
| `Cleanup not running` | Create `/terminate.sh` instead of using trap |
