#!/bin/bash
# Create browser pool. Run: source setup_pool.sh [size]
POOL_SIZE="${1:-100}"

if [ -z "$KERNEL_API_KEY" ]; then
  echo "Error: KERNEL_API_KEY not set"
  return 1
fi

echo "Creating pool with $POOL_SIZE browsers..."
RESP=$(curl -sS -X POST "https://api.onkernel.com/browser_pools" \
  -H "Authorization: Bearer $KERNEL_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"name\": \"rnow-browsers\", \"size\": $POOL_SIZE, \"timeout_seconds\": 300, \"fill_rate_per_minute\": 25, \"headless\": false, \"viewport\": {\"width\": 1024, \"height\": 768}}")

echo "$RESP"

export KERNEL_POOL_NAME="rnow-browsers"

echo ""
echo "Pool created. Browsers fill at 25/min (sandboxes wait up to 5min for availability)."
echo "Run: rnow run"
