#!/bin/bash
# Delete browser pool after training
# Uses pool name (not ID) - force=true terminates all acquired browsers

if [ -z "$KERNEL_API_KEY" ]; then
  echo "Error: KERNEL_API_KEY not set"
  exit 1
fi

POOL_NAME="${KERNEL_POOL_NAME:-rnow-browsers}"

echo "Deleting pool '$POOL_NAME' (force=true)..."
curl -sS -X DELETE "https://api.onkernel.com/browser_pools/$POOL_NAME" \
  -H "Authorization: Bearer $KERNEL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"force": true}' && echo "Pool deleted"
