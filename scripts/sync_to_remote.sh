#!/bin/bash
# Usage: REMOTE_HOST=your-user@your-host bash scripts/sync_to_remote.sh
# Requires ~/.ssh/id_ed25519 and remote write access.

cd "$(dirname "$0")/.."

echo "Current dir: $(pwd)"
echo "Using filter file: $(pwd)/.rsync-filter"

if [ -z "$REMOTE_HOST" ]; then
  echo "REMOTE_HOST is not set. Please run with:"
  echo "REMOTE_HOST=user@your-remote-ip bash scripts/sync_to_remote.sh"
  exit 1
fi

rsync -avz \
  --filter="merge .rsync-filter" \
  -e "ssh -i ~/.ssh/id_ed25519" \
  ./ "$REMOTE_HOST:~/mini-tensor"
