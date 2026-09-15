#!/usr/bin/env bash
set -euo pipefail
ROOT=/opt/open-notebook
KEEP_DAYS=14
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
DEST="$ROOT/backups/snapshots/$STAMP"
LOG="$ROOT/backups/backup-$STAMP.log"
mkdir -p "$DEST" "$ROOT/backups"
exec > >(tee -a "$LOG") 2>&1
echo "=== local backup $STAMP ==="
cd "$ROOT"
docker compose stop open_notebook surrealdb
tar czf "$DEST/open-notebook.tar.gz" surreal_data notebook_data .env docker-compose.yml CREDENTIALS.txt embeddings
docker compose start surrealdb embeddings speaches open_notebook
ln -sfn "$DEST" "$ROOT/backups/latest"
cp -f "$DEST/open-notebook.tar.gz" "$ROOT/backups/latest.tar.gz"
chmod 600 "$ROOT/backups/latest.tar.gz"
du -sh "$DEST/open-notebook.tar.gz"
find "$ROOT/backups/snapshots" -mindepth 1 -maxdepth 1 -type d -mtime +$KEEP_DAYS -exec rm -rf {} +
echo "=== done $DEST ==="
