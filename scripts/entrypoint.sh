#!/usr/bin/env bash
set -e

KG_PATH="${KG_PATH:-/app/chunk_metadata/kg.pkl}"
META_PATH="${META_PATH:-/app/chunk_metadata/chunk_metadata.json}"

if [ -f "$META_PATH" ] && [ ! -f "$KG_PATH" ]; then
    echo "[entrypoint] kg.pkl missing — building knowledge graph from $META_PATH"
    python /app/scripts/build_kg.py --metadata "$META_PATH" --output "$KG_PATH"
elif [ -f "$KG_PATH" ]; then
    echo "[entrypoint] kg.pkl present at $KG_PATH — skipping build"
else
    echo "[entrypoint] chunk_metadata.json not found — skipping KG build"
fi

exec "$@"
