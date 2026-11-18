#!/usr/bin/env bash
# Cross-compile a single Windows exe (UI embedded). Needs only Go >= 1.22.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p dist
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -trimpath -ldflags "-s -w -H=windowsgui" -o dist/desktop-agent.exe .
cp -n config.example.json dist/config.json
ls -la dist
