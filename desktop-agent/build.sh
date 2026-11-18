#!/usr/bin/env bash
# Cross-compile release packages. Needs only Go (the PDF library is vendored → works offline).
#   ./build.sh            → Windows + Linux (amd64, arm64)
#   ./build.sh windows    → only Windows        ./build.sh linux → only Linux
#   macOS also builds (GOOS=darwin) but is untested – add it to the list below if you need it.
set -euo pipefail
cd "$(dirname "$0")"
what="${1:-all}"
rm -rf dist && mkdir -p dist
ver="$(date +%Y%m%d)"
common() { cp docs/HUONG_DAN.md docs/HUONG_DAN_TAO_BOT.md "$1/"; cp config.example.json "$1/config.json"; }

if [[ "$what" == all || "$what" == windows ]]; then
  d=dist/windows; mkdir -p $d
  GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -trimpath -ldflags "-s -w -H=windowsgui" -o $d/desktop-agent.exe .
  common $d
  (cd $d && python3 -c "
import zipfile,sys;z=zipfile.ZipFile('../../desktop-agent-win.zip','w',zipfile.ZIP_DEFLATED)
[z.write(f) for f in ('desktop-agent.exe','config.json','HUONG_DAN.md','HUONG_DAN_TAO_BOT.md')]")
fi
if [[ "$what" == all || "$what" == linux ]]; then
  for arch in amd64 arm64; do
    d=dist/linux-$arch; mkdir -p $d/desktop-agent
    GOOS=linux GOARCH=$arch CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o $d/desktop-agent/desktop-agent .
    common $d/desktop-agent; cp desktop-agent.service $d/desktop-agent/
    tar -C $d -czf dist/desktop-agent-linux-$arch.tar.gz desktop-agent
    cp dist/desktop-agent-linux-$arch.tar.gz ./desktop-agent-linux-$arch.tar.gz
  done
fi
ls -la dist/*.zip dist/*.tar.gz 2>/dev/null || true; ls -la desktop-agent-win.zip desktop-agent-linux-*.tar.gz 2>/dev/null || true
