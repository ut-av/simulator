#!/bin/bash
# Download the prebuilt Unity simulator into simulator/build (gitignored), which
# is where src/utils/sim_starter.py looks for it. Does nothing if it is already
# there; delete simulator/build to re-download.

set -e

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
SIM_BUILD_DIR="$REPO_DIR/simulator/build"
SIM_URL="https://github.com/ut-av/simulator/releases/download/v1.0.0/linux.zip"

if [ -d "$SIM_BUILD_DIR" ] && [ "$(ls -A "$SIM_BUILD_DIR")" ]; then
    echo "Simulator already installed in $SIM_BUILD_DIR, delete the directory to re-download."
    exit 0
fi

echo "Downloading simulator release..."
mkdir -p "$SIM_BUILD_DIR"
TMP_DIR="$(mktemp -d)"
TMP_ZIP="$TMP_DIR/linux.zip"
curl -fL -o "$TMP_ZIP" "$SIM_URL"
unzip -o -q "$TMP_ZIP" -d "$SIM_BUILD_DIR"
rm -r "$TMP_DIR"
echo "Simulator installed in $SIM_BUILD_DIR"
