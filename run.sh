#!/bin/bash
# Launch wispr — hold Right Ctrl to dictate into any window
# Set WISPR_MODEL=base.en for faster/less-accurate, medium.en for slower/better
export DISPLAY="${DISPLAY:-:1}"
cd "$(dirname "$0")"
exec python3 wispr.py "$@"
