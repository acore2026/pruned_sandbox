#!/bin/sh
set -eu

LOG_DIR=${LOG_DIR:-/app/logs}
mkdir -p "$LOG_DIR"
chown sandbox:sandbox "$LOG_DIR"
export LOG_DIR

exec "$@"
