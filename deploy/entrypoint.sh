#!/bin/sh
set -eu

LOG_DIR=${LOG_DIR:-/app/logs}
mkdir -p "$LOG_DIR"
# This bind-mounted directory is shared with host-local runs. Keep both the
# host user and the container service user able to append and rotate files.
chmod a+rwx "$LOG_DIR"
find "$LOG_DIR" -maxdepth 1 -type f -name '*.log*' -exec chmod a+rw {} +
umask 0000
export LOG_DIR

exec "$@"
