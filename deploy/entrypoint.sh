#!/bin/sh
set -eu

LOG_DIR=${LOG_DIR:-/app/logs}
mkdir -p "$LOG_DIR"
chown sandbox:sandbox "$LOG_DIR"
export LOG_DIR

case "${CONFIGURE_N6_ROUTES:-true}" in
    1|true|TRUE|yes|YES)
        ip route replace "${UE_INTERNET_CIDR:-10.60.0.0/16}" via "${UPF_N6_IP:-172.30.0.2}"
        ip route replace "${UE_ACN_CIDR:-10.61.0.0/16}" via "${UPF_N6_IP:-172.30.0.2}"
        ;;
esac

exec "$@"
