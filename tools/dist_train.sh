#!/usr/bin/env bash

# Launch distributed ViewOcc training.
CONFIG=$1
GPUS=$2
WORK_DIR=$3
PORT=${PORT:-28108}

PYTHONPATH="$(dirname "$0")/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node="$GPUS" \
    --master_port="$PORT" "$(dirname "$0")/train.py" "$CONFIG" \
    --work-dir "$WORK_DIR" --launcher pytorch "${@:4}"
