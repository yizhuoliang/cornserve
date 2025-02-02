#!/usr/local/env bash

set -evo pipefail

PROTO_DIR="proto/v1"
PROTO_FILES=$(find $PROTO_DIR -name "*.proto")

PYTHON_OUTPUT_DIR="python/cornserve/services/pb"
mkdir -p "$PYTHON_OUTPUT_DIR"
for proto in $PROTO_FILES; do
  echo "Generating code for $proto in $PYTHON_OUTPUT_DIR"
  python -m grpc_tools.protoc \
    -I$PROTO_DIR \
    --python_out=$PYTHON_OUTPUT_DIR \
    --grpc_python_out=$PYTHON_OUTPUT_DIR \
    --pyi_out=$PYTHON_OUTPUT_DIR \
    $proto
done
