#!/usr/bin/env bash

set -evo pipefail

docker run -d \
  -p 5000:5000 \
  -e REGISTRY_STORAGE_DELETE_ENABLED=true \
  --restart=always \
  --name cornserve-registry \
  --volume /data/cornserve/registry:/var/lib/registry \
  registry:2
