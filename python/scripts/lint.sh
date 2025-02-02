#!/usr/bin/env bash

set -evo pipefail

echo ${BASH_SOURCE[0]}

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -z $GITHUB_ACTION ]]; then
  black cornserve
else
  black --check cornserve
fi

ruff check cornserve
pyright cornserve
