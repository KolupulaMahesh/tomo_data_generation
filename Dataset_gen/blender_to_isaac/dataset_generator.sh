#!/usr/bin/env bash
# The dataset generator's window (dataset_generator.py) with this folder's own Python (env/). Separate from the
# training window. One Isaac Sim at a time.
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$D/env/bin/python" "$D/dataset_generator.py" "$@"
