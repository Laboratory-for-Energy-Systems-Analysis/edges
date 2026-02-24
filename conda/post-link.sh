#!/usr/bin/env bash
set -euo pipefail

# CLIPSpy is only available on PyPI.
"${PREFIX}/bin/python" -m pip install --no-deps --disable-pip-version-check clipspy
