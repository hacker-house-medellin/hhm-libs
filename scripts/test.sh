#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
node --test tests/*.test.mjs
node --check src/index.mjs
python3 -m json.tool schemas/event-envelope.schema.json >/dev/null
python3 -m json.tool integrations/pins.json >/dev/null
git diff --check
if command -v cargo >/dev/null 2>&1; then
  cargo fmt --all -- --check
  cargo clippy --workspace --all-targets -- -D warnings
  cargo test --workspace --all-targets
fi
