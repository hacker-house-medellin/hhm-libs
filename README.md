# hhm-libs

Shared, runtime-light libraries for **Hacker House Medellín**.

- `crates/contracts` — stable event, actor, and request metadata contracts
- `crates/intelligence` — fail-closed chat/search policy, 4,100-slot embedding validation, deterministic correlation and regression
- `crates/routing` — deterministic routing and priority classification
- `src/` — JavaScript reference implementation for Workers and web tooling
- `schemas/` — JSON Schema documents for language-neutral validation
- `integrations/pins.json` — immutable provenance for middleware, rate-limit, telemetry, sync, chat, and H/HAUS interface contracts

The Rust crates deliberately use only the standard library in this bootstrap, keeping audits and downstream embedding straightforward.

```bash
./scripts/test.sh
```

See [the intelligence boundary](docs/intelligence.md) for authorization,
privacy, vector, discovery, and packaging invariants.
