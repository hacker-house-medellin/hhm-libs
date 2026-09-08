# Chat, search, embeddings, and discovery

`hhm-libs` owns reusable behavior; `hhm-interfaces` and the pinned upstream
interface repositories own wire contracts. This package does not generate a
second competing chat or search API.

## Authorization boundary

The server must create `ActorContext` from verified Shared Auth middleware.
Subject, tenant, role, and scope fields from JSON, query parameters, headers
outside the trusted gateway contract, or WebSocket messages are untrusted and
must never be copied into that context.

The policy is fail closed:

| Surface | Required trusted context | Maximum corpus |
| --- | --- | --- |
| Sales visitor | no tenant selection | public |
| External search | no tenant selection | public |
| Customer support | authenticated, exact tenant, `chat:support` | customer |
| Admin/owner support | exact tenant, `chat:admin`, privileged role | internal |
| Internal search | exact tenant, `search:internal`, privileged role | internal |

Public search cannot select a tenant. Internal and customer searches cannot
cross a tenant boundary. Search retrieval is server-side: callers do not submit
provider URLs, SQL, vector operators, or index names.

## Embedding contract

The fleet storage width is 4,100 slots, with at most 4,096 provider/source
dimensions and four reserved zero-padding slots. The library:

- rejects empty, non-finite, over-width, and zero vectors;
- L2-normalizes before padding;
- rejects stored vectors with dimension, normalization, or padding drift;
- rejects comparisons across tenants or model revisions;
- caps a single in-process candidate set at 1,000 and output at 100;
- uses candidate id as the deterministic tie-break; and
- redacts values from Rust debug output.

The ORM owns database queries and indexes. This crate exposes no raw SQL,
connection, migration, provider credential, or index-selection API.

## Regression and correlation

`fit_linear` / `fitLinearRegression` computes deterministic ordinary least
squares, Pearson correlation, R-squared, and residual sum of squares for 10 to
10,000 finite observations. It rejects degenerate predictor or response
variance. Every result is explicitly `association_only`: it is not causal
inference and cannot authorize access, choose housing, set prices, or make
eligibility decisions.

## Privacy and telemetry

`IntelligenceAuditEvent` contains only the surface, outcome, and bounded counts.
It has no field for customer text, message bodies, tenant or subject ids,
citations, provider credentials, or embedding values. ORES telemetry exporters
must receive this metadata rather than serializing inputs or model payloads.

## Integration provenance

[`integrations/pins.json`](../integrations/pins.json) records immutable source
revisions. Some organization repositories are private, so public CI must not
attempt cross-private source fetches. Deployment packaging may consume an
authorized artifact for those contracts, but must verify its recorded revision.
