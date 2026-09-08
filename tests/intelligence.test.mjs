import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import {
  authorizeIntelligenceSurface,
  cosineSimilarity,
  EMBEDDING_SOURCE_DIMENSIONS_MAX,
  EMBEDDING_STORAGE_DIMENSIONS,
  fitLinearRegression,
  makeIntelligenceAuditEvent,
  normalizeEmbedding,
  rankEmbeddings,
  validateStoredEmbedding,
} from '../src/index.mjs';

const tenantId = 'tenant_medellin';
const trustedActor = Object.freeze({
  subject: 'subject_123',
  tenantId,
  roles: ['resident'],
  scopes: ['chat:support'],
});

test('sales and external search are restricted to the public corpus', () => {
  assert.deepEqual(authorizeIntelligenceSurface('sales_visitor', null), { ok: true, visibility: 'public' });
  assert.deepEqual(authorizeIntelligenceSurface('external_search', tenantId), {
    ok: false,
    error: 'public_surface_cannot_select_tenant',
  });
});

test('customer support requires authentication, fixed scope, and exact tenant', () => {
  assert.deepEqual(authorizeIntelligenceSurface('customer_support', tenantId, trustedActor), {
    ok: true,
    visibility: 'customer',
  });
  assert.equal(authorizeIntelligenceSurface('customer_support', 'tenant_other', trustedActor).error, 'tenant_mismatch');
  assert.equal(authorizeIntelligenceSurface('customer_support', tenantId, {}).error, 'authentication_required');
  assert.equal(
    authorizeIntelligenceSurface('customer_support', tenantId, { ...trustedActor, scopes: [] }).error,
    'scope_required',
  );
});

test('admin and internal search require a privileged role and their exact scope', () => {
  const owner = { subject: 'subject_owner', tenantId, roles: ['owner'], scopes: ['chat:admin', 'search:internal'] };
  assert.equal(authorizeIntelligenceSurface('admin_owner_support', tenantId, owner).visibility, 'internal');
  assert.equal(authorizeIntelligenceSurface('internal_search', tenantId, owner).visibility, 'internal');
  assert.equal(authorizeIntelligenceSurface('internal_search', tenantId, trustedActor).error, 'scope_required');
  assert.equal(
    authorizeIntelligenceSurface('internal_search', tenantId, { ...trustedActor, scopes: ['search:internal'] }).error,
    'privileged_role_required',
  );
});

test('unknown intelligence surfaces fail closed', () => {
  assert.equal(authorizeIntelligenceSurface('owner_debug', tenantId, trustedActor).error, 'unknown_surface');
});

test('normalization pads provider values to the fleet 4100-slot width', () => {
  const embedding = normalizeEmbedding({ tenantId, modelRevision: 'model_v1', values: [3, 4] });
  assert.equal(embedding.values.length, EMBEDDING_STORAGE_DIMENSIONS);
  assert.ok(Math.abs(embedding.values[0] - 0.6) < 1e-12);
  assert.ok(Math.abs(embedding.values[1] - 0.8) < 1e-12);
  assert.ok(embedding.values.slice(2).every((value) => value === 0));
  assert.ok(Object.isFrozen(embedding.values));
});

test('normalization rejects empty, zero, non-finite, and oversized vectors', () => {
  assert.throws(() => normalizeEmbedding({ tenantId, modelRevision: 'v1', values: [] }), /empty_source/);
  assert.throws(() => normalizeEmbedding({ tenantId, modelRevision: 'v1', values: [0, 0] }), /zero_vector/);
  assert.throws(() => normalizeEmbedding({ tenantId, modelRevision: 'v1', values: [Number.NaN] }), /non_finite_value/);
  assert.throws(
    () => normalizeEmbedding({ tenantId, modelRevision: 'v1', values: new Array(EMBEDDING_SOURCE_DIMENSIONS_MAX + 1).fill(1) }),
    /source_too_wide/,
  );
});

test('embedding identity rejects URL-shaped and traversal-shaped metadata', () => {
  assert.throws(() => normalizeEmbedding({ tenantId: '../other', modelRevision: 'v1', values: [1] }), /invalid_tenant/);
  assert.throws(() => normalizeEmbedding({ tenantId: 'tenant other', modelRevision: 'v1', values: [1] }), /invalid_tenant/);
  assert.throws(() => normalizeEmbedding({ tenantId, modelRevision: 'https://provider', values: [1] }), /invalid_model_revision/);
});

test('stored embedding validation rejects width, norm, non-finite, and padding drift', () => {
  assert.throws(() => validateStoredEmbedding({ tenantId, modelRevision: 'v1', values: [1] }), /invalid_storage_dimensions/);
  const zero = new Array(EMBEDDING_STORAGE_DIMENSIONS).fill(0);
  assert.throws(() => validateStoredEmbedding({ tenantId, modelRevision: 'v1', values: zero }), /zero_vector/);
  const badNorm = [...zero];
  badNorm[0] = 2;
  assert.throws(() => validateStoredEmbedding({ tenantId, modelRevision: 'v1', values: badNorm }), /not_normalized/);
  const badPadding = [...zero];
  badPadding[0] = 1;
  badPadding[EMBEDDING_SOURCE_DIMENSIONS_MAX] = 0.25;
  assert.throws(() => validateStoredEmbedding({ tenantId, modelRevision: 'v1', values: badPadding }), /non_zero_padding/);
  const nonFinite = [...zero];
  nonFinite[0] = Number.POSITIVE_INFINITY;
  assert.throws(() => validateStoredEmbedding({ tenantId, modelRevision: 'v1', values: nonFinite }), /non_finite_value/);
});

test('cosine similarity rejects cross-tenant and cross-model comparisons', () => {
  const query = normalizeEmbedding({ tenantId, modelRevision: 'v1', values: [1, 0] });
  const otherTenant = normalizeEmbedding({ tenantId: 'tenant_other', modelRevision: 'v1', values: [1, 0] });
  const otherModel = normalizeEmbedding({ tenantId, modelRevision: 'v2', values: [1, 0] });
  assert.throws(() => cosineSimilarity(query, otherTenant), /tenant_mismatch/);
  assert.throws(() => cosineSimilarity(query, otherModel), /model_revision_mismatch/);
  assert.throws(
    () => cosineSimilarity(query, { tenantId, modelRevision: 'v1', values: new Array(4100).fill(0) }),
    /validated_embedding_required/,
  );
});

test('ranking is deterministic and uses candidate id as a tie-break', () => {
  const query = normalizeEmbedding({ tenantId, modelRevision: 'v1', values: [1, 0] });
  const same = normalizeEmbedding({ tenantId, modelRevision: 'v1', values: [1, 0] });
  const orthogonal = normalizeEmbedding({ tenantId, modelRevision: 'v1', values: [0, 1] });
  const ranked = rankEmbeddings(query, [
    { candidateId: 'z_candidate', embedding: same },
    { candidateId: 'a_candidate', embedding: same },
    { candidateId: 'middle', embedding: orthogonal },
  ]);
  assert.deepEqual(ranked.map((hit) => hit.candidateId), ['a_candidate', 'z_candidate', 'middle']);
});

test('ranking rejects invalid ids, limits, excess candidates, and tenant mixing', () => {
  const query = normalizeEmbedding({ tenantId, modelRevision: 'v1', values: [1] });
  assert.throws(() => rankEmbeddings(query, [{ candidateId: '../bad', embedding: query }]), /invalid_candidate_id/);
  assert.throws(() => rankEmbeddings(query, [], 101), /invalid_limit/);
  assert.throws(
    () => rankEmbeddings(query, new Array(1001).fill({ candidateId: 'item', embedding: query })),
    /too_many_candidates/,
  );
  const foreign = normalizeEmbedding({ tenantId: 'tenant_other', modelRevision: 'v1', values: [1] });
  assert.throws(() => rankEmbeddings(query, [{ candidateId: 'foreign', embedding: foreign }]), /tenant_mismatch/);
});

test('linear regression recovers a known line and perfect correlation', () => {
  const points = Array.from({ length: 10 }, (_, predictor) => ({ predictor, response: 2 * predictor + 3 }));
  const fit = fitLinearRegression(points);
  assert.ok(Math.abs(fit.slope - 2) < 1e-12);
  assert.ok(Math.abs(fit.intercept - 3) < 1e-12);
  assert.ok(Math.abs(fit.pearsonR - 1) < 1e-12);
  assert.ok(Math.abs(fit.rSquared - 1) < 1e-12);
  assert.equal(fit.interpretation, 'association_only');
});

test('linear regression rejects invalid, constant, and unbounded cohorts', () => {
  assert.throws(() => fitLinearRegression([{ predictor: 1, response: 2 }]), /too_few_observations/);
  assert.throws(
    () => fitLinearRegression(new Array(10).fill(null).map((_, response) => ({ predictor: 1, response }))),
    /zero_predictor_variance/,
  );
  assert.throws(
    () => fitLinearRegression(new Array(10).fill(null).map((_, predictor) => ({ predictor, response: 1 }))),
    /zero_response_variance/,
  );
  const nonFinite = new Array(10).fill(null).map((_, predictor) => ({ predictor, response: predictor }));
  nonFinite[0].response = Number.NEGATIVE_INFINITY;
  assert.throws(() => fitLinearRegression(nonFinite), /non_finite_observation/);
});

test('audit events cannot carry customer content or identities', () => {
  const event = makeIntelligenceAuditEvent({ surface: 'customer_support', status: 'completed', candidateCount: 12, resultCount: 3 });
  assert.deepEqual(Object.keys(event), ['operation', 'surface', 'status', 'candidateCount', 'resultCount']);
  assert.ok(!JSON.stringify(event).includes(tenantId));
  assert.throws(() => makeIntelligenceAuditEvent({ surface: 'x', status: 'debug' }), /invalid_audit_status/);
});

test('integration provenance is immutable and contains only full commit ids', async () => {
  const pins = JSON.parse(await readFile(new URL('../integrations/pins.json', import.meta.url), 'utf8'));
  assert.equal(Object.keys(pins.repositories).length, 6);
  for (const [repository, record] of Object.entries(pins.repositories)) {
    assert.match(repository, /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/);
    assert.match(record.revision, /^[0-9a-f]{40}$/);
    assert.ok(['public', 'private'].includes(record.visibility));
    assert.ok(record.purpose.length > 20);
  }
});
