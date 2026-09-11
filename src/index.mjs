export const serviceCatalog = Object.freeze({
  org: "hacker-house-medellin",
  title: "Hacker House Medellín",
  tagline: "A living operating system for technical coliving, coworking, and community build weeks in Medellín.",
  capabilities: ['intake', 'events', 'alerts', 'leads', 'status', 'analytics', 'chat', 'search', 'discovery'],
  integrations: ["Stripe", "Google Calendar", "WhatsApp links", "Slack/Discord", "Door/access logs", "Maps"],
});

export function normalizeEmail(email) {
  if (typeof email !== 'string') return null;
  const value = email.trim().toLowerCase();
  return /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(value) ? value : null;
}

export function validateLead(input) {
  if (!input || typeof input !== 'object') return { ok: false, error: 'lead must be an object' };
  const email = normalizeEmail(input.email);
  if (!email) return { ok: false, error: 'valid email is required' };
  const name = String(input.name || '').trim();
  if (name.length < 2) return { ok: false, error: 'name must be at least two characters' };
  return { ok: true, value: { ...input, email, name } };
}

export function makeEvent(type, payload = {}, meta = {}) {
  if (!/^[a-z][a-z0-9_.-]+$/.test(type)) throw new TypeError('event type must be a namespaced lowercase identifier');
  return { id: meta.id || crypto.randomUUID(), type, payload, product: serviceCatalog.org, occurredAt: meta.occurredAt || new Date().toISOString() };
}

export function classifyPriority(signal) {
  const severity = String(signal?.severity || '').toLowerCase();
  if (['critical', 'p0', 'sev0', 'sev1'].includes(severity)) return 'urgent';
  if (['high', 'p1', 'sev2'].includes(severity)) return 'high';
  if (['medium', 'p2', 'warn', 'warning'].includes(severity)) return 'normal';
  return 'low';
}

export const EMBEDDING_SOURCE_DIMENSIONS_MAX = 4096;
export const EMBEDDING_STORAGE_DIMENSIONS = 4100;
export const DISCOVERY_OBSERVATIONS_MIN = 10;
export const DISCOVERY_OBSERVATIONS_MAX = 10_000;
export const SEARCH_CANDIDATES_MAX = 1_000;

const publicSurfaces = new Set(['sales_visitor', 'external_search']);
const privilegedRoles = new Set(['house_manager', 'owner', 'administrator']);
const embeddingBrand = Symbol('hhm.embedding');

function validOpaque(value) {
  return typeof value === 'string'
    && /^[A-Za-z0-9][A-Za-z0-9._:/-]*$/u.test(value)
    && value.length <= 256
    && !value.includes('..')
    && !value.includes('://');
}

/**
 * Resolve the maximum visible corpus for a trusted server-side actor context.
 * Roles, scopes, subject and tenant must come from authentication middleware,
 * never from a browser request body.
 */
export function authorizeIntelligenceSurface(surface, requestedTenant, actor = {}) {
  if (publicSurfaces.has(surface)) {
    if (requestedTenant !== undefined && requestedTenant !== null) {
      return { ok: false, error: 'public_surface_cannot_select_tenant' };
    }
    return { ok: true, visibility: 'public' };
  }

  const authenticated = validOpaque(actor.subject) && validOpaque(actor.tenantId);
  if (!authenticated) return { ok: false, error: 'authentication_required' };
  if (!validOpaque(requestedTenant) || requestedTenant !== actor.tenantId) {
    return { ok: false, error: 'tenant_mismatch' };
  }

  const scopes = new Set(Array.isArray(actor.scopes) ? actor.scopes : []);
  const roles = new Set(Array.isArray(actor.roles) ? actor.roles : []);
  if (surface === 'customer_support') {
    return scopes.has('chat:support')
      ? { ok: true, visibility: 'customer' }
      : { ok: false, error: 'scope_required' };
  }
  if (surface === 'admin_owner_support' || surface === 'internal_search') {
    const scope = surface === 'admin_owner_support' ? 'chat:admin' : 'search:internal';
    if (!scopes.has(scope)) return { ok: false, error: 'scope_required' };
    if (![...roles].some((role) => privilegedRoles.has(role))) {
      return { ok: false, error: 'privileged_role_required' };
    }
    return { ok: true, visibility: 'internal' };
  }
  return { ok: false, error: 'unknown_surface' };
}

function validateEmbeddingIdentity(tenantId, modelRevision) {
  if (!validOpaque(tenantId)) throw new TypeError('invalid_tenant');
  if (!validOpaque(modelRevision)) throw new TypeError('invalid_model_revision');
}

/** Normalize a provider vector and pad it to the fleet 4100-slot storage width. */
export function normalizeEmbedding({ tenantId, modelRevision, values }) {
  validateEmbeddingIdentity(tenantId, modelRevision);
  if (!Array.isArray(values) && !ArrayBuffer.isView(values)) throw new TypeError('values_must_be_array');
  const source = Array.from(values);
  if (source.length === 0) throw new RangeError('empty_source');
  if (source.length > EMBEDDING_SOURCE_DIMENSIONS_MAX) throw new RangeError('source_too_wide');
  if (source.some((value) => typeof value !== 'number' || !Number.isFinite(value))) {
    throw new TypeError('non_finite_value');
  }
  const squaredNorm = source.reduce((sum, value) => sum + value * value, 0);
  if (!Number.isFinite(squaredNorm) || squaredNorm <= Number.EPSILON) throw new RangeError('zero_vector');
  const norm = Math.sqrt(squaredNorm);
  const normalized = source.map((value) => value / norm);
  normalized.length = EMBEDDING_STORAGE_DIMENSIONS;
  normalized.fill(0, source.length);
  return Object.freeze({ tenantId, modelRevision, values: Object.freeze(normalized), [embeddingBrand]: true });
}

/** Validate an exact-width normalized vector returned by a persistence adapter. */
export function validateStoredEmbedding({ tenantId, modelRevision, values }) {
  validateEmbeddingIdentity(tenantId, modelRevision);
  if (!Array.isArray(values) || values.length !== EMBEDDING_STORAGE_DIMENSIONS) {
    throw new RangeError('invalid_storage_dimensions');
  }
  if (values.some((value) => typeof value !== 'number' || !Number.isFinite(value))) {
    throw new TypeError('non_finite_value');
  }
  if (values.slice(EMBEDDING_SOURCE_DIMENSIONS_MAX).some((value) => value !== 0)) {
    throw new RangeError('non_zero_padding');
  }
  const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
  if (norm <= Number.EPSILON) throw new RangeError('zero_vector');
  if (Math.abs(norm - 1) > 1e-4) throw new RangeError('not_normalized');
  return Object.freeze({ tenantId, modelRevision, values: Object.freeze([...values]), [embeddingBrand]: true });
}

export function cosineSimilarity(left, right) {
  if (left?.[embeddingBrand] !== true || right?.[embeddingBrand] !== true) throw new TypeError('validated_embedding_required');
  if (left.tenantId !== right.tenantId) throw new TypeError('tenant_mismatch');
  if (left.modelRevision !== right.modelRevision) throw new TypeError('model_revision_mismatch');
  if (left.values.length !== EMBEDDING_STORAGE_DIMENSIONS || right.values.length !== EMBEDDING_STORAGE_DIMENSIONS) {
    throw new RangeError('invalid_storage_dimensions');
  }
  let similarity = 0;
  for (let index = 0; index < EMBEDDING_STORAGE_DIMENSIONS; index += 1) {
    similarity += left.values[index] * right.values[index];
  }
  return Math.max(-1, Math.min(1, similarity));
}

export function rankEmbeddings(query, candidates, limit = 10) {
  if (!Array.isArray(candidates)) throw new TypeError('candidates_must_be_array');
  if (candidates.length > SEARCH_CANDIDATES_MAX) throw new RangeError('too_many_candidates');
  if (!Number.isSafeInteger(limit) || limit < 0 || limit > 100) throw new RangeError('invalid_limit');
  const hits = candidates.map(({ candidateId, embedding }) => {
    if (!validOpaque(candidateId)) throw new TypeError('invalid_candidate_id');
    return { candidateId, similarity: cosineSimilarity(query, embedding) };
  });
  hits.sort((left, right) => {
    const bySimilarity = right.similarity - left.similarity;
    if (bySimilarity !== 0) return bySimilarity;
    if (left.candidateId < right.candidateId) return -1;
    if (left.candidateId > right.candidateId) return 1;
    return 0;
  });
  return hits.slice(0, limit);
}

/** Deterministic OLS and Pearson correlation. Association is not causation. */
export function fitLinearRegression(observations) {
  if (!Array.isArray(observations) || observations.length < DISCOVERY_OBSERVATIONS_MIN) {
    throw new RangeError('too_few_observations');
  }
  if (observations.length > DISCOVERY_OBSERVATIONS_MAX) throw new RangeError('too_many_observations');
  if (observations.some(({ predictor, response }) => !Number.isFinite(predictor) || !Number.isFinite(response))) {
    throw new TypeError('non_finite_observation');
  }
  const count = observations.length;
  const meanX = observations.reduce((sum, point) => sum + point.predictor, 0) / count;
  const meanY = observations.reduce((sum, point) => sum + point.response, 0) / count;
  let sumXX = 0;
  let sumYY = 0;
  let sumXY = 0;
  for (const point of observations) {
    const centeredX = point.predictor - meanX;
    const centeredY = point.response - meanY;
    sumXX += centeredX * centeredX;
    sumYY += centeredY * centeredY;
    sumXY += centeredX * centeredY;
  }
  if (sumXX <= Number.EPSILON) throw new RangeError('zero_predictor_variance');
  if (sumYY <= Number.EPSILON) throw new RangeError('zero_response_variance');
  const slope = sumXY / sumXX;
  const intercept = meanY - slope * meanX;
  const pearsonR = Math.max(-1, Math.min(1, sumXY / Math.sqrt(sumXX * sumYY)));
  const rSquared = Math.max(0, Math.min(1, pearsonR * pearsonR));
  const residualSumSquares = observations.reduce((sum, point) => {
    const residual = point.response - (intercept + slope * point.predictor);
    return sum + residual * residual;
  }, 0);
  const result = { observations: count, slope, intercept, pearsonR, rSquared, residualSumSquares, interpretation: 'association_only' };
  if (Object.values(result).some((value) => typeof value === 'number' && !Number.isFinite(value))) {
    throw new RangeError('non_finite_result');
  }
  return Object.freeze(result);
}

/** Build telemetry metadata without customer text, subject ids, tenant ids, or vectors. */
export function makeIntelligenceAuditEvent({ surface, status, candidateCount = 0, resultCount = 0 }) {
  if (!['allowed', 'denied', 'completed', 'failed'].includes(status)) throw new TypeError('invalid_audit_status');
  if (!Number.isSafeInteger(candidateCount) || candidateCount < 0 || candidateCount > SEARCH_CANDIDATES_MAX) {
    throw new RangeError('invalid_candidate_count');
  }
  if (!Number.isSafeInteger(resultCount) || resultCount < 0 || resultCount > 100) {
    throw new RangeError('invalid_result_count');
  }
  return Object.freeze({ operation: 'intelligence', surface, status, candidateCount, resultCount });
}
