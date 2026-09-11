//! Runtime-light intelligence behavior for H/HAUS.
//!
//! Authentication material is established by the server middleware. This crate
//! never trusts roles, scopes, subjects, or tenant identifiers supplied in a
//! browser request body. Search text and embedding values are deliberately
//! absent from every audit type in this module.

use std::cmp::Ordering;
use std::fmt;

pub const EMBEDDING_SOURCE_DIMENSIONS_MAX: usize = 4096;
pub const EMBEDDING_STORAGE_DIMENSIONS: usize = 4100;
pub const DISCOVERY_OBSERVATIONS_MIN: usize = 10;
pub const DISCOVERY_OBSERVATIONS_MAX: usize = 10_000;
pub const SEARCH_CANDIDATES_MAX: usize = 1_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Role {
    Resident,
    HouseManager,
    Owner,
    Administrator,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntelligenceSurface {
    SalesVisitor,
    CustomerSupport,
    AdminOwnerSupport,
    ExternalSearch,
    InternalSearch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchVisibility {
    Public,
    Customer,
    Internal,
}

#[derive(Clone, Copy)]
pub struct ActorContext<'a> {
    pub subject: Option<&'a str>,
    pub tenant_id: Option<&'a str>,
    pub roles: &'a [Role],
    pub scopes: &'a [&'a str],
}

impl fmt::Debug for ActorContext<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ActorContext")
            .field("authenticated", &self.authenticated())
            .field("tenant_bound", &self.tenant_id.is_some())
            .field("role_count", &self.roles.len())
            .field("scope_count", &self.scopes.len())
            .finish()
    }
}

impl ActorContext<'_> {
    fn authenticated(&self) -> bool {
        self.subject.is_some_and(valid_opaque) && self.tenant_id.is_some_and(valid_opaque)
    }

    fn has_scope(&self, required: &str) -> bool {
        self.scopes.contains(&required)
    }

    fn has_privileged_role(&self) -> bool {
        self.roles
            .iter()
            .any(|role| matches!(role, Role::HouseManager | Role::Owner | Role::Administrator))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyError {
    PublicSurfaceCannotSelectTenant,
    AuthenticationRequired,
    TenantMismatch,
    ScopeRequired,
    PrivilegedRoleRequired,
}

/// Returns the maximum corpus visibility for a server-authenticated context.
///
/// Public surfaces cannot select a tenant. Authenticated surfaces require the
/// same tenant in the trusted actor context and the request route, plus a fixed
/// scope. Admin and internal search also require a privileged role.
pub fn authorize_surface(
    surface: IntelligenceSurface,
    requested_tenant: Option<&str>,
    actor: ActorContext<'_>,
) -> Result<SearchVisibility, PolicyError> {
    match surface {
        IntelligenceSurface::SalesVisitor | IntelligenceSurface::ExternalSearch => {
            if requested_tenant.is_some() {
                return Err(PolicyError::PublicSurfaceCannotSelectTenant);
            }
            Ok(SearchVisibility::Public)
        }
        IntelligenceSurface::CustomerSupport => {
            require_same_tenant(requested_tenant, actor)?;
            if !actor.has_scope("chat:support") {
                return Err(PolicyError::ScopeRequired);
            }
            Ok(SearchVisibility::Customer)
        }
        IntelligenceSurface::AdminOwnerSupport => {
            require_same_tenant(requested_tenant, actor)?;
            if !actor.has_scope("chat:admin") {
                return Err(PolicyError::ScopeRequired);
            }
            if !actor.has_privileged_role() {
                return Err(PolicyError::PrivilegedRoleRequired);
            }
            Ok(SearchVisibility::Internal)
        }
        IntelligenceSurface::InternalSearch => {
            require_same_tenant(requested_tenant, actor)?;
            if !actor.has_scope("search:internal") {
                return Err(PolicyError::ScopeRequired);
            }
            if !actor.has_privileged_role() {
                return Err(PolicyError::PrivilegedRoleRequired);
            }
            Ok(SearchVisibility::Internal)
        }
    }
}

fn require_same_tenant(
    requested_tenant: Option<&str>,
    actor: ActorContext<'_>,
) -> Result<(), PolicyError> {
    if !actor.authenticated() {
        return Err(PolicyError::AuthenticationRequired);
    }
    let requested_tenant = requested_tenant.filter(|value| valid_opaque(value));
    if requested_tenant != actor.tenant_id {
        return Err(PolicyError::TenantMismatch);
    }
    Ok(())
}

fn valid_opaque(value: &str) -> bool {
    if value.is_empty() || value.len() > 256 || value.contains("..") || value.contains("://") {
        return false;
    }
    let mut characters = value.chars();
    characters
        .next()
        .is_some_and(|character| character.is_ascii_alphanumeric())
        && characters.all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | ':' | '/' | '-')
        })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddingError {
    InvalidTenant,
    InvalidModelRevision,
    EmptySource,
    SourceTooWide,
    InvalidStorageDimensions,
    NonFiniteValue,
    ZeroVector,
    NotNormalized,
    NonZeroPadding,
    TenantMismatch,
    ModelRevisionMismatch,
    TooManyCandidates,
    InvalidCandidateId,
}

#[derive(Clone)]
pub struct Embedding {
    tenant_id: String,
    model_revision: String,
    values: Vec<f32>,
}

impl fmt::Debug for Embedding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Embedding")
            .field("tenant_bound", &true)
            .field("model_revision_bound", &true)
            .field("dimensions", &self.values.len())
            .field("values", &"[REDACTED]")
            .finish()
    }
}

impl Embedding {
    /// Normalizes at most 4096 source values and pads them to 4100 storage slots.
    pub fn normalize_source(
        tenant_id: impl Into<String>,
        model_revision: impl Into<String>,
        source: &[f32],
    ) -> Result<Self, EmbeddingError> {
        let tenant_id = tenant_id.into();
        let model_revision = model_revision.into();
        validate_identity(&tenant_id, &model_revision)?;
        if source.is_empty() {
            return Err(EmbeddingError::EmptySource);
        }
        if source.len() > EMBEDDING_SOURCE_DIMENSIONS_MAX {
            return Err(EmbeddingError::SourceTooWide);
        }
        if source.iter().any(|value| !value.is_finite()) {
            return Err(EmbeddingError::NonFiniteValue);
        }
        let squared_norm = source
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>();
        if !squared_norm.is_finite() || squared_norm <= f64::EPSILON {
            return Err(EmbeddingError::ZeroVector);
        }
        let norm = squared_norm.sqrt();
        let mut values = Vec::with_capacity(EMBEDDING_STORAGE_DIMENSIONS);
        values.extend(source.iter().map(|value| (f64::from(*value) / norm) as f32));
        values.resize(EMBEDDING_STORAGE_DIMENSIONS, 0.0);
        Ok(Self {
            tenant_id,
            model_revision,
            values,
        })
    }

    /// Validates an already-padded vector read from a trusted persistence adapter.
    pub fn from_storage(
        tenant_id: impl Into<String>,
        model_revision: impl Into<String>,
        values: Vec<f32>,
    ) -> Result<Self, EmbeddingError> {
        let tenant_id = tenant_id.into();
        let model_revision = model_revision.into();
        validate_identity(&tenant_id, &model_revision)?;
        if values.len() != EMBEDDING_STORAGE_DIMENSIONS {
            return Err(EmbeddingError::InvalidStorageDimensions);
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(EmbeddingError::NonFiniteValue);
        }
        if values[EMBEDDING_SOURCE_DIMENSIONS_MAX..]
            .iter()
            .any(|value| *value != 0.0)
        {
            return Err(EmbeddingError::NonZeroPadding);
        }
        let squared_norm = values
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>();
        if squared_norm <= f64::EPSILON {
            return Err(EmbeddingError::ZeroVector);
        }
        if (squared_norm.sqrt() - 1.0).abs() > 1e-4 {
            return Err(EmbeddingError::NotNormalized);
        }
        Ok(Self {
            tenant_id,
            model_revision,
            values,
        })
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn tenant_id(&self) -> &str {
        &self.tenant_id
    }

    pub fn model_revision(&self) -> &str {
        &self.model_revision
    }
}

fn validate_identity(tenant_id: &str, model_revision: &str) -> Result<(), EmbeddingError> {
    if !valid_opaque(tenant_id) {
        return Err(EmbeddingError::InvalidTenant);
    }
    if !valid_opaque(model_revision) {
        return Err(EmbeddingError::InvalidModelRevision);
    }
    Ok(())
}

pub fn cosine_similarity(left: &Embedding, right: &Embedding) -> Result<f64, EmbeddingError> {
    if left.tenant_id != right.tenant_id {
        return Err(EmbeddingError::TenantMismatch);
    }
    if left.model_revision != right.model_revision {
        return Err(EmbeddingError::ModelRevisionMismatch);
    }
    let similarity = left
        .values
        .iter()
        .zip(&right.values)
        .map(|(left, right)| f64::from(*left) * f64::from(*right))
        .sum::<f64>();
    Ok(similarity.clamp(-1.0, 1.0))
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchHit {
    pub candidate_id: String,
    pub similarity: f64,
}

pub fn rank_embeddings(
    query: &Embedding,
    candidates: &[(&str, &Embedding)],
    limit: usize,
) -> Result<Vec<SearchHit>, EmbeddingError> {
    if candidates.len() > SEARCH_CANDIDATES_MAX {
        return Err(EmbeddingError::TooManyCandidates);
    }
    let mut hits = Vec::with_capacity(candidates.len());
    for (candidate_id, embedding) in candidates {
        if !valid_opaque(candidate_id) {
            return Err(EmbeddingError::InvalidCandidateId);
        }
        hits.push(SearchHit {
            candidate_id: (*candidate_id).to_owned(),
            similarity: cosine_similarity(query, embedding)?,
        });
    }
    hits.sort_by(|left, right| {
        right
            .similarity
            .partial_cmp(&left.similarity)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    hits.truncate(limit.min(100));
    Ok(hits)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Observation {
    pub predictor: f64,
    pub response: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegressionFit {
    pub observations: usize,
    pub slope: f64,
    pub intercept: f64,
    pub pearson_r: f64,
    pub r_squared: f64,
    pub residual_sum_squares: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegressionError {
    TooFewObservations,
    TooManyObservations,
    NonFiniteObservation,
    ZeroPredictorVariance,
    ZeroResponseVariance,
    NonFiniteResult,
}

/// Deterministic ordinary least squares and Pearson correlation.
///
/// The result describes association only. It is not causal inference and must
/// not be used as an authorization, pricing, housing, or access-control signal.
pub fn fit_linear(observations: &[Observation]) -> Result<RegressionFit, RegressionError> {
    if observations.len() < DISCOVERY_OBSERVATIONS_MIN {
        return Err(RegressionError::TooFewObservations);
    }
    if observations.len() > DISCOVERY_OBSERVATIONS_MAX {
        return Err(RegressionError::TooManyObservations);
    }
    if observations
        .iter()
        .any(|point| !point.predictor.is_finite() || !point.response.is_finite())
    {
        return Err(RegressionError::NonFiniteObservation);
    }

    let count = observations.len() as f64;
    let mean_x = observations
        .iter()
        .map(|point| point.predictor)
        .sum::<f64>()
        / count;
    let mean_y = observations.iter().map(|point| point.response).sum::<f64>() / count;
    let (sum_xx, sum_yy, sum_xy) =
        observations
            .iter()
            .fold((0.0, 0.0, 0.0), |(sum_xx, sum_yy, sum_xy), point| {
                let centered_x = point.predictor - mean_x;
                let centered_y = point.response - mean_y;
                (
                    sum_xx + centered_x * centered_x,
                    sum_yy + centered_y * centered_y,
                    sum_xy + centered_x * centered_y,
                )
            });
    if sum_xx <= f64::EPSILON {
        return Err(RegressionError::ZeroPredictorVariance);
    }
    if sum_yy <= f64::EPSILON {
        return Err(RegressionError::ZeroResponseVariance);
    }
    let slope = sum_xy / sum_xx;
    let intercept = mean_y - slope * mean_x;
    let pearson_r = (sum_xy / (sum_xx * sum_yy).sqrt()).clamp(-1.0, 1.0);
    let r_squared = (pearson_r * pearson_r).clamp(0.0, 1.0);
    let residual_sum_squares = observations
        .iter()
        .map(|point| {
            let residual = point.response - (intercept + slope * point.predictor);
            residual * residual
        })
        .sum::<f64>();
    let fit = RegressionFit {
        observations: observations.len(),
        slope,
        intercept,
        pearson_r,
        r_squared,
        residual_sum_squares,
    };
    if [
        fit.slope,
        fit.intercept,
        fit.pearson_r,
        fit.r_squared,
        fit.residual_sum_squares,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        return Err(RegressionError::NonFiniteResult);
    }
    Ok(fit)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuditStatus {
    Allowed,
    Denied,
    Completed,
    Failed,
}

/// Content-free audit metadata safe for telemetry exporters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntelligenceAuditEvent {
    pub surface: IntelligenceSurface,
    pub status: AuditStatus,
    pub candidate_count: u16,
    pub result_count: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    const TENANT: &str = "tenant_medellin";
    const SUBJECT: &str = "subject_123";

    fn actor<'a>(roles: &'a [Role], scopes: &'a [&'a str]) -> ActorContext<'a> {
        ActorContext {
            subject: Some(SUBJECT),
            tenant_id: Some(TENANT),
            roles,
            scopes,
        }
    }

    #[test]
    fn public_surfaces_are_public_only() {
        let anonymous = ActorContext {
            subject: None,
            tenant_id: None,
            roles: &[],
            scopes: &[],
        };
        assert_eq!(
            authorize_surface(IntelligenceSurface::SalesVisitor, None, anonymous),
            Ok(SearchVisibility::Public)
        );
        assert_eq!(
            authorize_surface(IntelligenceSurface::ExternalSearch, Some(TENANT), anonymous),
            Err(PolicyError::PublicSurfaceCannotSelectTenant)
        );
    }

    #[test]
    fn customer_support_requires_trusted_tenant_and_scope() {
        assert_eq!(
            authorize_surface(
                IntelligenceSurface::CustomerSupport,
                Some(TENANT),
                actor(&[Role::Resident], &["chat:support"]),
            ),
            Ok(SearchVisibility::Customer)
        );
        assert_eq!(
            authorize_surface(
                IntelligenceSurface::CustomerSupport,
                Some("tenant_other"),
                actor(&[Role::Resident], &["chat:support"]),
            ),
            Err(PolicyError::TenantMismatch)
        );
        assert_eq!(
            authorize_surface(
                IntelligenceSurface::CustomerSupport,
                Some(TENANT),
                actor(&[Role::Resident], &[]),
            ),
            Err(PolicyError::ScopeRequired)
        );
    }

    #[test]
    fn internal_surfaces_require_role_and_scope() {
        assert_eq!(
            authorize_surface(
                IntelligenceSurface::InternalSearch,
                Some(TENANT),
                actor(&[Role::Owner], &["search:internal"]),
            ),
            Ok(SearchVisibility::Internal)
        );
        assert_eq!(
            authorize_surface(
                IntelligenceSurface::InternalSearch,
                Some(TENANT),
                actor(&[Role::Resident], &["search:internal"]),
            ),
            Err(PolicyError::PrivilegedRoleRequired)
        );
    }

    #[test]
    fn embedding_is_normalized_padded_and_redacted() {
        let embedding = Embedding::normalize_source(TENANT, "model_v1", &[3.0, 4.0]).unwrap();
        assert_eq!(embedding.values().len(), EMBEDDING_STORAGE_DIMENSIONS);
        assert!((embedding.values()[0] - 0.6).abs() < 1e-6);
        assert!((embedding.values()[1] - 0.8).abs() < 1e-6);
        assert!(embedding.values()[2..].iter().all(|value| *value == 0.0));
        let debug = format!("{embedding:?}");
        assert!(debug.contains("REDACTED"));
        assert!(!debug.contains("0.6"));
    }

    #[test]
    fn embedding_rejects_bad_inputs() {
        assert_eq!(
            Embedding::normalize_source(TENANT, "model_v1", &[]).unwrap_err(),
            EmbeddingError::EmptySource
        );
        assert_eq!(
            Embedding::normalize_source(TENANT, "model_v1", &[0.0]).unwrap_err(),
            EmbeddingError::ZeroVector
        );
        assert_eq!(
            Embedding::normalize_source(TENANT, "model_v1", &[f32::NAN]).unwrap_err(),
            EmbeddingError::NonFiniteValue
        );
        assert_eq!(
            Embedding::normalize_source(
                TENANT,
                "model_v1",
                &vec![1.0; EMBEDDING_SOURCE_DIMENSIONS_MAX + 1],
            )
            .unwrap_err(),
            EmbeddingError::SourceTooWide
        );
    }

    #[test]
    fn storage_embedding_rejects_width_norm_and_padding_drift() {
        assert_eq!(
            Embedding::from_storage(TENANT, "model_v1", vec![1.0]).unwrap_err(),
            EmbeddingError::InvalidStorageDimensions
        );
        let mut bad_norm = vec![0.0; EMBEDDING_STORAGE_DIMENSIONS];
        bad_norm[0] = 2.0;
        assert_eq!(
            Embedding::from_storage(TENANT, "model_v1", bad_norm).unwrap_err(),
            EmbeddingError::NotNormalized
        );
        let mut bad_padding = vec![0.0; EMBEDDING_STORAGE_DIMENSIONS];
        bad_padding[0] = 1.0;
        bad_padding[EMBEDDING_SOURCE_DIMENSIONS_MAX] = 0.5;
        assert_eq!(
            Embedding::from_storage(TENANT, "model_v1", bad_padding).unwrap_err(),
            EmbeddingError::NonZeroPadding
        );
    }

    #[test]
    fn similarity_fails_closed_across_tenants_and_models() {
        let query = Embedding::normalize_source(TENANT, "model_v1", &[1.0, 0.0]).unwrap();
        let other_tenant =
            Embedding::normalize_source("tenant_other", "model_v1", &[1.0, 0.0]).unwrap();
        let other_model = Embedding::normalize_source(TENANT, "model_v2", &[1.0, 0.0]).unwrap();
        assert_eq!(
            cosine_similarity(&query, &other_tenant),
            Err(EmbeddingError::TenantMismatch)
        );
        assert_eq!(
            cosine_similarity(&query, &other_model),
            Err(EmbeddingError::ModelRevisionMismatch)
        );
    }

    #[test]
    fn ranking_is_stable_with_id_tiebreak() {
        let query = Embedding::normalize_source(TENANT, "model_v1", &[1.0, 0.0]).unwrap();
        let same = Embedding::normalize_source(TENANT, "model_v1", &[1.0, 0.0]).unwrap();
        let orthogonal = Embedding::normalize_source(TENANT, "model_v1", &[0.0, 1.0]).unwrap();
        let hits = rank_embeddings(
            &query,
            &[
                ("z_candidate", &same),
                ("a_candidate", &same),
                ("middle", &orthogonal),
            ],
            3,
        )
        .unwrap();
        assert_eq!(hits[0].candidate_id, "a_candidate");
        assert_eq!(hits[1].candidate_id, "z_candidate");
        assert_eq!(hits[2].candidate_id, "middle");
    }

    #[test]
    fn regression_recovers_known_line_and_correlation() {
        let points = (0..10)
            .map(|value| Observation {
                predictor: f64::from(value),
                response: 2.0 * f64::from(value) + 3.0,
            })
            .collect::<Vec<_>>();
        let fit = fit_linear(&points).unwrap();
        assert!((fit.slope - 2.0).abs() < 1e-12);
        assert!((fit.intercept - 3.0).abs() < 1e-12);
        assert!((fit.pearson_r - 1.0).abs() < 1e-12);
        assert!((fit.r_squared - 1.0).abs() < 1e-12);
        assert!(fit.residual_sum_squares < 1e-20);
    }

    #[test]
    fn regression_rejects_invalid_cohorts() {
        assert_eq!(
            fit_linear(
                &[Observation {
                    predictor: 1.0,
                    response: 2.0,
                }; 9]
            ),
            Err(RegressionError::TooFewObservations)
        );
        assert_eq!(
            fit_linear(
                &[Observation {
                    predictor: 1.0,
                    response: 2.0,
                }; 10]
            ),
            Err(RegressionError::ZeroPredictorVariance)
        );
        let mut non_finite = [Observation {
            predictor: 1.0,
            response: 2.0,
        }; 10];
        non_finite[0].response = f64::INFINITY;
        assert_eq!(
            fit_linear(&non_finite),
            Err(RegressionError::NonFiniteObservation)
        );
    }
}
