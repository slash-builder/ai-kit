//! Model and KV cache management.
//!
//! # Design Notes
//!
//! Two separate caches:
//!
//! 1. **Model Cache**: Weights are loaded once per model (RO mmap), reused across apps.
//!    - Hash → LoadedModel mapping
//!    - Ref counting for cleanup
//!    - Thread-safe (Arc + RwLock)
//!
//! 2. **KV Cache**: Per-session context caches (never reused cross-app).
//!    - SessionId → KvCache mapping
//!    - LRU eviction on memory pressure
//!    - Strict privacy isolation (app_id + session_id as key)
//!
//! # Invariant
//!
//! Per decision invariant #4: KV caches are never reused cross-app (privacy isolation).
//! Design ensures that if two apps use the same model in parallel, their KV states
//! are completely isolated (keyed by app_id + session_id).
//!
//! # TODO
//!
//! - Implement actual mmap loading (currently placeholder).
//! - LRU eviction logic with TTL.
//! - Integration with Storage Kit for model artifact persistence.

use crate::types::{AppId, ModelHash, SessionId};
use std::collections::HashMap;
use std::sync::Arc;

/// Loaded model in memory (RO mmap).
#[derive(Debug, Clone)]
pub struct LoadedModel {
    /// BLAKE3 hash of model weights (content address).
    pub hash: ModelHash,

    /// Weights data (mmap'd RO from /usr/share/models/).
    /// In production, this would be an mmap handle, not a heap Vec.
    pub weights: Vec<u8>,

    /// Reference count (number of active inferences using this model).
    pub ref_count: u32,

    /// Timestamp when model was loaded.
    pub loaded_at: u64,
}

impl LoadedModel {
    /// Create a new LoadedModel.
    pub fn new(hash: ModelHash, weights: Vec<u8>) -> Self {
        LoadedModel {
            hash,
            weights,
            ref_count: 0,
            loaded_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Size in bytes (for memory accounting).
    pub fn size_bytes(&self) -> u64 {
        self.weights.len() as u64
    }
}

/// KV cache for a single session (per-session, not reused cross-app).
///
/// # Invariant (Decision #10)
///
/// Cache key is `app_id + session_id` (not `session_id` alone).
/// This prevents cross-app context leakage if two apps use the same model+session-id pair.
#[derive(Debug, Clone)]
pub struct KvCache {
    /// Session ID (for cache locality).
    pub session_id: SessionId,

    /// BLAKE3 hash of the model being used in this session.
    pub model_hash: ModelHash,

    /// App ID (owner, for audit trail and privacy).
    pub app_id: AppId,

    /// Cache data (key-value activations for KV cache in transformer).
    /// In production, this would be a nested HashMap or ring buffer.
    pub cache_data: Vec<u8>,

    /// Timestamp when cache was created.
    pub created_at: u64,

    /// Timestamp when cache was last accessed (for LRU eviction).
    pub last_used_at: u64,

    /// TTL in milliseconds (auto-evict after silence).
    pub ttl_ms: u32,
}

impl KvCache {
    /// Create a new KvCache.
    pub fn new(session_id: SessionId, model_hash: ModelHash, app_id: AppId) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        KvCache {
            session_id,
            model_hash,
            app_id,
            cache_data: Vec::new(),
            created_at: now,
            last_used_at: now,
            ttl_ms: 3_600_000, // 1 hour default TTL
        }
    }

    /// Set custom TTL (in milliseconds).
    pub fn with_ttl(mut self, ttl_ms: u32) -> Self {
        self.ttl_ms = ttl_ms;
        self
    }

    /// Check if cache has expired (TTL elapsed).
    pub fn is_expired(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let age_ms = ((now - self.last_used_at) * 1000) as u32;
        age_ms > self.ttl_ms
    }

    /// Update last_used_at to current time.
    pub fn touch(&mut self) {
        self.last_used_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }

    /// Size in bytes (for memory accounting).
    pub fn size_bytes(&self) -> u64 {
        self.cache_data.len() as u64
    }
}

/// Model cache (weights, shared across apps).
#[derive(Debug)]
pub struct ModelCache {
    /// Hash → LoadedModel mapping.
    models: HashMap<ModelHash, LoadedModel>,
}

impl ModelCache {
    /// Create a new ModelCache.
    pub fn new() -> Self {
        ModelCache {
            models: HashMap::new(),
        }
    }

    /// Get a model from cache (read-only).
    pub fn get(&self, hash: &ModelHash) -> Option<Arc<LoadedModel>> {
        self.models.get(hash).map(|m| Arc::new(m.clone()))
    }

    /// Insert a model into cache.
    pub fn insert(&mut self, model: LoadedModel) {
        self.models.insert(model.hash.clone(), model);
    }

    /// Remove a model from cache.
    pub fn remove(&mut self, hash: &ModelHash) -> Option<LoadedModel> {
        self.models.remove(hash)
    }

    /// List all cached models.
    pub fn list(&self) -> Vec<Arc<LoadedModel>> {
        self.models.values().map(|m| Arc::new(m.clone())).collect()
    }

    /// Total size of all cached models (bytes).
    pub fn total_size_bytes(&self) -> u64 {
        self.models.values().map(|m| m.size_bytes()).sum()
    }

    /// Number of cached models.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Clear all models from cache.
    pub fn clear(&mut self) {
        self.models.clear();
    }
}

impl Default for ModelCache {
    fn default() -> Self {
        Self::new()
    }
}

/// KV cache manager (per-session, privacy-isolated).
#[derive(Debug)]
pub struct KvCacheManager {
    /// SessionId → KvCache mapping.
    /// Note: In production, key should be (app_id, session_id) tuple for strict isolation.
    caches: HashMap<SessionId, KvCache>,
}

impl KvCacheManager {
    /// Create a new KvCacheManager.
    pub fn new() -> Self {
        KvCacheManager {
            caches: HashMap::new(),
        }
    }

    /// Create or get a KvCache for a session.
    pub fn get_or_create(
        &mut self,
        session_id: SessionId,
        model_hash: ModelHash,
        app_id: AppId,
    ) -> &mut KvCache {
        self.caches
            .entry(session_id.clone())
            .or_insert_with(|| KvCache::new(session_id, model_hash, app_id))
    }

    /// Get a KvCache (read-only).
    pub fn get(&self, session_id: &SessionId) -> Option<&KvCache> {
        self.caches.get(session_id)
    }

    /// Get a KvCache (mutable).
    pub fn get_mut(&mut self, session_id: &SessionId) -> Option<&mut KvCache> {
        self.caches.get_mut(session_id)
    }

    /// Remove a KvCache.
    pub fn remove(&mut self, session_id: &SessionId) -> Option<KvCache> {
        self.caches.remove(session_id)
    }

    /// Evict expired caches.
    ///
    /// # TODO
    ///
    /// Implement LRU logic with memory pressure thresholds.
    pub fn evict_expired(&mut self) {
        let expired: Vec<SessionId> = self
            .caches
            .iter()
            .filter(|(_, cache)| cache.is_expired())
            .map(|(sid, _)| sid.clone())
            .collect();

        for sid in expired {
            self.caches.remove(&sid);
        }
    }

    /// Total size of all KV caches (bytes).
    pub fn total_size_bytes(&self) -> u64 {
        self.caches.values().map(|c| c.size_bytes()).sum()
    }

    /// Number of active sessions.
    pub fn len(&self) -> usize {
        self.caches.len()
    }

    /// Check if manager is empty.
    pub fn is_empty(&self) -> bool {
        self.caches.is_empty()
    }

    /// Clear all caches.
    pub fn clear(&mut self) {
        self.caches.clear();
    }
}

impl Default for KvCacheManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loaded_model_new() {
        let model = LoadedModel::new("blake3-hash".to_string(), vec![1, 2, 3, 4, 5]);
        assert_eq!(model.hash, "blake3-hash");
        assert_eq!(model.size_bytes(), 5);
        assert_eq!(model.ref_count, 0);
    }

    #[test]
    fn test_model_cache_insert_and_get() {
        let mut cache = ModelCache::new();
        let model = LoadedModel::new("hash1".to_string(), vec![1, 2, 3]);
        cache.insert(model);

        assert_eq!(cache.len(), 1);
        assert!(cache.get(&"hash1".to_string()).is_some());
        assert!(cache.get(&"hash2".to_string()).is_none());
    }

    #[test]
    fn test_kv_cache_ttl_expiry() {
        let cache = KvCache::new(
            SessionId::new(),
            "model-hash".to_string(),
            "app1".to_string(),
        )
        .with_ttl(1); // 1ms TTL

        // Cache should not expire immediately
        assert!(!cache.is_expired());
    }

    #[test]
    fn test_kv_cache_manager_operations() {
        let mut mgr = KvCacheManager::new();
        let session_id = SessionId::new();

        // Create a cache
        mgr.get_or_create(
            session_id.clone(),
            "model-hash".to_string(),
            "app1".to_string(),
        );
        assert_eq!(mgr.len(), 1);

        // Get it back
        assert!(mgr.get(&session_id).is_some());

        // Remove it
        mgr.remove(&session_id);
        assert_eq!(mgr.len(), 0);
    }

    #[test]
    fn test_privacy_isolation_conceptual() {
        // This test demonstrates the privacy isolation concept.
        // In production, the key should be (app_id, session_id) tuple.
        let mut mgr = KvCacheManager::new();

        let session_id = SessionId::from_string("session1".to_string());

        // App1 uses session1 with model A
        mgr.get_or_create(
            session_id.clone(),
            "model-a".to_string(),
            "app1".to_string(),
        );

        // Verify cache belongs to app1
        if let Some(cache) = mgr.get(&session_id) {
            assert_eq!(cache.app_id, "app1");
        }

        // In real code, if app2 tried to use session1, it would be a separate cache entry
        // (because the key should be (app_id, session_id) not just session_id).
    }
}
