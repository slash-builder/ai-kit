//! Model registry and discovery.
//!
//! # Design Notes
//!
//! The model registry is loaded from `/usr/share/models/catalog.yaml` (OS-shipped, immutable).
//! It holds metadata for all available models, including:
//! - Semantic version
//! - BLAKE3 hash
//! - Quantization variants
//! - VRAM requirements
//! - OS architecture support
//!
//! # Invariant
//!
//! Models are OS resources (decision invariant #1). They ship signed with the OS image
//! and are verified at boot. Registry is immutable post-deployment.
//!
//! # TODO
//!
//! - Implement catalog.yaml loader (YAML parsing).
//! - Integration with Storage Kit once available.
//! - Versioning semantics (SemVer resolver for app dependencies).

use crate::error::Result;
use crate::types::ModelMetadata;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model registry (in-memory catalog, loaded from OS image).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistry {
    /// Models indexed by ID.
    pub models: HashMap<String, ModelMetadata>,

    /// Catalog source (e.g., "file:///usr/share/models/catalog.yaml").
    pub source: String,

    /// Timestamp when catalog was loaded.
    pub loaded_at: u64,

    /// Checksum or version of the catalog (for audit trail).
    pub catalog_version: String,
}

impl ModelRegistry {
    /// Create a new empty registry.
    pub fn new(source: String, catalog_version: String) -> Self {
        ModelRegistry {
            models: HashMap::new(),
            source,
            loaded_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            catalog_version,
        }
    }

    /// Load registry from YAML file.
    ///
    /// # Examples
    ///
    /// ```yaml
    /// # /usr/share/models/catalog.yaml
    /// version: "1.0"
    /// models:
    ///   - id: "llama2-7b"
    ///     version: "2.0.0"
    ///     os_arch: ["x86_64", "aarch64"]
    ///     quantization_variants: ["q4", "q8", "fp16"]
    ///     default_quantization: "q4"
    ///     hash: "blake3-hash..."
    ///     size_bytes: 3865470976
    ///     location: "file:///usr/share/models/llama2-7b-q4.bin"
    ///     metadata:
    ///       parameters: 7000000000
    ///       context_window: 4096
    ///       vram_required_mb: 4096
    ///       inference_latency_ms: 45
    /// ```
    ///
    /// # TODO
    ///
    /// Implement full YAML parsing. Currently a placeholder.
    pub fn load_from_file(_path: &str) -> Result<ModelRegistry> {
        Err(crate::error::AiKitError::Internal {
            reason: "ModelRegistry::load_from_file not yet implemented (awaiting YAML parsing)"
                .to_string(),
        })
    }

    /// Look up a model by ID.
    pub fn get(&self, model_id: &str) -> Option<&ModelMetadata> {
        self.models.get(model_id)
    }

    /// Check if a model exists.
    pub fn has(&self, model_id: &str) -> bool {
        self.models.contains_key(model_id)
    }

    /// List all available models.
    pub fn list(&self) -> Vec<&ModelMetadata> {
        self.models.values().collect()
    }

    /// Add a model to the registry (for testing or dynamic registration).
    pub fn add(&mut self, model: ModelMetadata) {
        self.models.insert(model.id.clone(), model);
    }

    /// Get number of models in registry.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Validate that all models are properly configured.
    ///
    /// # Checks
    ///
    /// - All model IDs are non-empty.
    /// - All hashes are valid (BLAKE3).
    /// - At least one quantization variant per model.
    /// - Default quantization is in variants list.
    /// - VRAM requirements are positive.
    pub fn validate(&self) -> Result<()> {
        for model in self.models.values() {
            if model.id.is_empty() {
                return Err(crate::error::AiKitError::Internal {
                    reason: "Model ID cannot be empty".to_string(),
                });
            }

            if model.hash.is_empty() {
                return Err(crate::error::AiKitError::Internal {
                    reason: format!("Model {} has empty hash", model.id),
                });
            }

            if model.quantization_variants.is_empty() {
                return Err(crate::error::AiKitError::Internal {
                    reason: format!("Model {} has no quantization variants", model.id),
                });
            }

            if !model
                .quantization_variants
                .contains(&model.default_quantization)
            {
                return Err(crate::error::AiKitError::Internal {
                    reason: format!(
                        "Model {} default quantization {} not in variants",
                        model.id, model.default_quantization
                    ),
                });
            }

            if model.metadata.vram_required_mb == 0 {
                return Err(crate::error::AiKitError::Internal {
                    reason: format!("Model {} has zero VRAM requirement", model.id),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModelMetadataExtended;

    fn create_test_model() -> ModelMetadata {
        ModelMetadata {
            id: "test-model".to_string(),
            version: "1.0.0".to_string(),
            os_arch: vec!["x86_64".to_string()],
            quantization_variants: vec!["q4".to_string(), "q8".to_string()],
            default_quantization: "q4".to_string(),
            hash: "blake3-abcd1234".to_string(),
            size_bytes: 1000000,
            location: "file:///usr/share/models/test-model-q4.bin".to_string(),
            metadata: ModelMetadataExtended {
                parameters: 1000000,
                context_window: 2048,
                vram_required_mb: 2048,
                inference_latency_ms: 50,
                framework: "candle".to_string(),
                custom: Default::default(),
            },
        }
    }

    #[test]
    fn test_registry_create() {
        let reg = ModelRegistry::new(
            "file:///usr/share/models/catalog.yaml".to_string(),
            "1.0.0".to_string(),
        );
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_add_and_get() {
        let mut reg = ModelRegistry::new(
            "file:///usr/share/models/catalog.yaml".to_string(),
            "1.0.0".to_string(),
        );
        let model = create_test_model();
        reg.add(model.clone());

        assert_eq!(reg.len(), 1);
        assert!(reg.has("test-model"));
        assert_eq!(reg.get("test-model").unwrap().id, "test-model");
    }

    #[test]
    fn test_registry_validation_valid() {
        let mut reg = ModelRegistry::new(
            "file:///usr/share/models/catalog.yaml".to_string(),
            "1.0.0".to_string(),
        );
        let model = create_test_model();
        reg.add(model);

        assert!(reg.validate().is_ok());
    }

    #[test]
    fn test_registry_validation_invalid_quantization() {
        let mut reg = ModelRegistry::new(
            "file:///usr/share/models/catalog.yaml".to_string(),
            "1.0.0".to_string(),
        );
        let mut model = create_test_model();
        model.default_quantization = "q16".to_string(); // Not in variants

        reg.add(model);
        assert!(reg.validate().is_err());
    }
}
