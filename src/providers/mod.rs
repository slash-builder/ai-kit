//! Real provider adapters. Each is behind its own Cargo feature and talks to its vendor's API over HTTPS, so none is compiled by default.
//!
//! - [`anthropic`] (`provider-anthropic`): the Claude Messages API.
//! - [`gemini`] (`provider-gemini`): Google Gemini (`generateContent`, `embedContent`).
//! - [`openai_compat`] (`provider-openai-compat`): any OpenAI-compatible server (Ollama, llama.cpp, a gateway).
//!
//! Every adapter follows the same rules: the credential is a *name* resolved through a [`crate::credentials::CredentialSource`] on each
//! call; errors are mapped to a [`crate::provider::ProviderError`] class and never carry an upstream body or header; a call that would not fit
//! the [`crate::budget::Ledger`] is refused before anything is sent.

use crate::budget::{Ledger, Pricing};
use std::fmt;
use std::sync::Arc;

/// Why an adapter could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConfigError {
    /// `Locality::Device` but the base URL is not a loopback address.
    DeviceMustBeLoopback,
    /// `Locality::Home` but the host is a public address.
    HomeMustBePrivate,
    /// `Locality::Saas` needs a credential name and a budget.
    SaasNeedsCredentialAndBudget,
    /// A credential name was configured but cannot be loaded now.
    CredentialUnavailable,
    BadUrl,
    HttpClient,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ConfigError::DeviceMustBeLoopback => "a Device provider must be on a loopback address",
            ConfigError::HomeMustBePrivate => "a Home provider must be on a private address",
            ConfigError::SaasNeedsCredentialAndBudget => {
                "a SaaS provider needs a credential name and a budget"
            }
            ConfigError::CredentialUnavailable => "the configured credential cannot be loaded",
            ConfigError::BadUrl => "the base URL is not valid",
            ConfigError::HttpClient => "the HTTP client could not be built",
        })
    }
}
impl std::error::Error for ConfigError {}

/// The spend guard a SaaS adapter must carry.
#[derive(Clone)]
pub struct Budget {
    pub ledger: Arc<Ledger>,
    pub pricing: Pricing,
}

#[cfg(feature = "provider-anthropic")]
pub mod anthropic;
#[cfg(feature = "provider-gemini")]
pub mod gemini;
#[cfg(feature = "provider-openai-compat")]
pub mod openai_compat;
