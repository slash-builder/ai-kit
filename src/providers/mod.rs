//! Real provider adapters. Each is behind its own Cargo feature and talks to its vendor's API over HTTPS, so none is compiled by default.
//!
//! - [`anthropic`] (`provider-anthropic`): the Claude Messages API.
//! - [`openai_compat`] (`provider-openai-compat`): any OpenAI-compatible server (Ollama, llama.cpp, a gateway).
//!
//! Every adapter follows the same rules: the credential is a *name* resolved through a [`crate::credentials::CredentialSource`] on each
//! call; errors are mapped to a [`crate::provider::ProviderError`] class and never carry an upstream body or header; a call that would not fit
//! the [`crate::budget::Ledger`] is refused before anything is sent.

#[cfg(feature = "provider-anthropic")]
pub mod anthropic;
#[cfg(feature = "provider-openai-compat")]
pub mod openai_compat;
