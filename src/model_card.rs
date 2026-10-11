//! What a provider says a model can do. **Unknown means false.**
//!
//! A [`ModelCard`] is built from what the provider's own model list reports (or from what the owner declares), with every capability
//! a three-valued [`Cap`]: a flag the provider did not report is `Unknown`, and the router treats `Unknown` as *not available*. Cards
//! record where each fact came from and when, because model lists drift and a stale probe must not look current.

use crate::modality::RequestKind;
use serde::{Deserialize, Serialize};

/// A capability the provider reported, denied, or said nothing about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cap {
    #[default]
    Unknown,
    Yes,
    No,
}

impl Cap {
    pub fn from_flag(flag: Option<bool>) -> Self {
        match flag {
            Some(true) => Cap::Yes,
            Some(false) => Cap::No,
            None => Cap::Unknown,
        }
    }
    /// Unknown means false.
    pub fn is_yes(self) -> bool {
        self == Cap::Yes
    }
}

/// Where a card's facts came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CardSource {
    /// Written by the owner / the catalog.
    Declared,
    /// Read from the provider's model list on this date (`YYYY-MM-DD`).
    Probed { on: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCard {
    pub provider_id: String,
    pub model_id: String,
    pub kinds: Vec<RequestKind>,
    pub max_input_tokens: Option<u64>,
    pub max_output_tokens: Option<u64>,
    /// When the provider says the model is retired (a date string as reported), if it does.
    pub retires_at: Option<String>,
    pub image_input: Cap,
    pub structured_outputs: Cap,
    pub tool_use: Cap,
    pub source: CardSource,
}

impl ModelCard {
    /// Can this model take `input_tokens` of input? Unknown limit means no.
    pub fn fits_input(&self, input_tokens: u64) -> bool {
        self.max_input_tokens.is_some_and(|m| input_tokens <= m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_means_false() {
        assert!(!Cap::from_flag(None).is_yes());
        assert!(!Cap::from_flag(Some(false)).is_yes());
        assert!(Cap::from_flag(Some(true)).is_yes());
        let c = ModelCard {
            provider_id: "p".into(),
            model_id: "m".into(),
            kinds: vec![RequestKind::TextGeneration],
            max_input_tokens: None,
            max_output_tokens: None,
            retires_at: None,
            image_input: Cap::Unknown,
            structured_outputs: Cap::Unknown,
            tool_use: Cap::Unknown,
            source: CardSource::Declared,
        };
        assert!(
            !c.fits_input(1),
            "an unknown context limit is not assumed to be big enough"
        );
    }
}
