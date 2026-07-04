//! Raw-bytes provenance anchor: the untouched provider response body,
//! content-addressed, with the request that produced it fingerprinted
//! alongside it.
//!
//! Every [`crate::record::JudgementRecord`] points at a
//! [`ProviderCapture`] by [`CaptureId`](crate::ontology::CaptureId), and the
//! id is a hash of the exact bytes that came back over the wire — not a
//! re-serialization, not a parsed structure. If a provider byte flips, the
//! id changes; nothing downstream can silently drift from what was actually
//! said.

use crate::ontology::{CaptureId, ContentId};
use serde::{Deserialize, Serialize};

/// One raw provider response, anchored by the exact bytes of its body.
///
/// Constructed only via [`ProviderCapture::new`], which derives `id` from
/// `raw`. Fields are public for read access and serde, but the id is never
/// meant to be hand-set — use [`ProviderCapture::verify`] to catch drift
/// (deserialized-then-tampered records, hand-edited fixtures, etc.).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProviderCapture {
    /// Content id over the exact bytes of `raw`.
    pub id: CaptureId,
    /// The untouched provider response body, byte for byte.
    pub raw: String,
    /// Content id over the serialized request that produced this response,
    /// WITHOUT the API key (the request is provenance, the key is a secret).
    pub request_fingerprint: ContentId,
    /// Model identifier as sent to the provider (e.g. `"openai/gpt-4.1-mini"`).
    pub model: String,
    /// The request path the call was made against (e.g. `"/chat/completions"`).
    pub url_path: String,
    /// Unix millis at capture time.
    pub created_at_ms: u64,
}

impl ProviderCapture {
    /// Build a capture, deriving `id` from `raw`'s exact bytes.
    pub fn new(
        raw: impl Into<String>,
        request_fingerprint: ContentId,
        model: impl Into<String>,
        url_path: impl Into<String>,
        created_at_ms: u64,
    ) -> Self {
        let raw = raw.into();
        let id = CaptureId::derive(raw.as_bytes());
        Self {
            id,
            raw,
            request_fingerprint,
            model: model.into(),
            url_path: url_path.into(),
            created_at_ms,
        }
    }

    /// Recompute the id from `raw`; must equal `self.id` for an untampered
    /// capture (matches the pattern of
    /// [`JudgementRecord::verify_id`](crate::record::JudgementRecord::verify_id)).
    pub fn verify(&self) -> bool {
        CaptureId::derive(self.raw.as_bytes()) == self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fingerprint(bytes: &[u8]) -> ContentId {
        ContentId::derive("seriate/gateway-request", bytes)
    }

    #[test]
    fn id_is_derived_from_raw_bytes_and_stable() {
        let c1 = ProviderCapture::new(
            r#"{"choices":[]}"#,
            fingerprint(b"req-a"),
            "openai/gpt-4.1-mini",
            "/chat/completions",
            1_700_000_000_000,
        );
        let c2 = ProviderCapture::new(
            r#"{"choices":[]}"#,
            // Different fingerprint and model, same raw bytes: same id.
            fingerprint(b"req-b"),
            "anthropic/claude",
            "/chat/completions",
            1_700_000_000_001,
        );
        assert_eq!(c1.id, c2.id, "id is a function of raw bytes only");
        assert!(c1.verify());
        assert!(c2.verify());
    }

    #[test]
    fn different_raw_bytes_yield_different_ids() {
        let a = ProviderCapture::new(
            "byte-exact-response-a",
            fingerprint(b"req"),
            "m",
            "/chat/completions",
            0,
        );
        let b = ProviderCapture::new(
            "byte-exact-response-b",
            fingerprint(b"req"),
            "m",
            "/chat/completions",
            0,
        );
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn tampering_with_raw_breaks_verification() {
        let mut c = ProviderCapture::new(
            "original body",
            fingerprint(b"req"),
            "m",
            "/chat/completions",
            0,
        );
        assert!(c.verify());
        c.raw = "tampered body".into();
        assert!(!c.verify(), "id must not follow tampered raw bytes");
    }

    #[test]
    fn serde_round_trip_preserves_id_and_fields() {
        let c = ProviderCapture::new(
            r#"{"ok":true}"#,
            fingerprint(b"req-payload"),
            "openrouter/x-ai/grok-4.20",
            "/chat/completions",
            1_700_000_123_456,
        );
        let json = serde_json::to_string(&c).expect("serializes");
        let back: ProviderCapture = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(back, c);
        assert!(back.verify());
    }
}
