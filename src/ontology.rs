//! Entities, attributes, presentations: the things judgements are ABOUT.
//!
//! An entity set admits many meaningful orderings; each ordering corresponds
//! to an attribute. Both are first-class, content-addressed values — never
//! bare strings — so every judgement pins exactly what was compared, on
//! what, in which presented order.

use serde::{Deserialize, Serialize};

/// Content-addressed identifier: blake3 of a domain-tagged canonical
/// serialization, hex-encoded. The domain tag prevents cross-type collisions.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ContentId(pub String);

impl ContentId {
    pub fn derive(domain: &str, bytes: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(domain.as_bytes());
        hasher.update(&[0x1f]);
        hasher.update(bytes);
        Self(hasher.finalize().to_hex().to_string())
    }

    pub fn short(&self) -> &str {
        &self.0[..12.min(self.0.len())]
    }
}

macro_rules! content_id_newtype {
    ($(#[$doc:meta])* $name:ident, $domain:literal) => {
        $(#[$doc])*
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name(pub ContentId);

        impl $name {
            pub fn derive(bytes: &[u8]) -> Self {
                Self(ContentId::derive($domain, bytes))
            }
            pub fn short(&self) -> &str {
                self.0.short()
            }
        }
    };
}

content_id_newtype!(
    /// Identifies an entity by its content.
    EntityId, "seriate/entity"
);
content_id_newtype!(
    /// Identifies an attribute by its full text.
    AttributeId, "seriate/attribute"
);
content_id_newtype!(
    /// Identifies a rendered prompt template (system + user skeleton).
    TemplateHash, "seriate/template"
);
content_id_newtype!(
    /// Identifies a raw provider capture by its bytes.
    CaptureId, "seriate/capture"
);
content_id_newtype!(
    /// Identifies one immutable judgement record.
    JudgementId, "seriate/judgement"
);

/// A thing that can be judged. `body` is the text shown to the judge.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entity {
    pub id: EntityId,
    pub body: String,
    /// Optional caller-facing label (never shown to the judge).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl Entity {
    pub fn new(body: impl Into<String>) -> Self {
        let body = body.into();
        Self {
            id: EntityId::derive(body.as_bytes()),
            body,
            label: None,
        }
    }

    pub fn labeled(body: impl Into<String>, label: impl Into<String>) -> Self {
        let mut e = Self::new(body);
        e.label = Some(label.into());
        e
    }
}

/// One way an entity set can be ordered.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attribute {
    pub id: AttributeId,
    /// Short handle, e.g. "rawness".
    pub name: String,
    /// The full judging text (rubric); used verbatim in prompts and hashed
    /// into the id, so a reworded attribute is a different attribute.
    pub text: String,
}

impl Attribute {
    pub fn new(name: impl Into<String>, text: impl Into<String>) -> Self {
        let name = name.into();
        let text = text.into();
        let mut key = Vec::with_capacity(name.len() + text.len() + 1);
        key.extend_from_slice(name.as_bytes());
        key.push(0x1f);
        key.extend_from_slice(text.as_bytes());
        Self {
            id: AttributeId::derive(&key),
            name,
            text,
        }
    }
}

/// Canonical unordered pair of entities (lexicographic by id).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PairKey {
    pub lo: EntityId,
    pub hi: EntityId,
}

impl PairKey {
    pub fn new(a: &EntityId, b: &EntityId) -> Self {
        if a <= b {
            Self {
                lo: a.clone(),
                hi: b.clone(),
            }
        } else {
            Self {
                lo: b.clone(),
                hi: a.clone(),
            }
        }
    }
}

/// Which entity was presented in which slot. Judgements are stored in
/// PRESENTED coordinates alongside their presentation, so canonical-pair
/// coordinates are always recoverable and counterbalancing stays auditable.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Presentation {
    pub slot_a: EntityId,
    pub slot_b: EntityId,
}

impl Presentation {
    /// True when slot order equals canonical (lo, hi) order.
    pub fn is_canonical(&self, pair: &PairKey) -> bool {
        self.slot_a == pair.lo && self.slot_b == pair.hi
    }

    pub fn pair_key(&self) -> PairKey {
        PairKey::new(&self.slot_a, &self.slot_b)
    }

    pub fn swapped(&self) -> Self {
        Self {
            slot_a: self.slot_b.clone(),
            slot_b: self.slot_a.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_ids_are_deterministic_and_domain_separated() {
        let e1 = Entity::new("hello");
        let e2 = Entity::new("hello");
        assert_eq!(e1.id, e2.id);
        // Same bytes, different domain -> different id.
        let a = Attribute::new("hello", "");
        assert_ne!(e1.id.0 .0, a.id.0 .0);
    }

    #[test]
    fn attribute_identity_is_the_full_text() {
        let a = Attribute::new("rawness", "how raw and unguarded the writing is");
        let b = Attribute::new("rawness", "how raw and unguarded the writing is.");
        assert_ne!(
            a.id, b.id,
            "one character of rubric drift is a new attribute"
        );
    }

    #[test]
    fn pair_key_is_order_invariant() {
        let x = Entity::new("x");
        let y = Entity::new("y");
        assert_eq!(PairKey::new(&x.id, &y.id), PairKey::new(&y.id, &x.id));
    }

    #[test]
    fn presentation_swap_round_trips() {
        let x = Entity::new("x");
        let y = Entity::new("y");
        let p = Presentation {
            slot_a: x.id.clone(),
            slot_b: y.id.clone(),
        };
        assert_eq!(p.swapped().swapped(), p);
        assert_eq!(p.pair_key(), p.swapped().pair_key());
        assert_ne!(
            p.is_canonical(&p.pair_key()),
            p.swapped().is_canonical(&p.pair_key())
        );
    }
}
