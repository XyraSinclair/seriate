#![forbid(unsafe_code)]

//! # seriate
//!
//! Provenanced attribute annotation and LLM prior elicitation over entity
//! sets. The unit is the structured, provenanced judgement: an immutable,
//! content-addressed evidence record traceable to raw provider bytes. The
//! object elicited is the LLM's prior over the orderings of an entity set.
//!
//! Invariants:
//! 1. Nothing fabricates a number without a judgement-record ancestor.
//! 2. Ordinal evidence suffices; ratio magnitudes are an upgrade.
//! 3. Logprobs are harnessed where real and degradation is loud where not:
//!    every PMF carries its [`evidence::PmfCompleteness`].

pub mod atom;
pub mod capture;
pub mod compile;
pub mod evidence;
pub mod gateway;
pub mod instrument;
#[cfg(feature = "sqlite")]
pub mod log;
pub mod ontology;
pub mod probe;
pub mod record;

pub use atom::{interpolate_ratio, AnswerAtom, Side, RATIO_LADDER};
pub use capture::ProviderCapture;
pub use compile::{compile, CompileError, CompiledPosterior, EntityPosterior, Tolerances};
pub use evidence::{
    evidence_from_logprobs, evidence_from_resamples, fused_evidence, jsd, AnswerEvidence,
    AtomLogprob, AtomProb, EvidenceError, PmfCompleteness,
};
pub use gateway::{ChatOutcome, ChatSpec, Gateway, GatewayError, TokenLogprob, Usage};
#[cfg(feature = "sqlite")]
pub use log::{EvidenceLog, LogError, ProvenanceChain};
pub use ontology::{
    Attribute, AttributeId, CaptureId, ContentId, Entity, EntityId, JudgementId, PairKey,
    Presentation, TemplateHash,
};
pub use record::{
    AcquisitionMode, Cost, DecodeConfig, EvidenceHealth, InstrumentKind, JudgementRecord,
    ParserVersion,
};
