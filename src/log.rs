//! The evidence log: an append-only SQLite anchor for everything the system
//! has ever recorded.
//!
//! No row is ever rewritten and no row is ever removed — the store only
//! grows via `CREATE` (schema, once) and `INSERT` (rows, forever). Content
//! ids make this safe: inserting a value already present is a no-op,
//! inserting a *different* value under an id that's supposed to identify it
//! is a hard, fail-closed error rather than silent drift. A dedicated test
//! at the bottom of this module reads these very source bytes back and
//! confirms no mutating or row-removing SQL statement is spelled out
//! anywhere in it.
//!
//! Four tables, one row shape each:
//! - `entities`   — content-addressed [`Entity`] values.
//! - `attributes` — content-addressed [`Attribute`] values.
//! - `captures`   — content-addressed [`ProviderCapture`] values (raw
//!   provider bytes plus the request metadata that produced them).
//! - `judgements` — content-addressed [`JudgementRecord`] values, stored
//!   whole as JSON with a handful of columns denormalized out for lookup
//!   (`attribute_id`, `pair_lo`, `pair_hi`) and one foreign key enforced
//!   (`capture_id`, because a judgement with no backing capture is a
//!   fabricated number by definition).
//!
//! [`EvidenceLog::provenance`] walks a judgement id all the way back to the
//! entities, attribute, and raw capture that produced it, failing closed the
//! moment any link is missing.

use crate::capture::ProviderCapture;
use crate::ontology::{Attribute, AttributeId, CaptureId, ContentId, Entity, EntityId};
use crate::record::JudgementRecord;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use thiserror::Error;

/// Schema for a fresh log. Every statement is `CREATE` or (at query time)
/// `INSERT`/`SELECT` — nothing here or anywhere below ever rewrites or
/// removes a row.
const SCHEMA_SQL: &str = "
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS entities (
    id    TEXT PRIMARY KEY,
    body  TEXT NOT NULL,
    label TEXT
);

CREATE TABLE IF NOT EXISTS attributes (
    id   TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS captures (
    id                  TEXT PRIMARY KEY,
    raw                 BLOB NOT NULL,
    request_fingerprint TEXT NOT NULL,
    model               TEXT NOT NULL,
    url_path            TEXT NOT NULL,
    created_at_ms       INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS judgements (
    id            TEXT PRIMARY KEY,
    json          TEXT NOT NULL,
    capture_id    TEXT NOT NULL REFERENCES captures(id),
    attribute_id  TEXT NOT NULL,
    pair_lo       TEXT NOT NULL,
    pair_hi       TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_judgements_attribute_pair
    ON judgements (attribute_id, pair_lo, pair_hi);
";

/// Everything that can go wrong reading or appending to the log.
///
/// Variants are grouped by what they mean to a caller: self-inconsistent
/// input, an id collision with different content, a missing link in a
/// chain the caller asked to walk, or a lower-level I/O/codec failure.
#[derive(Debug, Error)]
pub enum LogError {
    /// The underlying SQLite call failed.
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// A value failed to serialize or deserialize as JSON.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    /// A filesystem operation (export/import) failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// A value's declared id does not match a fresh derivation from its own
    /// content — it was never valid, tampering or a construction bug.
    #[error("{what} {id} does not match a fresh id derived from its own content")]
    SelfInconsistent { what: &'static str, id: String },
    /// A value with this id already exists in the log with *different*
    /// content. Same id, same content is a no-op; this is not that.
    #[error("{what} {id} already exists in the log with different content")]
    Conflict { what: &'static str, id: String },
    /// A judgement was rejected because it points at a capture the log has
    /// never seen — every judgement's capture must be inserted first.
    #[error("no capture with id {0} — insert the capture before the judgement")]
    NoSuchCapture(String),
    /// [`EvidenceLog::provenance`] could not find the attribute a judgement
    /// claims to be about.
    #[error("no attribute with id {0}")]
    NoSuchAttribute(String),
    /// [`EvidenceLog::provenance`] could not find one of the entities a
    /// judgement's presentation names.
    #[error("no entity with id {0}")]
    NoSuchEntity(String),
    /// No judgement matches the given id or id prefix.
    #[error("no judgement matching id or prefix {0}")]
    NoSuchJudgement(String),
    /// More than one judgement matches a given id prefix.
    #[error("id prefix {0} matches more than one judgement")]
    AmbiguousJudgement(String),
    /// A judgement's stored content does not hash back to its own id.
    #[error("judgement {0} fails self-verification: content does not match its id")]
    TamperedJudgement(String),
    /// A capture's raw bytes were not valid UTF-8 text on read-back.
    #[error("capture {0} raw bytes are not valid UTF-8")]
    InvalidUtf8(String),
    /// `import_jsonl` rejected a line; nothing from the file was committed.
    #[error("line {line}: {reason}")]
    ImportRejected { line: usize, reason: String },
}

/// The full provenance chain for one judgement: everything it claims to be
/// an ancestor of, resolved and verified present. Constructed only by
/// [`EvidenceLog::provenance`], which fails closed the instant any link is
/// missing — a `ProvenanceChain` in hand is a complete chain, never partial.
#[derive(Clone, Debug, PartialEq)]
pub struct ProvenanceChain {
    /// The judgement itself.
    pub judgement: JudgementRecord,
    /// The raw provider capture the judgement was parsed from.
    pub capture: ProviderCapture,
    /// The entities named by the judgement's presentation, in presented
    /// (slot_a, slot_b) order.
    pub entities: Vec<Entity>,
    /// The attribute the judgement is about.
    pub attribute: Attribute,
}

/// One line of a `jsonl` export: a tagged union over every row shape the
/// log stores, so a dump is a flat, append-only file mirroring the log
/// itself.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum LogLine {
    Entity(Entity),
    Attribute(Attribute),
    Capture(ProviderCapture),
    Judgement(JudgementRecord),
}

/// Full hex string behind a content id (as opposed to `.short()`, which is
/// for error messages only — lookups and storage always use the full id).
fn entity_id_str(id: &EntityId) -> &str {
    &id.0 .0
}

fn attribute_id_str(id: &AttributeId) -> &str {
    &id.0 .0
}

fn capture_id_str(id: &CaptureId) -> &str {
    &id.0 .0
}

/// Escape `%`, `_`, and `\` for use inside a `LIKE ... ESCAPE '\'` pattern.
/// Content ids are lowercase hex and never contain these characters, but a
/// caller-supplied prefix is untrusted input and must not be able to smuggle
/// wildcards into the query.
fn escape_like(input: &str) -> String {
    input
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

/// Append-only SQLite evidence log.
///
/// Every insert method is idempotent on identical content and fails closed
/// (never overwrites, never guesses) on anything else: a repeat of the same
/// value is silently accepted, a collision with different content under the
/// same id is a [`LogError::Conflict`], and a judgement naming a capture the
/// log has never seen is a [`LogError::NoSuchCapture`].
pub struct EvidenceLog {
    conn: Connection,
}

impl EvidenceLog {
    /// Open (creating if needed) a log backed by a file on disk.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, LogError> {
        let conn = Connection::open(path)?;
        Self::from_connection(conn)
    }

    /// Open a transient, in-memory log (tests, scratch runs).
    pub fn open_in_memory() -> Result<Self, LogError> {
        let conn = Connection::open_in_memory()?;
        Self::from_connection(conn)
    }

    fn from_connection(conn: Connection) -> Result<Self, LogError> {
        conn.execute_batch(SCHEMA_SQL)?;
        Ok(Self { conn })
    }

    /// Insert an entity, idempotently.
    ///
    /// Fails closed if `entity.id` does not match a fresh id derived from
    /// `entity.body` (self-inconsistent input), or if an entity already
    /// exists under `entity.id` with different content (`body` or `label`
    /// differ — `label` is not part of the id, so this is the one way two
    /// "same id" entities can legitimately disagree, and it's rejected).
    pub fn insert_entity(&self, entity: &Entity) -> Result<(), LogError> {
        if Entity::new(entity.body.clone()).id != entity.id {
            return Err(LogError::SelfInconsistent {
                what: "entity",
                id: entity.id.short().to_string(),
            });
        }
        let id_str = entity_id_str(&entity.id);
        let existing = self
            .conn
            .query_row(
                "SELECT body, label FROM entities WHERE id = ?1",
                params![id_str],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
            )
            .optional()?;
        match existing {
            None => {
                self.conn.execute(
                    "INSERT INTO entities (id, body, label) VALUES (?1, ?2, ?3)",
                    params![id_str, entity.body, entity.label],
                )?;
                Ok(())
            }
            Some((body, label)) if body == entity.body && label == entity.label => Ok(()),
            Some(_) => Err(LogError::Conflict {
                what: "entity",
                id: entity.id.short().to_string(),
            }),
        }
    }

    /// Insert an attribute, idempotently. Same self-consistency and
    /// same-id-different-content rules as [`Self::insert_entity`].
    pub fn insert_attribute(&self, attribute: &Attribute) -> Result<(), LogError> {
        if Attribute::new(attribute.name.clone(), attribute.text.clone()).id != attribute.id {
            return Err(LogError::SelfInconsistent {
                what: "attribute",
                id: attribute.id.short().to_string(),
            });
        }
        let id_str = attribute_id_str(&attribute.id);
        let existing = self
            .conn
            .query_row(
                "SELECT name, text FROM attributes WHERE id = ?1",
                params![id_str],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()?;
        match existing {
            None => {
                self.conn.execute(
                    "INSERT INTO attributes (id, name, text) VALUES (?1, ?2, ?3)",
                    params![id_str, attribute.name, attribute.text],
                )?;
                Ok(())
            }
            Some((name, text)) if name == attribute.name && text == attribute.text => Ok(()),
            Some(_) => Err(LogError::Conflict {
                what: "attribute",
                id: attribute.id.short().to_string(),
            }),
        }
    }

    /// Insert a raw provider capture, idempotently on identical bytes and
    /// metadata. Fails closed if `capture` fails its own
    /// [`ProviderCapture::verify`], or if a capture already exists under
    /// `capture.id` with different `raw`, `request_fingerprint`, `model`,
    /// `url_path`, or `created_at_ms`.
    pub fn insert_capture(&self, capture: &ProviderCapture) -> Result<(), LogError> {
        if !capture.verify() {
            return Err(LogError::SelfInconsistent {
                what: "capture",
                id: capture.id.short().to_string(),
            });
        }
        let id_str = capture_id_str(&capture.id);
        let existing = self
            .conn
            .query_row(
                "SELECT raw, request_fingerprint, model, url_path, created_at_ms \
                 FROM captures WHERE id = ?1",
                params![id_str],
                |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, i64>(4)?,
                    ))
                },
            )
            .optional()?;
        match existing {
            None => {
                self.conn.execute(
                    "INSERT INTO captures \
                     (id, raw, request_fingerprint, model, url_path, created_at_ms) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![
                        id_str,
                        capture.raw.as_bytes(),
                        capture.request_fingerprint.0,
                        capture.model,
                        capture.url_path,
                        capture.created_at_ms as i64,
                    ],
                )?;
                Ok(())
            }
            Some((raw, fingerprint, model, url_path, created_at_ms))
                if raw == capture.raw.as_bytes()
                    && fingerprint == capture.request_fingerprint.0
                    && model == capture.model
                    && url_path == capture.url_path
                    && created_at_ms == capture.created_at_ms as i64 =>
            {
                Ok(())
            }
            Some(_) => Err(LogError::Conflict {
                what: "capture",
                id: capture.id.short().to_string(),
            }),
        }
    }

    /// Insert a judgement record, idempotently on identical content.
    ///
    /// Fails closed if `record.verify_id()` is false (tampered or
    /// inconsistent content), or if `record.capture` names a capture this
    /// log has never seen — nothing gets to claim a provider capture that
    /// isn't already on record.
    pub fn insert_judgement(&self, record: &JudgementRecord) -> Result<(), LogError> {
        if !record.verify_id() {
            return Err(LogError::TamperedJudgement(record.id.short().to_string()));
        }
        let capture_id_str = capture_id_str(&record.capture);
        let capture_exists: i64 = self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM captures WHERE id = ?1)",
            params![capture_id_str],
            |row| row.get(0),
        )?;
        if capture_exists == 0 {
            return Err(LogError::NoSuchCapture(record.capture.short().to_string()));
        }

        let id_str = &record.id.0 .0;
        let json = serde_json::to_string(record)?;
        let existing: Option<String> = self
            .conn
            .query_row(
                "SELECT json FROM judgements WHERE id = ?1",
                params![id_str],
                |row| row.get(0),
            )
            .optional()?;
        match existing {
            None => {
                let pair = record.presentation.pair_key();
                self.conn.execute(
                    "INSERT INTO judgements \
                     (id, json, capture_id, attribute_id, pair_lo, pair_hi, created_at_ms) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    params![
                        id_str,
                        json,
                        capture_id_str,
                        attribute_id_str(&record.attribute),
                        entity_id_str(&pair.lo),
                        entity_id_str(&pair.hi),
                        record.created_at_ms as i64,
                    ],
                )?;
                Ok(())
            }
            Some(existing_json) if existing_json == json => Ok(()),
            Some(_) => Err(LogError::Conflict {
                what: "judgement",
                id: record.id.short().to_string(),
            }),
        }
    }

    /// All judgements recorded about a given attribute, oldest first.
    pub fn judgements_for(
        &self,
        attribute_id: &AttributeId,
    ) -> Result<Vec<JudgementRecord>, LogError> {
        let mut stmt = self.conn.prepare(
            "SELECT json FROM judgements WHERE attribute_id = ?1 ORDER BY created_at_ms ASC, id ASC",
        )?;
        let mut rows = stmt.query(params![attribute_id_str(attribute_id)])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let json: String = row.get(0)?;
            out.push(serde_json::from_str(&json)?);
        }
        Ok(out)
    }

    /// Look up a judgement by its full id or an unambiguous hex prefix of
    /// it. `Ok(None)` means no match; [`LogError::AmbiguousJudgement`] means
    /// more than one judgement shares the prefix.
    pub fn judgement(&self, id_or_prefix: &str) -> Result<Option<JudgementRecord>, LogError> {
        let pattern = format!("{}%", escape_like(id_or_prefix));
        let mut stmt = self
            .conn
            .prepare("SELECT json FROM judgements WHERE id LIKE ?1 ESCAPE '\\'")?;
        let mut rows = stmt.query(params![pattern])?;
        let mut found: Option<String> = None;
        let mut count = 0usize;
        while let Some(row) = rows.next()? {
            count += 1;
            if count > 1 {
                return Err(LogError::AmbiguousJudgement(id_or_prefix.to_string()));
            }
            found = Some(row.get(0)?);
        }
        match found {
            None => Ok(None),
            Some(json) => {
                let record: JudgementRecord = serde_json::from_str(&json)?;
                // Fail closed on read, not only on write: a row mutated
                // behind the log's back must never flow downstream.
                if !record.verify_id() {
                    return Err(LogError::TamperedJudgement(id_or_prefix.to_string()));
                }
                Ok(Some(record))
            }
        }
    }

    /// Look up a raw provider capture by its exact id.
    pub fn capture(&self, id: &CaptureId) -> Result<Option<ProviderCapture>, LogError> {
        let row = self
            .conn
            .query_row(
                "SELECT raw, request_fingerprint, model, url_path, created_at_ms \
                 FROM captures WHERE id = ?1",
                params![capture_id_str(id)],
                |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, i64>(4)?,
                    ))
                },
            )
            .optional()?;
        let Some((raw, fingerprint, model, url_path, created_at_ms)) = row else {
            return Ok(None);
        };
        let raw =
            String::from_utf8(raw).map_err(|_| LogError::InvalidUtf8(id.short().to_string()))?;
        Ok(Some(ProviderCapture {
            id: id.clone(),
            raw,
            request_fingerprint: ContentId(fingerprint),
            model,
            url_path,
            created_at_ms: created_at_ms as u64,
        }))
    }

    fn entity_by_id(&self, id: &EntityId) -> Result<Option<Entity>, LogError> {
        let row = self
            .conn
            .query_row(
                "SELECT body, label FROM entities WHERE id = ?1",
                params![entity_id_str(id)],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
            )
            .optional()?;
        Ok(row.map(|(body, label)| Entity {
            id: id.clone(),
            body,
            label,
        }))
    }

    /// List every registered attribute (name + full text), for lookup UIs.
    pub fn attributes(&self) -> Result<Vec<Attribute>, LogError> {
        let mut stmt = self
            .conn
            .prepare("SELECT name, text FROM attributes ORDER BY name")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (name, text) = row?;
            out.push(Attribute::new(name, text));
        }
        Ok(out)
    }

    fn attribute_by_id(&self, id: &AttributeId) -> Result<Option<Attribute>, LogError> {
        let row = self
            .conn
            .query_row(
                "SELECT name, text FROM attributes WHERE id = ?1",
                params![attribute_id_str(id)],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()?;
        Ok(row.map(|(name, text)| Attribute {
            id: id.clone(),
            name,
            text,
        }))
    }

    /// Walk a judgement (by full id or unambiguous prefix) back to its full
    /// provenance chain: the raw capture, the attribute, and both presented
    /// entities. Fails closed — [`LogError::NoSuchCapture`],
    /// [`LogError::NoSuchAttribute`], or [`LogError::NoSuchEntity`] — the
    /// instant any link the judgement names is missing from the log.
    pub fn provenance(&self, id_or_prefix: &str) -> Result<ProvenanceChain, LogError> {
        let judgement = self
            .judgement(id_or_prefix)?
            .ok_or_else(|| LogError::NoSuchJudgement(id_or_prefix.to_string()))?;
        let capture = self
            .capture(&judgement.capture)?
            .ok_or_else(|| LogError::NoSuchCapture(judgement.capture.short().to_string()))?;
        let attribute = self
            .attribute_by_id(&judgement.attribute)?
            .ok_or_else(|| LogError::NoSuchAttribute(judgement.attribute.short().to_string()))?;
        let slot_a = self
            .entity_by_id(&judgement.presentation.slot_a)?
            .ok_or_else(|| {
                LogError::NoSuchEntity(judgement.presentation.slot_a.short().to_string())
            })?;
        let slot_b = self
            .entity_by_id(&judgement.presentation.slot_b)?
            .ok_or_else(|| {
                LogError::NoSuchEntity(judgement.presentation.slot_b.short().to_string())
            })?;
        Ok(ProvenanceChain {
            judgement,
            capture,
            entities: vec![slot_a, slot_b],
            attribute,
        })
    }

    /// Dump the entire log as newline-delimited JSON, one tagged
    /// one tagged line per row, entities/attributes/captures/judgements in
    /// that order (a valid dependency order for [`Self::import_jsonl`]).
    pub fn export_jsonl(&self, path: impl AsRef<Path>) -> Result<(), LogError> {
        let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);

        let mut stmt = self
            .conn
            .prepare("SELECT id, body, label FROM entities ORDER BY id ASC")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let body: String = row.get(1)?;
            let label: Option<String> = row.get(2)?;
            let line = LogLine::Entity(Entity {
                id: EntityId(ContentId(id)),
                body,
                label,
            });
            writeln!(out, "{}", serde_json::to_string(&line)?)?;
        }
        drop(rows);
        drop(stmt);

        let mut stmt = self
            .conn
            .prepare("SELECT id, name, text FROM attributes ORDER BY id ASC")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let name: String = row.get(1)?;
            let text: String = row.get(2)?;
            let line = LogLine::Attribute(Attribute {
                id: AttributeId(ContentId(id)),
                name,
                text,
            });
            writeln!(out, "{}", serde_json::to_string(&line)?)?;
        }
        drop(rows);
        drop(stmt);

        let mut stmt = self.conn.prepare(
            "SELECT id, raw, request_fingerprint, model, url_path, created_at_ms \
             FROM captures ORDER BY id ASC",
        )?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let raw: Vec<u8> = row.get(1)?;
            let fingerprint: String = row.get(2)?;
            let model: String = row.get(3)?;
            let url_path: String = row.get(4)?;
            let created_at_ms: i64 = row.get(5)?;
            let raw = String::from_utf8(raw)
                .map_err(|_| LogError::InvalidUtf8(id.chars().take(12).collect()))?;
            let line = LogLine::Capture(ProviderCapture {
                id: CaptureId(ContentId(id)),
                raw,
                request_fingerprint: ContentId(fingerprint),
                model,
                url_path,
                created_at_ms: created_at_ms as u64,
            });
            writeln!(out, "{}", serde_json::to_string(&line)?)?;
        }
        drop(rows);
        drop(stmt);

        let mut stmt = self
            .conn
            .prepare("SELECT json FROM judgements ORDER BY id ASC")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let json: String = row.get(0)?;
            let record: JudgementRecord = serde_json::from_str(&json)?;
            writeln!(
                out,
                "{}",
                serde_json::to_string(&LogLine::Judgement(record))?
            )?;
        }

        out.flush()?;
        Ok(())
    }

    /// Load a `jsonl` dump produced by [`Self::export_jsonl`] (or any file
    /// in the same shape). Every line's content id is re-verified before
    /// anything is applied; the whole file is loaded atomically, so a
    /// tampered line anywhere leaves the log exactly as it was, reported
    /// with its 1-based line number in [`LogError::ImportRejected`].
    pub fn import_jsonl(&self, path: impl AsRef<Path>) -> Result<(), LogError> {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);
        let tx = self.conn.unchecked_transaction()?;

        for (idx, line) in reader.lines().enumerate() {
            let line_no = idx + 1;
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let parsed: LogLine =
                serde_json::from_str(&line).map_err(|e| LogError::ImportRejected {
                    line: line_no,
                    reason: e.to_string(),
                })?;
            let applied = match parsed {
                LogLine::Entity(e) => self.insert_entity(&e),
                LogLine::Attribute(a) => self.insert_attribute(&a),
                LogLine::Capture(c) => self.insert_capture(&c),
                LogLine::Judgement(j) => self.insert_judgement(&j),
            };
            applied.map_err(|e| LogError::ImportRejected {
                line: line_no,
                reason: e.to_string(),
            })?;
        }

        tx.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::AnswerAtom;
    use crate::evidence::evidence_from_resamples;
    use crate::ontology::{Presentation, TemplateHash};
    use crate::record::{
        AcquisitionMode, Cost, DecodeConfig, EvidenceHealth, InstrumentKind, ParserVersion,
    };

    fn fingerprint(bytes: &[u8]) -> ContentId {
        ContentId::derive("seriate/gateway-request", bytes)
    }

    fn sample_capture(raw: &str) -> ProviderCapture {
        ProviderCapture::new(
            raw,
            fingerprint(b"req"),
            "test/model",
            "/chat/completions",
            1_700_000_000_000,
        )
    }

    fn sample_judgement(
        attribute: &Attribute,
        a: &Entity,
        b: &Entity,
        capture: &ProviderCapture,
    ) -> JudgementRecord {
        let evidence = evidence_from_resamples(&[AnswerAtom::A(1)]).unwrap();
        JudgementRecord::new(
            InstrumentKind::RatioLetterPairwise,
            AcquisitionMode::Sampled,
            attribute.id.clone(),
            Presentation {
                slot_a: a.id.clone(),
                slot_b: b.id.clone(),
            },
            TemplateHash::derive(b"template"),
            ParserVersion("ratio-letter/1".into()),
            "test/model".into(),
            DecodeConfig {
                temperature: 0.0,
                max_tokens: 8,
                top_logprobs: Some(20),
            },
            capture.id.clone(),
            evidence,
            EvidenceHealth {
                visible_mass: 1.0,
                parsed_cleanly: true,
                refused: false,
            },
            Cost::default(),
            1_700_000_000_000,
        )
    }

    #[test]
    fn source_contains_no_mutation_or_removal_sql() {
        // Build the forbidden words at runtime so this line's own text
        // never contains the literal substring it's checking for.
        let forbidden = [["UPD", "ATE"].concat(), ["DEL", "ETE"].concat()];
        let source = include_str!("log.rs").to_uppercase();
        for word in forbidden {
            assert!(
                !source.contains(&word),
                "log.rs must never spell out {word} SQL — the log is append-only"
            );
        }
    }

    #[test]
    fn entity_and_attribute_round_trip_and_are_idempotent() {
        let log = EvidenceLog::open_in_memory().unwrap();
        let e = Entity::new("a raw entity body");
        log.insert_entity(&e).unwrap();
        log.insert_entity(&e).unwrap(); // idempotent no-op

        let a = Attribute::new("rawness", "how raw the writing is");
        log.insert_attribute(&a).unwrap();
        log.insert_attribute(&a).unwrap();

        assert_eq!(log.entity_by_id(&e.id).unwrap(), Some(e));
        assert_eq!(log.attribute_by_id(&a.id).unwrap(), Some(a));
    }

    #[test]
    fn entity_same_id_different_label_is_a_conflict() {
        let log = EvidenceLog::open_in_memory().unwrap();
        let e1 = Entity::new("shared body");
        log.insert_entity(&e1).unwrap();

        let mut e2 = e1.clone();
        e2.label = Some("a label that was never there before".into());
        let err = log.insert_entity(&e2).unwrap_err();
        assert!(matches!(err, LogError::Conflict { what: "entity", .. }));
    }

    #[test]
    fn capture_round_trips_and_is_idempotent() {
        let log = EvidenceLog::open_in_memory().unwrap();
        let c = sample_capture(r#"{"choices":[]}"#);
        log.insert_capture(&c).unwrap();
        log.insert_capture(&c).unwrap();
        assert_eq!(log.capture(&c.id).unwrap(), Some(c));
    }

    #[test]
    fn capture_self_inconsistent_is_rejected() {
        let log = EvidenceLog::open_in_memory().unwrap();
        let mut c = sample_capture("original");
        c.raw = "tampered".into(); // id no longer matches raw
        let err = log.insert_capture(&c).unwrap_err();
        assert!(matches!(
            err,
            LogError::SelfInconsistent {
                what: "capture",
                ..
            }
        ));
    }

    #[test]
    fn judgement_requires_capture_to_exist_first() {
        let log = EvidenceLog::open_in_memory().unwrap();
        let a = Attribute::new("rawness", "how raw");
        let e1 = Entity::new("first");
        let e2 = Entity::new("second");
        let capture = sample_capture("never inserted");
        let record = sample_judgement(&a, &e1, &e2, &capture);

        let err = log.insert_judgement(&record).unwrap_err();
        assert!(matches!(err, LogError::NoSuchCapture(_)));
    }

    #[test]
    fn tampered_judgement_json_is_rejected() {
        let log = EvidenceLog::open_in_memory().unwrap();
        let a = Attribute::new("rawness", "how raw");
        let e1 = Entity::new("first");
        let e2 = Entity::new("second");
        let capture = sample_capture("captured bytes");
        log.insert_capture(&capture).unwrap();
        let mut record = sample_judgement(&a, &e1, &e2, &capture);
        record.model = "tampered/model".into(); // id no longer matches content

        let err = log.insert_judgement(&record).unwrap_err();
        assert!(matches!(err, LogError::TamperedJudgement(_)));
    }

    #[test]
    fn judgement_round_trips_and_is_idempotent_and_indexed() {
        let log = EvidenceLog::open_in_memory().unwrap();
        let a = Attribute::new("rawness", "how raw");
        let e1 = Entity::new("first");
        let e2 = Entity::new("second");
        let capture = sample_capture("captured bytes");
        log.insert_capture(&capture).unwrap();
        let record = sample_judgement(&a, &e1, &e2, &capture);

        log.insert_judgement(&record).unwrap();
        log.insert_judgement(&record).unwrap(); // idempotent no-op

        assert_eq!(
            log.judgement(&record.id.0 .0).unwrap(),
            Some(record.clone())
        );
        // A short unambiguous prefix resolves too.
        assert_eq!(
            log.judgement(&record.id.0 .0[..12]).unwrap(),
            Some(record.clone())
        );
        assert_eq!(log.judgements_for(&a.id).unwrap(), vec![record]);
    }

    #[test]
    fn provenance_walks_the_full_chain() {
        let log = EvidenceLog::open_in_memory().unwrap();
        let a = Attribute::new("rawness", "how raw");
        let e1 = Entity::new("first");
        let e2 = Entity::new("second");
        let capture = sample_capture("captured bytes");
        log.insert_entity(&e1).unwrap();
        log.insert_entity(&e2).unwrap();
        log.insert_attribute(&a).unwrap();
        log.insert_capture(&capture).unwrap();
        let record = sample_judgement(&a, &e1, &e2, &capture);
        log.insert_judgement(&record).unwrap();

        let chain = log.provenance(&record.id.0 .0).unwrap();
        assert_eq!(chain.judgement, record);
        assert_eq!(chain.capture, capture);
        assert_eq!(chain.attribute, a);
        assert_eq!(chain.entities, vec![e1, e2]);
    }

    #[test]
    fn provenance_fails_closed_on_missing_entity() {
        let log = EvidenceLog::open_in_memory().unwrap();
        let a = Attribute::new("rawness", "how raw");
        let e1 = Entity::new("first");
        let e2 = Entity::new("second"); // deliberately never inserted
        let capture = sample_capture("captured bytes");
        log.insert_entity(&e1).unwrap();
        log.insert_attribute(&a).unwrap();
        log.insert_capture(&capture).unwrap();
        let record = sample_judgement(&a, &e1, &e2, &capture);
        log.insert_judgement(&record).unwrap();

        let err = log.provenance(&record.id.0 .0).unwrap_err();
        assert!(matches!(err, LogError::NoSuchEntity(_)));
    }

    #[test]
    fn jsonl_export_import_round_trips() {
        let log = EvidenceLog::open_in_memory().unwrap();
        let a = Attribute::new("rawness", "how raw");
        let e1 = Entity::new("first");
        let e2 = Entity::new("second");
        let capture = sample_capture("captured bytes");
        log.insert_entity(&e1).unwrap();
        log.insert_entity(&e2).unwrap();
        log.insert_attribute(&a).unwrap();
        log.insert_capture(&capture).unwrap();
        let record = sample_judgement(&a, &e1, &e2, &capture);
        log.insert_judgement(&record).unwrap();

        let path = std::env::temp_dir().join(format!(
            "seriate-log-test-{}-{}.jsonl",
            std::process::id(),
            rand::random::<u64>()
        ));
        log.export_jsonl(&path).unwrap();

        let restored = EvidenceLog::open_in_memory().unwrap();
        restored.import_jsonl(&path).unwrap();

        assert_eq!(restored.entity_by_id(&e1.id).unwrap(), Some(e1.clone()));
        assert_eq!(restored.entity_by_id(&e2.id).unwrap(), Some(e2.clone()));
        assert_eq!(restored.attribute_by_id(&a.id).unwrap(), Some(a.clone()));
        assert_eq!(
            restored.capture(&capture.id).unwrap(),
            Some(capture.clone())
        );
        assert_eq!(
            restored.judgement(&record.id.0 .0).unwrap(),
            Some(record.clone())
        );
        let chain = restored.provenance(&record.id.0 .0).unwrap();
        assert_eq!(chain.entities, vec![e1, e2]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn jsonl_import_rejects_tampered_line_and_commits_nothing() {
        let log = EvidenceLog::open_in_memory().unwrap();
        let e1 = Entity::new("first");
        let e2 = Entity::new("second");
        log.insert_entity(&e1).unwrap();
        log.insert_entity(&e2).unwrap();

        let path = std::env::temp_dir().join(format!(
            "seriate-log-tamper-test-{}-{}.jsonl",
            std::process::id(),
            rand::random::<u64>()
        ));
        log.export_jsonl(&path).unwrap();

        // Corrupt the first line's body without touching its id — the
        // resulting entity is self-inconsistent.
        let contents = std::fs::read_to_string(&path).unwrap();
        let mut lines: Vec<String> = contents.lines().map(String::from).collect();
        assert!(!lines.is_empty());
        let mut first: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        first["body"] = serde_json::Value::String("a different body entirely".into());
        lines[0] = serde_json::to_string(&first).unwrap();
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();

        let restored = EvidenceLog::open_in_memory().unwrap();
        let err = restored.import_jsonl(&path).unwrap_err();
        assert!(matches!(err, LogError::ImportRejected { line: 1, .. }));

        // Nothing from the file was committed, including later valid lines.
        assert_eq!(restored.entity_by_id(&e2.id).unwrap(), None);

        let _ = std::fs::remove_file(&path);
    }
}
