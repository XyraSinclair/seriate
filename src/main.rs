#![forbid(unsafe_code)]

//! seriate CLI: annotate entity sets with provenanced judgements, compile
//! ordering posteriors, walk provenance chains, probe logprob reality.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use seriate::instrument::ordinal::OrdinalInstrument;
use seriate::instrument::ratio_letter::RatioLetterInstrument;
use seriate::instrument::Instrument;
use seriate::{
    compile, probe, Attribute, ChatSpec, Cost, DecodeConfig, Entity, EvidenceLog, Gateway,
    JudgementRecord, Presentation, Tolerances,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum InstrumentArg {
    RatioLetter,
    Ordinal,
}

#[derive(Parser)]
#[command(
    name = "seriate",
    version,
    about = "Provenanced attribute annotation and LLM prior elicitation over entity sets"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Elicit pairwise judgements over an entity set and store them with
    /// full provenance
    Annotate {
        /// File with one entity body per line
        #[arg(long)]
        entities: PathBuf,
        /// Short attribute handle, e.g. "rawness"
        #[arg(long)]
        attribute_name: String,
        /// Full judging text; hashed into the attribute id
        #[arg(long)]
        attribute_text: String,
        /// Model slug (OpenRouter)
        #[arg(long, default_value = "openai/gpt-5.4-mini")]
        model: String,
        /// Elicitation instrument
        #[arg(long, value_enum, default_value = "ratio-letter")]
        instrument: InstrumentArg,
        /// Pair budget: "all" or "random:N"
        #[arg(long, default_value = "all")]
        pairs: String,
        /// Ask each pair in one canonical order only (default: both orders)
        #[arg(long)]
        no_counterbalance: bool,
        /// Evidence log path
        #[arg(long, default_value = "seriate.sqlite")]
        db: PathBuf,
        /// Seed for random pair sampling
        #[arg(long, default_value_t = 7)]
        seed: u64,
        /// Requested top-k logprob depth
        #[arg(long, default_value_t = 20)]
        top_logprobs: u8,
    },
    /// Compile stored judgements for an attribute into an ordering posterior
    Compile {
        /// Attribute handle as given to annotate
        #[arg(long)]
        attribute_name: String,
        #[arg(long, default_value = "seriate.sqlite")]
        db: PathBuf,
        /// Emit the full posterior as JSON on stdout
        #[arg(long)]
        json: bool,
    },
    /// Walk a judgement's provenance chain down to raw provider bytes
    Provenance {
        /// Judgement id prefix (unique)
        id_prefix: String,
        #[arg(long, default_value = "seriate.sqlite")]
        db: PathBuf,
        /// Include the raw provider response body
        #[arg(long)]
        raw: bool,
    },
    /// Measure logprob reality for a list of models
    Probe {
        /// Comma-separated model slugs
        #[arg(long)]
        models: String,
        /// Sampled runs per model for the agreement check
        #[arg(long, default_value_t = 5)]
        samples: u8,
        /// Write reports to this JSONL path
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Export the evidence log to JSONL
    Export {
        #[arg(long, default_value = "seriate.sqlite")]
        db: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Import a JSONL export (verifying every content id)
    Import {
        #[arg(long, default_value = "seriate.sqlite")]
        db: PathBuf,
        #[arg(long, name = "in")]
        input: PathBuf,
    },
}

fn require_key() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("OPENROUTER_API_KEY").is_err() {
        return Err("OPENROUTER_API_KEY is not set. Create a key at \
             https://openrouter.ai/keys and `export OPENROUTER_API_KEY=...`."
            .into());
    }
    Ok(())
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn resolve_attribute(
    log: &EvidenceLog,
    name: &str,
) -> Result<Attribute, Box<dyn std::error::Error>> {
    let matches: Vec<Attribute> = log
        .attributes()?
        .into_iter()
        .filter(|a| a.name == name)
        .collect();
    match matches.len() {
        0 => Err(format!("no attribute named {name:?} in the log").into()),
        1 => Ok(matches.into_iter().next().expect("len checked")),
        n => Err(format!(
            "{n} attributes named {name:?} (different texts); disambiguate by re-annotating \
             with a distinct name"
        )
        .into()),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Annotate {
            entities,
            attribute_name,
            attribute_text,
            model,
            instrument,
            pairs,
            no_counterbalance,
            db,
            seed,
            top_logprobs,
        } => {
            require_key()?;
            let bodies = std::fs::read_to_string(&entities)
                .map_err(|e| format!("failed to read {}: {e}", entities.display()))?;
            let roster: Vec<Entity> = bodies
                .lines()
                .map(|l| l.trim_end_matches('\r'))
                .filter(|l| !l.trim().is_empty())
                .map(Entity::new)
                .collect();
            if roster.len() < 2 {
                return Err("need at least 2 entities".into());
            }
            let attribute = Attribute::new(&attribute_name, &attribute_text);
            let log = EvidenceLog::open(&db)?;
            for e in &roster {
                log.insert_entity(e)?;
            }
            log.insert_attribute(&attribute)?;

            // Pair selection.
            let mut all_pairs = Vec::new();
            for i in 0..roster.len() {
                for j in (i + 1)..roster.len() {
                    all_pairs.push((i, j));
                }
            }
            let selected: Vec<(usize, usize)> = if let Some(spec) = pairs.strip_prefix("random:") {
                let n: usize = spec.parse().map_err(|_| format!("bad --pairs {pairs:?}"))?;
                let mut rng = StdRng::seed_from_u64(seed);
                let mut shuffled = all_pairs.clone();
                shuffled.shuffle(&mut rng);
                shuffled.truncate(n);
                shuffled
            } else if pairs == "all" {
                all_pairs
            } else {
                return Err(format!("bad --pairs {pairs:?}: use \"all\" or \"random:N\"").into());
            };

            let gateway = Gateway::from_env()?;
            let instrument: Box<dyn Instrument> = match instrument {
                InstrumentArg::RatioLetter => Box::new(RatioLetterInstrument),
                InstrumentArg::Ordinal => Box::new(OrdinalInstrument),
            };

            let mut judgements = 0usize;
            let mut refusals = 0usize;
            let mut nanodollars: i64 = 0;
            for &(i, j) in &selected {
                let orders: Vec<(usize, usize)> = if no_counterbalance {
                    vec![(i, j)]
                } else {
                    vec![(i, j), (j, i)]
                };
                for (a_idx, b_idx) in orders {
                    let slot_a = &roster[a_idx];
                    let slot_b = &roster[b_idx];
                    let rendered = instrument.render(&attribute, slot_a, slot_b);
                    let spec = ChatSpec {
                        model: model.clone(),
                        system: rendered.system.clone(),
                        user: rendered.user.clone(),
                        temperature: 0.0,
                        max_tokens: 16,
                        top_logprobs: Some(top_logprobs),
                        response_format_json: false,
                    };
                    let outcome = gateway.chat(&spec).await?;
                    nanodollars += outcome.usage.cost_nanodollars;
                    let parsed = match instrument
                        .parse(&outcome.content, outcome.answer_logprobs.as_deref())
                    {
                        Ok(parsed) => parsed,
                        Err(err) => {
                            eprintln!(
                                "{}~{} unparseable ({err}); capture kept, no judgement",
                                slot_a.id.short(),
                                slot_b.id.short()
                            );
                            log.insert_capture(&outcome.capture)?;
                            continue;
                        }
                    };
                    log.insert_capture(&outcome.capture)?;
                    if parsed.health.refused {
                        refusals += 1;
                    }
                    let top = parsed
                        .evidence
                        .support()
                        .iter()
                        .max_by(|x, y| x.p.total_cmp(&y.p))
                        .map(|x| x.atom.letter().unwrap_or('?'))
                        .unwrap_or('?');
                    let record = JudgementRecord::new(
                        instrument.kind(),
                        parsed.mode,
                        attribute.id.clone(),
                        Presentation {
                            slot_a: slot_a.id.clone(),
                            slot_b: slot_b.id.clone(),
                        },
                        rendered.template.clone(),
                        instrument.parser_version(),
                        model.clone(),
                        DecodeConfig {
                            temperature: 0.0,
                            max_tokens: 16,
                            top_logprobs: Some(top_logprobs),
                        },
                        outcome.capture.id.clone(),
                        parsed.evidence,
                        parsed.health.clone(),
                        Cost {
                            input_tokens: outcome.usage.input_tokens,
                            output_tokens: outcome.usage.output_tokens,
                            nanodollars: outcome.usage.cost_nanodollars,
                            is_estimate: outcome.usage.cost_is_estimate,
                        },
                        now_ms(),
                    );
                    log.insert_judgement(&record)?;
                    judgements += 1;
                    eprintln!(
                        "{}>{} [{}] top {} · visible {:.2} · ${:.6}",
                        slot_a.id.short(),
                        slot_b.id.short(),
                        record.id.short(),
                        top,
                        parsed.health.visible_mass,
                        outcome.usage.cost_nanodollars as f64 / 1e9,
                    );
                }
            }
            eprintln!(
                "annotated {judgements} judgements ({refusals} refusals) · ${:.4} · db {}",
                nanodollars as f64 / 1e9,
                db.display()
            );
        }
        Commands::Compile {
            attribute_name,
            db,
            json,
        } => {
            let log = EvidenceLog::open(&db)?;
            let attribute = resolve_attribute(&log, &attribute_name)?;
            let records = log.judgements_for(&attribute.id)?;
            if records.is_empty() {
                return Err(format!("no judgements for attribute {attribute_name:?}").into());
            }
            // Roster = every entity referenced by the records, in first-seen
            // order, resolved through the log.
            let mut roster: Vec<Entity> = Vec::new();
            for record in &records {
                for id in [
                    record.presentation.slot_a.clone(),
                    record.presentation.slot_b.clone(),
                ] {
                    if !roster.iter().any(|e| e.id == id) {
                        let chain = log.provenance(&record.id.0 .0)?;
                        for e in chain.entities {
                            if e.id == id && !roster.iter().any(|r| r.id == e.id) {
                                roster.push(e);
                            }
                        }
                    }
                }
            }
            let posterior = compile(&records, &roster, &Tolerances::default())?;
            if json {
                println!("{}", serde_json::to_string_pretty(&posterior)?);
            } else {
                let mut order: Vec<usize> = (0..roster.len()).collect();
                order.sort_by(|&a, &b| {
                    posterior.entities[b]
                        .latent_mean
                        .total_cmp(&posterior.entities[a].latent_mean)
                });
                for idx in order {
                    let ep = &posterior.entities[idx];
                    let rank = ep.rank.map(|r| r.to_string()).unwrap_or_else(|| "-".into());
                    let body: String = roster[idx].body.chars().take(60).collect();
                    println!(
                        "{rank:>3}  {:+.3} ± {:.3}  {}",
                        ep.latent_mean, ep.latent_std, body
                    );
                }
            }
            eprintln!(
                "compiled {} records ({} refused, {} uninformative skipped) · {} components",
                posterior.records_used,
                posterior.records_skipped_refused,
                posterior.records_skipped_uninformative,
                posterior.components.len(),
            );
        }
        Commands::Provenance { id_prefix, db, raw } => {
            let log = EvidenceLog::open(&db)?;
            let chain = log.provenance(&id_prefix)?;
            let r = &chain.judgement;
            println!("judgement  {}", r.id.0 .0);
            println!("instrument {:?} · mode {:?}", r.instrument, r.mode);
            println!("model      {}", r.model);
            println!("parser     {}", r.parser.0);
            println!("template   {}", r.template.short());
            println!(
                "attribute  {} ({})",
                chain.attribute.name,
                r.attribute.short()
            );
            println!(
                "presented  A={} B={}",
                chain.entities[0].body.chars().take(40).collect::<String>(),
                chain.entities[1].body.chars().take(40).collect::<String>(),
            );
            println!("evidence   ({:?})", r.evidence.completeness);
            for point in r.evidence.support() {
                let letter = point.atom.letter().map(|c| c.to_string());
                let label = letter.unwrap_or_else(|| format!("{:?}", point.atom));
                println!("  {label:>12}  {:.4}", point.p);
            }
            println!(
                "health     visible {:.3} · clean {} · refused {}",
                r.health.visible_mass, r.health.parsed_cleanly, r.health.refused
            );
            println!(
                "cost       {} in / {} out tokens · ${:.6}{}",
                r.cost.input_tokens,
                r.cost.output_tokens,
                r.cost.nanodollars as f64 / 1e9,
                if r.cost.is_estimate {
                    " (estimate)"
                } else {
                    ""
                }
            );
            println!(
                "capture    {} · {} · {} · t={}",
                chain.capture.id.short(),
                chain.capture.model,
                chain.capture.url_path,
                chain.capture.created_at_ms
            );
            if raw {
                println!("--- raw provider bytes ---");
                println!("{}", chain.capture.raw);
            }
        }
        Commands::Probe {
            models,
            samples,
            out,
        } => {
            require_key()?;
            let gateway = Gateway::from_env()?;
            let attribute =
                Attribute::new("brightness", "how much light the object itself gives off");
            let a = Entity::new("the noonday sun over a desert");
            let b = Entity::new("a single birthday-cake candle");
            let mut reports = Vec::new();
            println!(
                "{:<40} {:>8} {:>6} {:>6} {:>8} {:>8} {:>10}",
                "model", "logprobs", "depth", "answer", "visible", "jsd", "cost$"
            );
            for model in models.split(',').map(str::trim).filter(|m| !m.is_empty()) {
                match probe::probe_model(&gateway, model, &attribute, &a, &b, samples).await {
                    Ok(report) => {
                        println!(
                            "{:<40} {:>8} {:>6} {:>6} {:>8} {:>8} {:>10.6}",
                            report.model,
                            if report.logprobs_returned {
                                "yes"
                            } else {
                                "NO"
                            },
                            report.top_k_depth,
                            if report.answer_position_found {
                                "yes"
                            } else {
                                "NO"
                            },
                            report
                                .visible_mass
                                .map(|m| format!("{m:.3}"))
                                .unwrap_or_else(|| "-".into()),
                            report
                                .sampled_agreement_jsd
                                .map(|d| format!("{d:.3}"))
                                .unwrap_or_else(|| "-".into()),
                            report.cost_nanodollars as f64 / 1e9,
                        );
                        reports.push(report);
                    }
                    Err(err) => {
                        println!("{model:<40} ERROR: {err}");
                    }
                }
            }
            if let Some(path) = out {
                probe::write_reports_jsonl(&path, &reports)?;
                eprintln!("wrote {} reports to {}", reports.len(), path.display());
            }
        }
        Commands::Export { db, out } => {
            let log = EvidenceLog::open(&db)?;
            log.export_jsonl(&out)?;
            eprintln!("exported to {}", out.display());
        }
        Commands::Import { db, input } => {
            let log = EvidenceLog::open(&db)?;
            log.import_jsonl(&input)?;
            eprintln!("imported from {}", input.display());
        }
    }
    Ok(())
}
