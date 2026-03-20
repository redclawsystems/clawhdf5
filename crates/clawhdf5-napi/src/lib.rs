//! Node.js native addon for clawhdf5-agent.
//!
//! Built with [napi-rs](https://napi.rs).  Exposes [`ClawhdfMemory`], a Node.js
//! class that wraps [`clawhdf5_agent::openclaw::ClawhdfBackend`] and the
//! hippocampal consolidation engine.
//!
//! # Build
//! ```
//! npm install -g @napi-rs/cli
//! cd packages/clawhdf5-node
//! napi build --platform --release
//! ```

#![deny(clippy::all)]
#![allow(clippy::too_many_arguments)]

use napi::bindgen_prelude::*;
use napi_derive::napi;

use clawhdf5_agent::openclaw::ClawhdfBackend;

// ─────────────────────────────────────────────────────────────────────────────
// Plain JS object types (returned by value from napi methods)
// ─────────────────────────────────────────────────────────────────────────────

/// A single result returned from a memory search.
#[napi(object)]
pub struct MemorySearchResult {
    /// Text content of the matching memory record.
    pub text: String,
    /// Relevance score (higher = more relevant). Uses f64 for JS number
    /// compatibility.
    pub score: f64,
    /// Source file / section path the record originated from.
    pub path: String,
    /// Optional `[start, end]` line range within the source file.
    pub line_range: Option<Vec<u32>>,
    /// Unix-epoch timestamp of the record, if available.
    pub timestamp: Option<f64>,
    /// Human-readable source description (channel / file type).
    pub source: String,
}

/// Aggregate statistics about the memory store.
#[napi(object)]
pub struct BackendStats {
    /// Number of active (non-deleted) records.
    pub total_records: u32,
    /// Number of records that have a non-empty embedding vector.
    pub total_embeddings: u32,
    /// On-disk size of the .h5 file in bytes.  Reported as f64 because JS
    /// does not have a native u64 (BigInt support would require an extra dep).
    pub file_size_bytes: f64,
    /// List of modalities present (e.g. `["text"]`).
    pub modalities: Vec<String>,
    /// Unix-epoch seconds of the most recently stored record, if any.
    pub last_updated: Option<f64>,
}

/// Per-tier counts and running totals from one consolidation cycle.
#[napi(object)]
pub struct ConsolidationStats {
    /// Number of records in the Working tier after consolidation.
    pub working_count: u32,
    /// Number of records in the Episodic tier after consolidation.
    pub episodic_count: u32,
    /// Number of records in the Semantic tier after consolidation.
    pub semantic_count: u32,
    /// Cumulative records evicted (capacity overflow, decay).
    pub total_evictions: f64,
    /// Cumulative records promoted (Working→Episodic, Episodic→Semantic).
    pub total_promotions: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// ClawhdfMemory class
// ─────────────────────────────────────────────────────────────────────────────

/// Node.js class wrapping the clawhdf5 HDF5-backed memory backend.
///
/// ```ts
/// import { ClawhdfMemory } from '@redclaw/clawhdf5';
///
/// const mem = ClawhdfMemory.openOrCreate('./agent.brain', 768);
/// mem.write('memory/user.md', '# Goals\n\nLearn Rust.');
/// const results = mem.search('Rust', new Float32Array(768), 5);
/// ```
#[napi]
pub struct ClawhdfMemory {
    inner: ClawhdfBackend,
}

// SAFETY: Node.js executes JavaScript on a single thread.  The napi-rs runtime
// serialises all JS↔native calls through the event loop, so there is no
// concurrent access to `inner` from multiple threads.
unsafe impl Send for ClawhdfMemory {}

#[napi]
impl ClawhdfMemory {
    // ── Lifecycle ────────────────────────────────────────────────────────────

    /// Create a new HDF5 memory store at `path`.
    ///
    /// `embeddingDim` must match the external embedder you plan to use
    /// (e.g. `384` for `all-MiniLM-L6-v2`, `768` for `nomic-embed-text`,
    /// `1536` for OpenAI `text-embedding-3-small`).
    #[napi(factory)]
    pub fn create(path: String, embedding_dim: u32) -> napi::Result<ClawhdfMemory> {
        let backend =
            ClawhdfBackend::create(std::path::Path::new(&path), embedding_dim as usize)
                .map_err(napi::Error::from_reason)?;
        Ok(Self { inner: backend })
    }

    /// Open an existing HDF5 memory store.  Replays the WAL if present.
    #[napi(factory)]
    pub fn open(path: String) -> napi::Result<ClawhdfMemory> {
        let backend =
            ClawhdfBackend::open(std::path::Path::new(&path))
                .map_err(napi::Error::from_reason)?;
        Ok(Self { inner: backend })
    }

    /// Open the store at `path` if it exists; otherwise create it.
    #[napi(factory)]
    pub fn open_or_create(path: String, embedding_dim: u32) -> napi::Result<ClawhdfMemory> {
        let backend =
            ClawhdfBackend::open_or_create(std::path::Path::new(&path), embedding_dim as usize)
                .map_err(napi::Error::from_reason)?;
        Ok(Self { inner: backend })
    }

    // ── MemoryBackend methods ─────────────────────────────────────────────────

    /// Hybrid vector + BM25 search.
    ///
    /// `queryEmbedding` must have the same dimension the store was created with.
    /// Pass a zero-length `Float32Array` to skip vector similarity and rely on
    /// BM25 text matching only.
    #[napi]
    pub fn search(
        &mut self,
        query_text: String,
        query_embedding: Float32Array,
        k: u32,
    ) -> Vec<MemorySearchResult> {
        use clawhdf5_agent::openclaw::MemoryBackend;
        let raw = self
            .inner
            .search(&query_text, query_embedding.as_ref(), k as usize);
        raw.into_iter()
            .map(|r| MemorySearchResult {
                text: r.text,
                score: r.score as f64,
                path: r.path,
                line_range: r.line_range.map(|(s, e)| vec![s as u32, e as u32]),
                timestamp: r.timestamp,
                source: r.source,
            })
            .collect()
    }

    /// Retrieve content stored at `path`.
    ///
    /// `fromLine` and `numLines` apply a line-range filter (0-indexed).
    /// Returns `null` when no records exist for that path.
    #[napi]
    pub fn get(
        &self,
        path: String,
        from_line: Option<u32>,
        num_lines: Option<u32>,
    ) -> Option<String> {
        use clawhdf5_agent::openclaw::MemoryBackend;
        self.inner
            .get(&path, from_line.map(|n| n as usize), num_lines.map(|n| n as usize))
    }

    /// Store raw `content` at `path`, overwriting any previous data for that path.
    #[napi]
    pub fn write(&mut self, path: String, content: String) -> napi::Result<()> {
        use clawhdf5_agent::openclaw::MemoryBackend;
        self.inner
            .write(&path, &content)
            .map_err(napi::Error::from_reason)
    }

    /// Parse `content` as Markdown, split into sections, and ingest each section.
    ///
    /// Returns the number of sections ingested.
    #[napi]
    pub fn ingest_markdown(&mut self, path: String, content: String) -> napi::Result<u32> {
        use clawhdf5_agent::openclaw::MemoryBackend;
        self.inner
            .ingest_markdown(&path, &content)
            .map(|n| n as u32)
            .map_err(napi::Error::from_reason)
    }

    /// Reconstruct stored sections for `path` back into a Markdown string.
    #[napi]
    pub fn export_markdown(&self, path: String) -> napi::Result<String> {
        use clawhdf5_agent::openclaw::MemoryBackend;
        self.inner
            .export_markdown(&path)
            .map_err(napi::Error::from_reason)
    }

    /// Return aggregate statistics about the memory store.
    #[napi]
    pub fn stats(&self) -> BackendStats {
        use clawhdf5_agent::openclaw::MemoryBackend;
        let s = self.inner.stats();
        BackendStats {
            total_records: s.total_records as u32,
            total_embeddings: s.total_embeddings as u32,
            file_size_bytes: s.file_size_bytes as f64,
            modalities: s.modalities,
            last_updated: s.last_updated,
        }
    }

    // ── Consolidation hooks (7.6) ─────────────────────────────────────────────

    /// Remove tombstoned entries and return the number of records compacted.
    #[napi]
    pub fn compact(&mut self) -> napi::Result<u32> {
        self.inner
            .run_compaction()
            .map(|(_, n, _)| n as u32)
            .map_err(napi::Error::from_reason)
    }

    /// Apply one Hebbian decay tick to all activation weights and flush to disk.
    #[napi]
    pub fn tick_session(&mut self) -> napi::Result<()> {
        self.inner
            .tick_session()
            .map_err(napi::Error::from_reason)
    }

    /// Force a WAL merge: flush the .h5 file and truncate the WAL log.
    #[napi]
    pub fn flush_wal(&mut self) -> napi::Result<()> {
        self.inner
            .flush_wal()
            .map_err(napi::Error::from_reason)
    }

    /// Run a full compaction cycle (decay + compact + WAL flush) and return
    /// stats.  Call this at the end of an agent session for best performance.
    #[napi]
    pub fn run_consolidation(&mut self, now_secs: f64) -> napi::Result<ConsolidationStats> {
        let s = self
            .inner
            .run_consolidation(now_secs)
            .map_err(napi::Error::from_reason)?;
        Ok(ConsolidationStats {
            working_count: s.working_count as u32,
            episodic_count: s.episodic_count as u32,
            semantic_count: s.semantic_count as u32,
            total_evictions: s.total_evictions as f64,
            total_promotions: s.total_promotions as f64,
        })
    }

    // ── Metadata ─────────────────────────────────────────────────────────────

    /// Number of pending WAL entries waiting to be merged (0 if WAL disabled).
    #[napi]
    pub fn wal_pending_count(&self) -> u32 {
        self.inner.wal_pending_count() as u32
    }
}
