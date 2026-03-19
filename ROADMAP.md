# ClawhDF5 Roadmap — Agent Memory Evolution

> Making clawhdf5 the defacto agentic memory solution.
> Single file. Pure Rust. Zero dependencies. Trusted everywhere.

---

## Track 1: Knowledge Graph in HDF5
**Status:** 🟢 Phase 1 Complete
**Priority:** Critical
**Crate:** `clawhdf5-agent`

- [x] **1.1** Entity storage — entities with properties, embeddings, timestamps (created_at/updated_at)
- [x] **1.2** Relation storage — typed edges with RelationType enum (Temporal/Causal/Associative/Hierarchical/Custom), metadata, timestamps
- [ ] **1.3** Entity extraction helpers — extract entities from text chunks on save
- [x] **1.4** Entity resolution — fuzzy name matching (Levenshtein distance) via resolve_or_create()
- [x] **1.5** Graph traversal queries — BFS neighbors with depth, subgraph extraction from seeds
- [x] **1.6** Spreading activation — weighted activation propagation with configurable decay
- [x] **1.7** Graph-aware retrieval — get_entity_context() for formatted context injection
- [x] **1.8** Tests — comprehensive tests for all new features

**Research:** Graph-Native Cognitive Memory (2026), Graph-based Agent Memory survey (2026), SYNAPSE (2025)

---

## Track 2: Memory Consolidation Engine
**Status:** 🟢 Phase 1 Complete
**Priority:** Critical
**Crate:** `clawhdf5-agent`

- [x] **2.1** Importance scoring — surprise (novelty), correction boost, length scoring with configurable weights
- [x] **2.2** Three-tier memory model — Working → Episodic → Semantic with bounded capacities
- [x] **2.3** Time-decay with reactivation — exponential decay with configurable half-life, access resets timestamp
- [x] **2.4** Bounded memory with graceful degradation — evict lowest-decay entries when over capacity
- [x] **2.5** Consolidation cycles — promote/evict across tiers based on importance and access thresholds
- [x] **2.6** Memory statistics — ConsolidationStats with per-tier counts, eviction/promotion tracking
- [x] **2.7** Tests — comprehensive tests for all features

**Research:** CraniMem (2026), D-MEM (2026), AI Hippocampus survey (2026)

---

## Track 3: Hybrid Retrieval Pipeline
**Status:** 🟢 Phase 1 Complete
**Priority:** High
**Crate:** `clawhdf5-agent`

- [x] **3.1** Reciprocal Rank Fusion (RRF) — rrf_hybrid_search() with k=60 constant
- [x] **3.2** Multi-factor re-ranking — temporal decay, source authority hierarchy, activation scores (reranker.rs)
- [x] **3.3** Low-confidence rejection — min_score threshold, gap filtering, max_results (confidence.rs)
- [ ] **3.4** Query expansion — generate related queries for broader recall
- [x] **3.5** Result explanation — ReRankResult with full score breakdown per factor
- [x] **3.6** Configurable pipeline — ReRankConfig + ConfidenceConfig with tunable weights/thresholds
- [ ] **3.7** Tests + MemX-comparable benchmarks

**Research:** MemX (2026), SwiftMem (2026)

---

## Track 4: Temporal Reasoning
**Status:** 🟢 Phase 1 Complete
**Priority:** High
**Crate:** `clawhdf5-agent`

- [x] **4.1** Temporal index — sorted timestamp index with binary search, insert/remove
- [x] **4.2** Time-range queries — range_query, before, after, latest, earliest
- [x] **4.3** Session DAG — parent/child linking, chain walking, time-range overlap queries
- [x] **4.4** Temporal re-ranking — query hint enum (Latest/Earliest/Around/Between/None) with boost scoring
- [x] **4.5** Temporal entity tracking — EntityTimeline with state change history + point-in-time reconstruction
- [x] **4.6** Tests — comprehensive tests for all features

**Research:** MemX temporal gaps (≤43.6% Hit@5), MemoryArena multi-session tasks (2026)

---

## Track 5: Memory Security & Provenance
**Status:** 🟢 Phase 1 Complete
**Priority:** Medium-High
**Crate:** `clawhdf5-agent`

- [x] **5.1** Source attribution — MemoryProvenance with source, creator, session, FNV-1a content hash
- [x] **5.2** Write anomaly detection — rate limiting, 15 injection patterns, source distribution analysis
- [x] **5.3** Source isolation — per-MemorySource sub-stores preventing cross-contamination
- [x] **5.4** Memory integrity verification — content hash comparison via verify_integrity()
- [x] **5.5** Poisoning resistance — pattern detection for prompt injection attempts
- [x] **5.6** Tests — comprehensive tests including adversarial patterns

**Research:** MemoryGraft (2025), SSGM Framework (2026)

---

## Track 6: Multi-Modal Memory
**Status:** 🟢 Phase 1 Complete
**Priority:** Medium
**Crate:** `clawhdf5-agent`

- [x] **6.1** Image embedding storage — ModalEmbedding with model provenance (CLIP, SigLIP, etc.)
- [x] **6.2** Audio fingerprints — Audio modality with embedding storage
- [x] **6.3** Multi-modal search — search_by_modality (filtered) + search_cross_modal (all embeddings)
- [x] **6.4** Observation records — raw perception vs interpretation with confidence scoring
- [x] **6.5** Media reference storage — MediaRef with Path/Url/Inline, MIME types, FNV-1a checksums
- [x] **6.6** Tests — 35 comprehensive tests

**Research:** Neuro-Symbolic Memory (2026), RAGdb multi-modal RAG (2025)

---

## Track 7: OpenClaw Integration
**Status:** 🔴 Not Started
**Priority:** Critical (for adoption)
**Crate:** new `clawhdf5-openclaw` or integration in zeroclaw

- [ ] **7.1** Memory backend trait — implement OpenClaw's memory interface with clawhdf5
- [ ] **7.2** Replace sqlite-vec — clawhdf5 vector search as drop-in replacement
- [ ] **7.3** Markdown import/export — bidirectional sync between .md files and .brain HDF5
- [ ] **7.4** memory_search tool — backed by clawhdf5 hybrid retrieval pipeline
- [ ] **7.5** memory_get tool — read specific memory records from HDF5
- [ ] **7.6** Compaction integration — clawhdf5 consolidation hooks into OpenClaw session compaction
- [ ] **7.7** Config surface — `memory.backend = "clawhdf5"` in openclaw.json
- [ ] **7.8** Documentation + migration guide

**Dependency:** Tracks 1-4 should be substantially complete first

---

## Track 8: Benchmarking & Validation
**Status:** 🔴 Not Started
**Priority:** High
**Crate:** workspace-level benchmarks

- [ ] **8.1** MemoryArena benchmark — run against Choi/Pentland's multi-session eval
- [ ] **8.2** LongMemEval benchmark — compare against MemX's numbers
- [ ] **8.3** Latency benchmarks — search latency at 1K, 10K, 100K, 1M records
- [ ] **8.4** Memory footprint — measure HDF5 file size vs record count
- [ ] **8.5** Consolidation efficiency — measure retrieval quality before/after consolidation
- [ ] **8.6** Cross-platform benchmarks — x86, ARM, Android, WASM
- [ ] **8.7** Published results in README + benchmark reports

---

## Implementation Order

**Phase 1 (Current Sprint):** Tracks 1, 2, 3 — core memory intelligence
**Phase 2:** Track 4 (temporal) + Track 5 (security)
**Phase 3:** Track 6 (multi-modal) + Track 7 (OpenClaw integration)
**Phase 4:** Track 8 (benchmarking + validation)

---

_Last updated: 2026-03-19_
