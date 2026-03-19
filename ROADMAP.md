# ClawhDF5 Roadmap — Agent Memory Evolution

> Making clawhdf5 the defacto agentic memory solution.
> Single file. Pure Rust. Zero dependencies. Trusted everywhere.

---

## Track 1: Knowledge Graph in HDF5
**Status:** 🔴 Not Started
**Priority:** Critical
**Crate:** `clawhdf5-agent`

- [ ] **1.1** Entity storage — HDF5 compound datasets for entities (id, type, name, properties, embeddings, timestamps)
- [ ] **1.2** Relation storage — typed edges as compound datasets (src, dst, relation_type, weight, metadata, timestamp)
- [ ] **1.3** Entity extraction helpers — extract entities from text chunks on save
- [ ] **1.4** Entity resolution — merge duplicate entities (fuzzy name matching + embedding similarity)
- [ ] **1.5** Graph traversal queries — BFS/DFS neighbors, shortest path, subgraph extraction
- [ ] **1.6** Spreading activation — SYNAPSE-style associative recall from seed entities
- [ ] **1.7** Graph-aware retrieval — combine graph context with vector search results
- [ ] **1.8** Tests + benchmarks

**Research:** Graph-Native Cognitive Memory (2026), Graph-based Agent Memory survey (2026), SYNAPSE (2025)

---

## Track 2: Memory Consolidation Engine
**Status:** 🔴 Not Started
**Priority:** Critical
**Crate:** `clawhdf5-agent`

- [ ] **2.1** Importance scoring — reward prediction error + surprise-based gating (D-MEM inspired)
- [ ] **2.2** Three-tier memory model — working → episodic → semantic (hippocampal consolidation)
- [ ] **2.3** Time-decay with reactivation — memories that get retrieved stay fresh, unused ones fade
- [ ] **2.4** Bounded memory with graceful degradation — hard limits per tier, oldest/lowest-importance evicted
- [ ] **2.5** Consolidation scheduler — background process that promotes/demotes/merges memories
- [ ] **2.6** Memory statistics — track access patterns, hit rates, consolidation metrics
- [ ] **2.7** Tests + benchmarks

**Research:** CraniMem (2026), D-MEM (2026), AI Hippocampus survey (2026)

---

## Track 3: Hybrid Retrieval Pipeline
**Status:** 🟡 Partial (have hybrid.rs, bm25.rs, vector_search.rs)
**Priority:** High
**Crate:** `clawhdf5-agent`

- [ ] **3.1** Reciprocal Rank Fusion (RRF) — proper RRF implementation replacing naive score fusion
- [ ] **3.2** Multi-factor re-ranking — temporal recency, entity relevance, source authority, semantic similarity
- [ ] **3.3** Low-confidence rejection — suppress spurious recalls when no good match exists
- [ ] **3.4** Query expansion — generate related queries for broader recall
- [ ] **3.5** Result explanation — annotate why each result was returned (which factors scored high)
- [ ] **3.6** Configurable pipeline — let callers pick which stages to run
- [ ] **3.7** Tests + MemX-comparable benchmarks

**Research:** MemX (2026), SwiftMem (2026)

---

## Track 4: Temporal Reasoning
**Status:** 🔴 Not Started
**Priority:** High
**Crate:** `clawhdf5-agent`

- [ ] **4.1** Temporal index — timestamp-based B-tree/sorted index over memory records
- [ ] **4.2** Time-range queries — "what happened between X and Y"
- [ ] **4.3** Session DAG — link sessions with parent/child/continuation relationships
- [ ] **4.4** Temporal re-ranking — boost recent memories, handle "latest" / "first" queries
- [ ] **4.5** Temporal entity tracking — track entity state changes over time
- [ ] **4.6** Tests + temporal reasoning benchmarks

**Research:** MemX temporal gaps (≤43.6% Hit@5), MemoryArena multi-session tasks (2026)

---

## Track 5: Memory Security & Provenance
**Status:** 🟡 Partial (HDF5 has SHINES provenance)
**Priority:** Medium-High
**Crate:** `clawhdf5-agent` + `clawhdf5-format`

- [ ] **5.1** Source attribution — every memory record tags its source (user, system, tool, retrieval)
- [ ] **5.2** Write anomaly detection — flag unusual memory write patterns
- [ ] **5.3** Source isolation — separate user-provided memories from system-derived ones
- [ ] **5.4** Memory integrity verification — periodic hash checks on memory datasets
- [ ] **5.5** Poisoning resistance — validate retrieved memories against provenance before use
- [ ] **5.6** Tests + adversarial memory injection tests

**Research:** MemoryGraft (2025), SSGM Framework (2026)

---

## Track 6: Multi-Modal Memory
**Status:** 🔴 Not Started
**Priority:** Medium
**Crate:** `clawhdf5-agent`

- [ ] **6.1** Image embedding storage — store CLIP/SigLIP embeddings alongside text
- [ ] **6.2** Audio fingerprints — store audio embeddings for voice/sound memories
- [ ] **6.3** Multi-modal search — query across text + image + audio embeddings
- [ ] **6.4** Observation records — structured "what agent saw" vs "what agent concluded"
- [ ] **6.5** Media reference storage — store references to large media (paths/URLs) with embeddings
- [ ] **6.6** Tests + multi-modal retrieval benchmarks

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
