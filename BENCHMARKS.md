# ClawhDF5 Benchmark Results

> Pure Rust. Zero C dependencies. Single file. Fast enough to forget it's there.

**System:** Intel i7-12650H (10C/16T, 4.7 GHz boost) · 32 GB DDR5 · Linux 6.8.0  
**Rust:** 1.96.0-nightly (2026-03-14) · `--release` profile  
**Date:** 2026-03-19

---

## Vector Search Latency

Brute-force cosine similarity over 384-dimensional embeddings (OpenAI text-embedding-3-small size).

| Scale | Flat Search | Pre-norm | IVF (nprobe=10) | IVF-PQ | RAIRS |
|-------|-------------|----------|-----------------|--------|-------|
| **1K** | 54 µs | 62 µs | — | — | — |
| **10K** | 753 µs | 706 µs | 27 µs | — | 159 µs |
| **100K** | 11.4 ms | — | 1.32 ms | 1.19 ms | — |

**Key insight:** At 10K records (typical agent memory), IVF search delivers **27 µs** — that's 26x faster than flat search. Even at 100K records, IVF-PQ keeps search under **1.2 ms**.

### Comparison to MemX (arxiv:2603.16171)

MemX claims end-to-end search under 90ms at 100K records (Rust + libSQL + FTS5).

| Metric | MemX (claimed) | ClawhDF5 | Speedup |
|--------|----------------|----------|---------|
| 100K flat search | <90 ms | 11.4 ms | **~8x** |
| 100K IVF-PQ search | — | 1.19 ms | **~76x** |
| Keyword search 10K | 1,100x improvement over unindexed | 583 µs (BM25) | Comparable |

---

## SIMD & Parallelism

384-dimensional cosine similarity at 10K scale.

| Strategy | Latency | vs Sequential |
|----------|---------|---------------|
| Sequential (scalar) | 1.07 ms | 1.0x |
| SIMD (auto-vectorized) | 545 µs | **2.0x** |
| Rayon (parallel) | 553 µs | **1.9x** |
| Adaptive (auto-select) | 564 µs | **1.9x** |

At 100K:
| Strategy | Latency |
|----------|---------|
| SIMD | 13.7 ms |
| Rayon parallel | 8.3 ms |

---

## Hybrid Search (Vector + BM25)

1K records, 384-dimensional embeddings with BM25 keyword index.

| Method | Latency | Notes |
|--------|---------|-------|
| Weighted fusion | 198 µs | Original min-max normalization |
| **RRF (k=60)** | **222 µs** | Reciprocal Rank Fusion — better quality, ~12% overhead |
| BM25-only 1K | 67 µs | Keyword search alone |
| Hybrid 10K | 2.04 ms | Full hybrid at 10K scale |

---

## Knowledge Graph

Graph traversal and entity operations.

| Operation | Scale | Latency |
|-----------|-------|---------|
| BFS traversal | 100 entities | 5.4 µs |
| BFS traversal | 1,000 entities | 24 µs |
| Spreading activation | 100 entities | 16.9 µs |
| Entity resolution (Levenshtein) | 100 entities | 64 µs |
| Alias resolution (short query) | 100 aliases | 10.4 µs |
| Alias resolution (long query) | 100 aliases | 11.6 µs |

**All graph operations complete in microseconds.** Spreading activation across 100 entities with 5 propagation steps finishes in 17 µs.

---

## Memory Consolidation

Hippocampal-inspired tiered memory management.

| Operation | Scale | Latency |
|-----------|-------|---------|
| Consolidation cycle | 100 records | 15 µs |
| Consolidation cycle | 1,000 records | 164 µs |
| Importance scoring | 100 records | 25 µs |

A full consolidation pass over 1,000 memories (eviction + promotion across Working → Episodic → Semantic) completes in **164 µs**. This can run on every memory write without perceptible latency.

---

## Temporal Index

Sorted timestamp index with binary search.

| Operation | Scale | Latency |
|-----------|-------|---------|
| Range query | 10K timestamps | **716 ns** |
| Batch insert | 10K timestamps | 4.69 ms |

Sub-microsecond temporal queries. "What happened between 3pm and 5pm?" over 10K records: **716 nanoseconds.**

---

## Write Path

HDF5 persistence with optional Write-Ahead Log.

| Operation | Latency | Notes |
|-----------|---------|-------|
| Single save (no WAL) | 91 µs | Direct HDF5 write |
| Single save (with WAL) | 134 µs | +47% for crash safety |
| Batch 100 | 723 µs | 7.2 µs per record |
| Batch 1,000 | 6.17 ms | 6.2 µs per record |
| WAL save (1K existing) | 539 µs | Incremental append |
| WAL flush 100 entries | 787 µs | Merge WAL → HDF5 |
| Session tick 1K | 5.76 ms | Full session maintenance |
| Session tick 10K | 89.8 ms | Background operation |

---

## Decision Gate

Trivial/non-trivial classification for memory write filtering.

| Check | Latency |
|-------|---------|
| Trivial skip ("ok", "yes") | 61 ns |
| Short phrase skip | 86 ns |
| Non-trivial pass | 705 ns |
| Ratio check | 488 ns |

**Sub-microsecond filtering.** The gate decides whether to save a memory in under 1 µs.

---

## Memory Strategy

End-to-end strategy evaluation including embedding operations.

| Strategy | Condition | Latency |
|----------|-----------|---------|
| SaveEveryExchange (substantive) | Saves | 923 ns |
| SaveEveryExchange (trivial) | Skips | 67 ns |
| SaveOnSemanticShift (empty store) | Saves | 941 ns |

---

## Summary

| Capability | Typical Latency | Scale |
|------------|----------------|-------|
| **Full memory search** | <1 ms | 10K records |
| **Hybrid vector+keyword** | <200 µs | 1K records |
| **Knowledge graph query** | <25 µs | 1K entities |
| **Temporal range query** | <1 µs | 10K timestamps |
| **Memory write** | <135 µs | Per record |
| **Consolidation cycle** | <165 µs | 1K records |
| **Importance gate** | <1 µs | Per record |

**The entire memory pipeline — search, retrieve, re-rank, filter — runs in single-digit milliseconds at agent-typical scales. Fast enough that memory becomes invisible infrastructure.**

---

_Benchmarks generated with Criterion.rs (50-100 samples per benchmark). Results may vary by hardware._
