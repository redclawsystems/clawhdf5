# ClawHDF5 — Pure-Rust HDF5 + AI Agent Memory

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![CI](https://img.shields.io/badge/CI-coming%20soon-lightgrey)
![crates.io](https://img.shields.io/badge/crates.io-coming%20soon-lightgrey)
![docs.rs](https://img.shields.io/badge/docs.rs-coming%20soon-lightgrey)

ClawHDF5 is the unified successor to [rustyhdf5](https://github.com/rustystack/rustyhdf5) and [edgehdf5](https://github.com/rustystack/edgehdf5). It combines a pure-Rust HDF5 reader/writer (zero C dependencies) with an HDF5-backed persistent memory store for on-device AI agents — all in a single workspace.

```
redclawsystems/clawhdf5          MIT License
Rust workspace · 15 crates
```

---

## What's in This Repo

### Core HDF5 Crates (from rustyhdf5)

| Crate | Description |
|---|---|
| `clawhdf5-format` | Binary format parsing and writing (`no_std` compatible core) |
| `clawhdf5-types` | HDF5 type system definitions (bottom layer, no deps) |
| `clawhdf5-io` | I/O abstraction layer (buffered, mmap, async, HSDS, VOL, sub-filing) |
| `clawhdf5-filters` | Filter/compression pipeline (deflate, shuffle, fletcher32, lz4) |
| `clawhdf5-derive` | Proc macros for deriving HDF5 traits |
| `clawhdf5` | High-level ergonomic API for reading and writing HDF5 files |
| `clawhdf5-netcdf4` | NetCDF-4 read support — pure Rust, no C dependencies |
| `clawhdf5-ann` | HNSW approximate nearest-neighbor index stored in HDF5 |
| `clawhdf5-accel` | SIMD acceleration (NEON, AVX2, AVX-512) |
| `clawhdf5-gpu` | GPU compute via wgpu (Metal/Vulkan/DX12) |
| `clawhdf5-py` | Python bindings via PyO3 |

### Agent Memory Crates (from edgehdf5)

| Crate | Description |
|---|---|
| `clawhdf5-agent` | HDF5-backed persistent memory store for on-device AI agents |
| `clawhdf5-migrate` | CLI to migrate SQLite agent memory databases to HDF5 |
| `clawhdf5-android` | Android JNI bridge for clawhdf5-agent |
| `clawhdf5-cli` | CLI for agent memory — create, save, search, recall, stats |

---

## Quick Start

### HDF5 (high-level API)

```toml
[dependencies]
clawhdf5 = "2.0"
```

```rust
use clawhdf5::{File, FileBuilder, AttrValue};

// Write
let mut builder = FileBuilder::new();
builder.create_dataset("temperatures")
    .with_f64_data(&[22.5, 23.1, 21.8])
    .with_shape(&[3]);
builder.set_attr("version", AttrValue::I64(1));
builder.write("output.h5").unwrap();

// Read
let file = File::open("output.h5").unwrap();
let ds = file.dataset("temperatures").unwrap();
let values = ds.read_f64().unwrap();
assert_eq!(values, vec![22.5, 23.1, 21.8]);
```

### AI Agent Memory

Enable the `agent` feature to pull in the full agent memory layer:

```toml
[dependencies]
clawhdf5-agent = { version = "2.0", features = ["agent", "float16"] }
```

```rust
use clawhdf5_agent::{HDF5Memory, MemoryConfig, MemoryEntry, AgentMemory};

let config = MemoryConfig {
    path: "agent_memory.h5".into(),
    agent_id: "my-agent".into(),
    embedder: "openai:text-embedding-3-small".into(),
    embedding_dim: 384,
    float16: true,
    compression: true,
    ..Default::default()
};

let mut memory = HDF5Memory::create(config)?;

let entry = MemoryEntry {
    chunk: "The user prefers dark mode and uses vim keybindings.".into(),
    embedding: embed("The user prefers dark mode..."),
    source_channel: "chat".into(),
    timestamp: 1700000000.0,
    session_id: "session-001".into(),
    tags: "preference,ui".into(),
};

let id = memory.save(entry)?;
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Applications                               │
├───────────────┬─────────────────────────────────────────────────────┤
│  clawhdf5-cli │  clawhdf5-py (PyO3)  │  clawhdf5-netcdf4            │
│  clawhdf5-    │  clawhdf5-ann        │  clawhdf5-android (JNI)      │
│  migrate      │                      │                              │
├───────────────┴──────────────────────┴──────────────────────────────┤
│                 clawhdf5-agent (AI agent memory layer)              │
│       sessions · WAL · BM25 · IVF-PQ · hybrid search · kg          │
├─────────────────────────────────────────────────────────────────────┤
│                   clawhdf5 (high-level HDF5 API)                    │
├──────────────────┬──────────────────────────────────────────────────┤
│  clawhdf5-io     │  clawhdf5-filters  │  clawhdf5-derive (macros)   │
├──────────────────┴──────────────────────────────────────────────────┤
│          clawhdf5-accel (SIMD)  ·  clawhdf5-gpu (wgpu)              │
├─────────────────────────────────────────────────────────────────────┤
│              clawhdf5-format (no_std core, zero C deps)             │
├─────────────────────────────────────────────────────────────────────┤
│              clawhdf5-types  (type definitions)                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   agent_memory.h5  │
                    │  /meta             │
                    │  /memory           │
                    │  /sessions         │
                    │  /knowledge_graph  │
                    └───────────────────┘
```

---

## Performance Benchmarks

### HDF5 I/O (clawhdf5 core)

Benchmarked on Apple MacBook M3 Max. Compared against h5py 3.14 / C HDF5 1.14.6.

| Operation | clawhdf5 | h5py (C HDF5) | Result |
|---|---|---|---|
| Metadata (parse superblock) | 19 ns | 2,080 µs | **308× faster** |
| Contiguous write (1M f64) | 0.82 ms | 1.60 ms | **2× faster** |
| Contiguous read (1M f64) | 0.28 ms | 0.65 ms | **2.3× faster** |
| Chunked read (1M f64, 100 chunks) | 0.34 ms | 0.86 ms | **2.5× faster** |
| Compressed write (deflate, 1M f64) | 172 ms | 344 ms | **2× faster** |
| Compressed read (deflate, 1M f64) | 6.95 ms | ~6.4 ms | **~parity** |
| Zero-copy read (1M f64, mmap) | 313 ns | N/A | **~2,000× faster** |
| File open (mmap vs buffered) | 19 µs | 472 µs | **25× faster** |

### Agent Memory Search (clawhdf5-agent)

Benchmarked on MacBook Pro M3 Max, 384-dimensional embeddings:

| Backend | 1K vectors | 10K vectors | 100K vectors | Notes |
|---|---|---|---|---|
| Scalar (baseline) | 42µs | 410µs | 4.1ms | No SIMD, no dependencies |
| SIMD brute-force | 18µs | 175µs | 1.7ms | clawhdf5-accel auto-dispatch |
| Apple Accelerate (cblas) | 15µs | **157µs** | 1.5ms | AMX coprocessor |
| BLAS (matrixmultiply) | 17µs | 168µs | 1.6ms | Cross-platform |
| Rayon parallel | 35µs | 120µs | 980µs | Scales with core count |
| GPU (wgpu) | 200µs | 190µs | 650µs | Wins at scale |
| **IVF-PQ** | N/A | 850µs | **380µs** | **6.2× faster than numpy** |

---

## Feature Flags

### `clawhdf5` (high-level HDF5 crate)

| Flag | Default | Description |
|---|---|---|
| `mmap` | yes | Memory-mapped file I/O for zero-copy reads |
| `fast-deflate` | yes | Use zlib-ng for faster deflate |
| `parallel` | no | Parallel chunk I/O via rayon |

### `clawhdf5-agent` (AI agent memory)

| Feature | Default | Description |
|---|---|---|
| `agent` | no | Enables AI agent memory layer (sessions, WAL, BM25, search) |
| `float16` | **yes** | Half-precision embedding storage |
| `parallel` | no | Rayon-based parallel search |
| `fast-math` | no | BLAS matrix-vector multiply (cross-platform) |
| `accelerate` | no | Apple Accelerate — cblas_sgemv on AMX (macOS only) |
| `openblas` | no | OpenBLAS cblas_sgemv (Linux) |
| `gpu` | no | GPU-accelerated search via wgpu |
| `async` | no | Tokio async wrapper with background flush |

### `clawhdf5-format` (low-level format crate)

| Flag | Default | Description |
|---|---|---|
| `std` | yes | Standard library support (disable for `no_std`) |
| `deflate` | yes | Deflate compression support |
| `checksum` | yes | Jenkins lookup3 checksum verification |
| `provenance` | yes | SHA-256 provenance attributes |
| `parallel` | no | Parallel chunk encoding via rayon |
| `fast-checksum` | no | CRC32 acceleration via `crc32fast` |
| `fast-deflate` | no | zlib-ng backend for deflate |

---

## HDF5 File Schema (agent memory, v1.0)

```
agent_memory.h5
├── /meta                          (attributes)
│   ├── schema_version: "1.0"
│   ├── clawhdf5_version: "2.0.0"
│   ├── agent_id, embedder, embedding_dim
│   └── created_at
├── /memory
│   ├── chunks:          string[N]
│   ├── embeddings:      f32[N × D]
│   ├── tombstones:      u8[N]           (0=active, 1=deleted)
│   └── norms:           f32[N]          (pre-computed L2 norms)
├── /sessions
│   ├── ids:             string[S]
│   └── summaries:       string[S]
└── /knowledge_graph
    ├── entity_names:     string[E]
    ├── relation_srcs:    i64[R]
    ├── relation_tgts:    i64[R]
    └── relation_types:   string[R]
```

---

## Migration from SQLite

The `clawhdf5-migrate` CLI converts existing SQLite agent memory databases to HDF5:

```bash
cargo install --path crates/clawhdf5-migrate

clawhdf5-migrate \
  --sqlite old_memory.db \
  --hdf5 agent_memory.h5 \
  --agent-id my-agent \
  --embedder openai:text-embedding-3-small \
  --embedding-dim 384 \
  --compression --verbose
```

---

## Building

```bash
# Default build
cargo build --workspace

# With Apple Accelerate (macOS)
cargo build -p clawhdf5-agent --features "float16,accelerate,parallel"

# All features (macOS)
cargo build -p clawhdf5-agent --features "float16,accelerate,parallel,gpu,async"

# Run all tests
cargo test --workspace

# Run benchmarks
cargo bench -p clawhdf5-agent
cargo bench -p clawhdf5-format
```

---

## Migration from rustyhdf5 / edgehdf5

| Old crate | New crate |
|---|---|
| `rustyhdf5` | `clawhdf5` |
| `rustyhdf5-format` | `clawhdf5-format` |
| `rustyhdf5-types` | `clawhdf5-types` |
| `rustyhdf5-io` | `clawhdf5-io` |
| `rustyhdf5-filters` | `clawhdf5-filters` |
| `rustyhdf5-derive` | `clawhdf5-derive` |
| `rustyhdf5-netcdf4` | `clawhdf5-netcdf4` |
| `rustyhdf5-ann` | `clawhdf5-ann` |
| `rustyhdf5-accel` | `clawhdf5-accel` |
| `rustyhdf5-gpu` | `clawhdf5-gpu` |
| `rustyhdf5-py` | `clawhdf5-py` |
| `edgehdf5-memory` | `clawhdf5-agent` |
| `edgehdf5-migrate` | `clawhdf5-migrate` |
| `edgehdf5-android-bridge` | `clawhdf5-android` |
| `edgehdf5-cli` | `clawhdf5-cli` |

In Rust source, replace `use rustyhdf5_*` → `use clawhdf5_*` and `use edgehdf5_memory` → `use clawhdf5_agent`.

---

## License

MIT
