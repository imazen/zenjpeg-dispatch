# Plan: `coefficient` - Distributed Codec Benchmarking Framework

## Overview

A new crate for scalable, reproducible codec benchmarking with:
- **Horizontal scaling**: Thousands of parallel workers, no coordination bottleneck
- **Complete provenance**: Every result traceable to exact code version + config
- **Flexible versioning**: Version mappings can be updated without re-running work
- **Syncable storage**: Works locally, syncs to GCS, runs distributed

## Name Rationale

"coefficient" references:
- DCT coefficients (core of JPEG encoding)
- Coefficients of determination (measurement/comparison)
- Short, memorable, available on crates.io

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STORED DATA                               │
│  (Immutable, complete provenance, audit trail)                  │
│                                                                  │
│  EncodingRecord: source + codec + commit + config + results     │
│  MetricRecord: encoding + algorithm + commit + config + value   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     VERSION MAPPING                              │
│  (Mutable rules, can be updated without touching data)          │
│                                                                  │
│  CodecVersionDef: "mozjpeg-420-v5" = commits X,Y + config {...} │
│  MetricVersionDef: "ssim2-gpu-v1" = impl + version constraint   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      TASK PLANNING                               │
│  (What work is missing for current version definitions?)        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       REPORTING                                  │
│  (Query/filter/group by versions, compare across versions)      │
└─────────────────────────────────────────────────────────────────┘
```

## Crate Structure

```
coefficient/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public API re-exports
│   │
│   ├── model/                    # Core data structures
│   │   ├── mod.rs
│   │   ├── source.rs             # SourceRecord (image metadata)
│   │   ├── encoding.rs           # EncodingRecord (full provenance)
│   │   ├── metric.rs             # MetricRecord (full provenance)
│   │   └── task.rs               # EncodingTask, MetricTask
│   │
│   ├── version/                  # Version mapping layer
│   │   ├── mod.rs
│   │   ├── codec.rs              # CodecVersionDef, matching rules
│   │   ├── metric.rs             # MetricVersionDef, matching rules
│   │   └── registry.rs           # Load/save version definitions
│   │
│   ├── store/                    # Storage backends
│   │   ├── mod.rs                # ResultStore trait
│   │   ├── layout.rs             # Path/key conventions
│   │   ├── local.rs              # Filesystem backend
│   │   ├── gcs.rs                # Google Cloud Storage (feature-gated)
│   │   ├── memory.rs             # In-memory for testing
│   │   └── sync.rs               # Sync between backends
│   │
│   ├── codec/                    # Codec trait and implementations
│   │   ├── mod.rs                # Codec trait
│   │   ├── provenance.rs         # Build-time version capture
│   │   ├── mozjpeg.rs            # mozjpeg-oxide wrapper
│   │   ├── jpegli.rs             # jpegli-rs wrapper
│   │   ├── avif.rs               # ravif wrapper (feature-gated)
│   │   └── webp.rs               # webp wrapper (feature-gated)
│   │
│   ├── metric/                   # Metric trait and implementations
│   │   ├── mod.rs                # MetricAlgorithm trait
│   │   ├── provenance.rs         # Build-time version capture
│   │   ├── ssimulacra2.rs        # CPU + GPU variants
│   │   ├── butteraugli.rs        # CPU + GPU variants
│   │   └── dssim.rs              # DSSIM implementation
│   │
│   ├── planner/                  # Work planning
│   │   ├── mod.rs
│   │   ├── manifest.rs           # JobManifest structure
│   │   ├── remaining.rs          # Find incomplete work
│   │   └── distribution.rs       # Split work across workers
│   │
│   ├── worker/                   # Task execution
│   │   ├── mod.rs
│   │   ├── encoding.rs           # Process encoding tasks
│   │   ├── metric.rs             # Process metric tasks
│   │   └── batch.rs              # GCP Batch worker entry
│   │
│   ├── analysis/                 # Querying and reporting
│   │   ├── mod.rs
│   │   ├── query.rs              # Version-aware queries
│   │   ├── aggregator.rs         # Build indices from results
│   │   ├── pareto.rs             # Pareto frontier analysis
│   │   └── export.rs             # CSV, Parquet, JSON export
│   │
│   └── util/                     # Utilities
│       ├── mod.rs
│       ├── hash.rs               # SHA256 helpers
│       └── image.rs              # PNG decode, RGB handling
│
├── bin/
│   ├── coefficient.rs            # Main CLI (local execution)
│   ├── coefficient-worker.rs     # GCP Batch worker entry
│   └── coefficient-aggregate.rs  # Post-batch aggregation
│
├── examples/
│   ├── simple_benchmark.rs       # Basic usage
│   ├── version_comparison.rs     # Compare codec versions
│   └── distributed_job.rs        # GCP Batch submission
│
└── tests/
    ├── model_tests.rs
    ├── version_matching_tests.rs
    ├── store_tests.rs
    └── integration/
```

## Core Data Models

### SourceRecord
```rust
pub struct SourceRecord {
    pub hash: String,              // SHA256 of RGB pixels
    pub name: String,              // Original filename
    pub corpus: String,            // "cid22", "clic2025", etc.
    pub width: u32,
    pub height: u32,
    pub pixels: u64,

    // Pre-computed analysis for stratification
    pub variance: Option<f32>,
    pub edge_density: Option<f32>,
    pub chroma_complexity: Option<f32>,

    pub created_at: DateTime<Utc>,
}
```

### EncodingRecord
```rust
pub struct EncodingRecord {
    pub id: String,                // hash(source_hash, version_id, quality)
    pub source_hash: String,
    pub quality: u8,

    // === Complete codec provenance ===
    pub codec_name: String,        // "mozjpeg", "jpegli", "avif", "webp"
    pub codec_crate: String,       // "mozjpeg-oxide", "jpegli-rs"
    pub codec_crate_version: String,
    pub codec_commit: Option<String>,
    pub codec_config: serde_json::Value,  // Full config struct

    // === Results ===
    pub blob_hash: String,         // SHA256 of encoded bytes
    pub size_bytes: usize,
    pub encode_time_ms: u64,

    // === Provenance ===
    pub worker_id: String,
    pub created_at: DateTime<Utc>,
}
```

### MetricRecord
```rust
pub struct MetricRecord {
    pub id: String,                // "{encoding_id}:{metric_version_id}"
    pub encoding_id: String,

    // === Complete metric provenance ===
    pub metric_name: String,       // "ssimulacra2", "butteraugli", "dssim"
    pub implementation: String,    // "fast-ssim2", "turbo-metrics-gpu"
    pub impl_crate: String,
    pub impl_version: String,
    pub impl_commit: Option<String>,
    pub impl_config: serde_json::Value,

    // === Results ===
    pub value: f64,
    pub compute_time_ms: u64,

    // === Provenance ===
    pub worker_id: String,
    pub computed_at: DateTime<Utc>,
}
```

## Version Mapping

### CodecVersionDef
```rust
pub struct CodecVersionDef {
    pub version_id: String,        // "mozjpeg-420-v5"
    pub codec_name: String,
    pub rules: MatchRules,
}

pub struct MatchRules {
    pub crate_version: Option<VersionReq>,  // ">=0.5.0"
    pub valid_commits: Vec<String>,          // Empty = any
    pub required_config: serde_json::Value,  // Fields that must match
}
```

### MetricVersionDef
```rust
pub struct MetricVersionDef {
    pub version_id: String,        // "ssim2-cpu-v2"
    pub metric_name: String,
    pub implementation: String,
    pub rules: MatchRules,
}
```

## Storage Layout

```
{store}/
├── sources/
│   └── {source_hash}.png
│
├── encodings/
│   └── {encoding_id}/
│       ├── record.json           # EncodingRecord
│       └── data.{ext}            # Encoded bytes
│
├── metrics/
│   └── {encoding_id}/
│       └── {metric_version_id}.json  # MetricRecord
│
├── manifests/
│   └── {job_id}.json             # JobManifest
│
├── versions/
│   ├── codecs.json               # Vec<CodecVersionDef>
│   └── metrics.json              # Vec<MetricVersionDef>
│
└── indices/                      # Built by aggregator
    ├── results.parquet
    └── results.db                # SQLite for queries
```

## Key Traits

### ResultStore
```rust
#[async_trait]
pub trait ResultStore: Send + Sync + Clone {
    // Sources
    async fn source_exists(&self, hash: &str) -> Result<bool>;
    async fn source_get(&self, hash: &str) -> Result<SourceRecord>;
    async fn source_put(&self, record: &SourceRecord, data: &[u8]) -> Result<()>;

    // Encodings
    async fn encoding_exists(&self, id: &str) -> Result<bool>;
    async fn encoding_get(&self, id: &str) -> Result<EncodingRecord>;
    async fn encoding_get_blob(&self, id: &str) -> Result<Vec<u8>>;
    async fn encoding_put(&self, record: &EncodingRecord, data: &[u8]) -> Result<()>;

    // Metrics
    async fn metric_exists(&self, encoding_id: &str, version_id: &str) -> Result<bool>;
    async fn metric_get(&self, encoding_id: &str, version_id: &str) -> Result<MetricRecord>;
    async fn metric_put(&self, record: &MetricRecord) -> Result<()>;

    // Listing (for aggregation)
    async fn list_sources(&self) -> Result<Vec<String>>;
    async fn list_encodings(&self) -> Result<Vec<String>>;
    async fn list_metrics(&self, encoding_id: &str) -> Result<Vec<String>>;
}
```

### Codec
```rust
pub trait Codec: Send + Sync {
    // Identity
    fn name(&self) -> &str;
    fn crate_name(&self) -> &str;
    fn crate_version(&self) -> &str;
    fn commit(&self) -> Option<&str>;
    fn config(&self) -> serde_json::Value;

    // Operations
    fn extension(&self) -> &str;
    fn encode(&self, rgb: &[u8], w: u32, h: u32, quality: u8) -> Result<Vec<u8>>;
    fn decode(&self, data: &[u8]) -> Result<(Vec<u8>, u32, u32)>;
}
```

### MetricAlgorithm
```rust
pub trait MetricAlgorithm: Send + Sync {
    // Identity
    fn name(&self) -> &str;
    fn implementation(&self) -> &str;
    fn crate_name(&self) -> &str;
    fn crate_version(&self) -> &str;
    fn commit(&self) -> Option<&str>;
    fn config(&self) -> serde_json::Value;

    // Computation
    fn compute(
        &self,
        original: &[u8], orig_w: u32, orig_h: u32,
        encoded: &[u8], enc_w: u32, enc_h: u32,
    ) -> Result<f64>;
}
```

## Implementation Phases

### Phase 1: Core Models and Local Storage ✅ COMPLETE
- [x] Create crate structure
- [x] Implement model/ structs with serde (SourceRecord, EncodingRecord, MetricRecord, tasks)
- [x] Implement version/ matching logic (CodecVersionDef, MetricVersionDef, MatchRules)
- [x] Implement store/local.rs (filesystem with hash-distributed paths)
- [x] Implement store/memory.rs (testing)
- [x] Add comprehensive tests for version matching (70 tests passing)

### Phase 1.5: Safety and Auto-Versioning ✅ COMPLETE
- [x] Fingerprint-based version equivalence (CanarySet, ConfigFingerprint, EquivalenceRegistry)
- [x] Write-once semantics (SafeStore wrapper)
- [x] Integrity verification (Checksum, Envelope)
- [x] Soft deletes with tombstones (recoverable)
- [x] Audit logging for all mutations

### Phase 2: Codec and Metric Implementations
- [ ] Implement Codec trait with provenance capture
- [ ] Wrap mozjpeg-oxide
- [ ] Wrap jpegli-rs
- [ ] Wrap ravif (feature-gated)
- [ ] Wrap webp (feature-gated)
- [ ] Implement MetricAlgorithm trait
- [ ] Wrap fast-ssim2
- [ ] Wrap butteraugli
- [ ] Wrap dssim

### Phase 3: Task Planning and Execution
- [ ] Implement planner/manifest.rs
- [ ] Implement planner/remaining.rs (find missing work)
- [ ] Implement worker/encoding.rs
- [ ] Implement worker/metric.rs
- [ ] Create bin/coefficient.rs (local CLI)
- [ ] Add end-to-end test with small corpus

### Phase 4: Distributed Execution
- [ ] Implement store/gcs.rs (feature-gated)
- [ ] Implement worker/batch.rs (GCP Batch entry)
- [ ] Implement planner/distribution.rs
- [ ] Create bin/coefficient-worker.rs
- [ ] Create Dockerfile

### Phase 5: Analysis and Reporting
- [ ] Implement analysis/aggregator.rs
- [ ] Implement analysis/query.rs
- [ ] Implement analysis/export.rs (CSV, Parquet)
- [ ] Create bin/coefficient-aggregate.rs
- [ ] Add Pareto analysis

### Phase 6: Sync and Polish
- [ ] Implement store/sync.rs (local <-> GCS)
- [ ] Add progress reporting
- [ ] Add resume capability
- [ ] Documentation
- [ ] Examples

## Feature Flags

```toml
[features]
default = ["local"]
local = []                    # Filesystem storage
gcs = ["google-cloud-storage", "tokio"]  # GCS storage
gpu = ["ssimulacra2-cuda", "butteraugli-cuda", "dssim-cuda"]
avif = ["ravif", "image"]
webp = ["webp", "image"]
all-codecs = ["avif", "webp"]
```

## Dependencies

```toml
[dependencies]
# Core
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
sha2 = "0.10"
hex = "0.4"
chrono = { version = "0.4", features = ["serde"] }
thiserror = "1.0"
anyhow = "1.0"

# Async (optional, for GCS)
tokio = { version = "1", features = ["full"], optional = true }
async-trait = "0.1"

# Image handling
png = "0.17"
imgref = "1.10"
rgb = "0.8"

# Codecs (via path dependencies initially)
mozjpeg-oxide = { path = "../mozjpeg-rs", package = "mozjpeg-rs" }
jpegli = { path = "../jpegli-rs/jpegli-rs", package = "jpegli-rs" }
ravif = { version = "0.12", optional = true }
webp = { version = "0.3", optional = true }

# Metrics
fast-ssim2 = { version = "0.6", features = ["imgref"] }
butteraugli = { path = "../butteraugli/butteraugli" }
dssim = "3.2"

# Analysis
parquet = { version = "53", optional = true }
rusqlite = { version = "0.32", optional = true }

# GCS (optional)
google-cloud-storage = { version = "0.20", optional = true }
```

## CLI Interface

```bash
# Local benchmark
coefficient run \
  --corpus ~/work/codec-corpus/CID22/CID22-512/training \
  --corpus ~/work/codec-corpus/clic2025/validation \
  --output ./benchmark-results \
  --quality-range 1-100 \
  --codecs mozjpeg-420,jpegli-420,webp \
  --metrics ssim2-cpu,butteraugli-cpu

# Check what work remains
coefficient status ./benchmark-results

# Run only missing work
coefficient run --output ./benchmark-results --resume

# Aggregate results (build indices)
coefficient aggregate ./benchmark-results

# Query results
coefficient query ./benchmark-results \
  --codec-version "mozjpeg-420-v5" \
  --min-quality 70 \
  --export results.csv

# Sync to GCS
coefficient sync push ./benchmark-results gs://bucket/benchmarks/run1

# GCP Batch submission
coefficient batch submit \
  --project my-project \
  --bucket my-bucket \
  --corpus gs://bucket/sources/cid22 \
  --container gcr.io/my-project/coefficient:latest
```

## Migration from discover_heuristics

1. Create coefficient crate with core functionality
2. Add thin adapter to call coefficient from discover_heuristics
3. Gradually migrate discover_heuristics features to coefficient
4. Eventually deprecate discover_heuristics in favor of coefficient CLI

## Success Criteria

- [ ] 1000+ parallel workers without coordination bottleneck
- [ ] Results traceable to exact commit + config
- [ ] Version definitions updateable without re-running work
- [ ] GPU and CPU metric variants stored and comparable
- [ ] Works identically local and distributed
- [ ] Sync between local and GCS seamless
- [ ] Sub-second existence checks (no directory listings)
- [ ] Aggregation builds queryable index in reasonable time
