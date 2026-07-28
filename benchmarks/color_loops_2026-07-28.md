# Full-image colour loops: `Vec::push` → slice writes — 2026-07-28

Platform: Apple Silicon (aarch64, NEON), darwin 25.5.0
Bench: `benches/color.rs` (zenbench, interleaved arms)

`convert_rgb_to_ycbcr` and `deinterleave_ycbcr` (`src/color.rs`) run over every pixel on the
encode path (`src/encode.rs` calls both) and were written as per-pixel `Vec::push` loops.
`push` pays a capacity check per element and blocks vectorization.

| case | `Vec::push` (was) | slice writes (now) | speedup |
|---|---|---|---|
| rgb→ycbcr 1024² | 4.3 ms | 1.4 ms | **3.0×** |
| rgb→ycbcr 4096² | 39.5 ms | 13.1 ms | **3.0×** |
| deinterleave 1024² | 3.0 ms | 0.2 ms | **15×** |
| deinterleave 4096² | 26.4 ms | 1.4 ms | **19×** |

`deinterleave_ycbcr` gains far more because it is a pure data movement — three planes out of
one interleaved buffer — so once the stores are into fixed-size slices LLVM widens them. The
colour conversion is arithmetic-bound (three f32 dot products, rounding and clamping per
pixel), so removing `push` uncovers the real work rather than eliminating it; 3× is the loop
overhead that was hiding it.

Not a SIMD change. Consistent with the rest of this aarch64 sweep: NEON is baseline, so LLVM
vectorizes ordinary slice loops and an explicit kernel only pays for permutes, table lookups
or horizontal reductions.

## Build blocker (pre-existing, NOT introduced here)

**This crate does not build in a clean checkout.** `Cargo.toml` declares five optional path
dependencies under `../turbo-metrics/` (`ssimulacra2-cuda`, `dssim-cuda`, `butteraugli-cuda`,
`cudarse-driver`, `cudarse-npp`) and that directory is absent. Cargo reads path dependencies'
manifests whether or not the feature that gates them is enabled, so *every* cargo command
fails with "failed to read .../butteraugli-cuda/Cargo.toml", not just `--features gpu`.

The numbers above were obtained by temporarily commenting those five deps out and emptying the
`gpu` feature, running the bench and `cargo test --lib` (108 pass), then restoring `Cargo.toml`
**byte-for-byte** (verified: `diff` = 0 lines). No workaround is committed.

To make this crate usable again, those path deps need to become optional git/registry deps, be
removed, or the sibling checkout restored. Until then `benches/color.rs` cannot run from a
clean tree either — same blocker, not a separate one.

The crate also has no in-workspace consumers, so nothing is currently affected by either the
blocker or the speedup.
