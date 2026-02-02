# Simulated Annealing for JPEG Quantization Tables

## Source
- **URL**: https://www.eecs.harvard.edu/~michaelm/SimAnneal/simulated_annealing_for_JPEG_quantization.html
- **Authors**: Max Hopkins, Michael Mitzenmacher, Sebastian Wagner-Carena (Harvard)

## Summary

The research uses simulated annealing to optimize JPEG quantization tables, achieving:
- **37-52% compression gains** over standard JPEG tables
- **Maintained quality** (error ratios typically > 0.85)
- Tables trained at specific quality levels (95, 75, 50, 35) but usable across ranges

### Key Results

| Table Type | Quality | Compression Ratio | Error Ratio | Notes |
|------------|---------|-------------------|-------------|-------|
| Overall Best | Q75 | 58.9% of standard | 0.862 | Best balance |
| Best Compression | Q50 | 47.6% of standard | 0.990 | Minimal quality loss |

## Relevance to zenjpeg

### Current State
zenjpeg uses:
1. Standard JPEG quantization tables
2. mozjpeg's modified tables (slightly optimized)
3. Quality scaling via standard formula

### Potential Integration

#### Option 1: Pre-computed Optimized Tables
Replace standard tables with Harvard's optimized tables at specific quality levels.

**Pros:**
- Zero runtime cost
- Drop-in replacement
- Proven 40%+ compression gains

**Cons:**
- Tables optimized for specific metric (FSIM)
- May not align with SSIMULACRA2/Butteraugli targets
- Limited to discrete quality levels (35, 50, 75, 95)

#### Option 2: Simulated Annealing at Encode Time
Run SA during encoding to find optimal table for each image.

**Pros:**
- Per-image optimization
- Can target any quality metric (SSIM2, Butteraugli)
- Potentially better results than pre-computed

**Cons:**
- Massive performance hit (many encode iterations)
- Unclear convergence for real-time use
- Only practical for batch/offline encoding

#### Option 3: Metric-Specific Pre-computed Tables
Train our own tables using SA, but optimized for:
- SSIMULACRA2 (our primary metric)
- Butteraugli (perceptual)
- DSSIM (structural)

**Pros:**
- Tables aligned with our target metrics
- One-time training cost
- Zero runtime cost after training

**Cons:**
- Requires significant upfront research
- Need large training corpus
- Tables may overfit to training images

## Implementation Plan

### Phase 1: Evaluation (1-2 days)
- [ ] Download/extract Harvard's optimized tables
- [ ] Add tables to `src/quant.rs` as alternative options
- [ ] Benchmark against standard tables on CID22 corpus
- [ ] Measure: file size, SSIMULACRA2, Butteraugli, DSSIM
- [ ] Compare to mozjpeg and jpegli baselines

### Phase 2: Integration (if Phase 1 shows wins)
- [ ] Add `QuantizationTable` enum: `Standard`, `Mozjpeg`, `HarvardOptimized(Quality)`
- [ ] Expose in `EncoderConfig`
- [ ] Update strategy selection to consider optimized tables
- [ ] Document when to use which table set

### Phase 3: Custom Training (stretch goal)
- [ ] Implement SA algorithm for table optimization
- [ ] Train tables optimized for SSIMULACRA2
- [ ] Train tables optimized for Butteraugli
- [ ] Cross-validate on held-out images
- [ ] Benchmark final tables

## Simulated Annealing Algorithm

### Core Concept
```
1. Start with standard quantization table Q
2. For temperature T from hot to cold:
   a. Generate neighbor Q' by perturbing Q
   b. Compute cost(Q') = encode_size(Q') + λ * quality_error(Q')
   c. If cost(Q') < cost(Q):
      Accept Q' unconditionally
   d. Else:
      Accept Q' with probability exp(-(cost(Q') - cost(Q)) / T)
3. Return best Q found
```

### Neighbor Generation
- Increment/decrement single coefficient
- Swap two coefficients
- Scale coefficient by small factor

### Cost Function
```
cost(Q) = α * file_size(Q) + β * (1 - quality_metric(Q))
```
Where quality_metric could be FSIM, SSIMULACRA2, etc.

### Temperature Schedule
- Start high (accept most moves)
- Cool slowly (geometric: T = T * 0.95)
- Stop when T is very low or no improvement

## Licensing

### Harvard SA Tables
- **Public Domain** - free to use
- 28 tables available for download in text format
- Authors: Max Hopkins, Michael Mitzenmacher, Sebastian Wagner-Carena

### jpegli Base Quantization Tables
jpegli uses **three different base quantization matrices** (from `jpegli-rs/src/consts.rs`):

| Matrix | Size | License | Source |
|--------|------|---------|--------|
| `BASE_QUANT_MATRIX_STD` | 128 values (2×64) | **Public Domain** | ITU-T T.81 Annex K (standard JPEG spec) |
| `BASE_QUANT_MATRIX_YCBCR` | 192 values (3×64) | **BSD-3-Clause** | libjxl/jpegli |
| `BASE_QUANT_MATRIX_XYB` | 192 values (3×64) | **BSD-3-Clause** | libjxl/jpegli |

**Key differences from standard JPEG:**
- jpegli's YCbCr matrices are custom-tuned for perceptual quality
- Per-frequency non-linear scaling via `distance_to_scale()`
- Global scale factors: `GLOBAL_SCALE_XYB = 1.439` and `GLOBAL_SCALE_YCBCR = 1.739`
- Additional 4:2:0 rescale factors (`K420_RESCALE`) for chroma preservation

**Example values (Y channel, first 8 coefficients):**
```
Standard JPEG:  16, 11, 10, 16, 24, 40, 51, 61
jpegli YCbCr:   1.24, 1.72, 2.92, 2.81, 3.34, 3.46, 3.84, 3.87
```
jpegli uses much finer granularity (floats scaled by distance), then rounds to integer.

## Questions to Answer

1. **Are Harvard tables available for download?**
   - Yes, 28 tables in text format are linked from the webpage
   - Need to verify format and extract coefficients

2. ~~**What license applies to Harvard tables?**~~ **Public Domain** ✓

3. **How do tables generalize across image types?**
   - Trained on specific corpus
   - May not be optimal for all content types

4. **Can we combine with trellis quantization?**
   - Optimized tables + trellis could compound gains
   - Or could interfere (trellis expects standard tables?)

5. **What about chrominance tables?**
   - Harvard research may focus on luminance only
   - Chrominance tables matter for 4:2:0 subsampling

## Related Work

- **mozjpeg**: Uses modified tables with slightly better compression
- **jpegli**: Uses adaptive quantization (per-block adjustment)
- **JPEG XL**: Uses entirely different approach (no fixed tables)
- **Neural compression**: Learning-based, but compute-heavy

## Next Steps

1. Fetch full paper (if available) for detailed algorithm
2. Look for published table coefficients
3. Set up comparison benchmark
4. Decide on integration approach based on results

## Notes

- FSIM (Feature SIMilarity) may correlate differently than SSIMULACRA2
- Tables trained at Q75 showed best overall results - matches our crossover point
- This approach is orthogonal to trellis/AQ - could combine all three
