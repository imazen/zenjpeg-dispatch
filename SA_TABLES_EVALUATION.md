# Simulated Annealing Quantization Tables Evaluation

## Harvard SA Tables (2019)

**Source**: https://www.eecs.harvard.edu/~michaelm/SimAnneal/
**Optimization metric**: FSIM (Feature Similarity Index)
**Tables tested**: Q35, Q50, Q75, Q95

### Benchmark Results (Jan 2026)

Tested on 8 images from codec corpus at matched BPP using SSIMULACRA2, Butteraugli, and DSSIM metrics.

#### Performance Summary

| SA Table | BPP | Equivalent Std Q | SSIM2 Δ | Butteraugli Δ | DSSIM Δ | Verdict |
|----------|-----|------------------|---------|---------------|---------|---------|
| Q50 | 0.345 | Q35 | -11.27 | -2.27 (worse) | -0.0049 (worse) | ✗ Much worse |
| Q35 | 0.424 | Q35 | -5.12 | -0.75 (worse) | -0.0019 (worse) | ✗ Worse |
| Q75 | 0.432 | Q35 | -3.85 | -0.52 (worse) | -0.0017 (worse) | ✗ Worse |
| Q95 | 0.645 | Q47-48 | **+1.67** | -0.54 (worse) | -0.00004 (worse) | ✗ Mixed |

**Key findings**:
- Only SA Q95 shows improvement, and only on SSIMULACRA2 (+1.67)
- All tables perform worse on Butteraugli and DSSIM
- Harvard's claimed "37-52% compression gains" do not materialize with modern metrics
- Tables optimized for FSIM don't transfer well to SSIMULACRA2/Butteraugli/DSSIM

### Why Harvard SA Tables Failed

1. **Metric mismatch**: Optimized for FSIM, not SSIMULACRA2/Butteraugli/DSSIM
2. **Low quality tables too aggressive**: Q35/Q50/Q75 sacrifice too much quality for compression
3. **Single-metric optimization**: Modern encoders balance multiple perceptual metrics

### Conclusion

**Do not integrate Harvard SA tables into zenjpeg.** The minimal gain (+1.67 SSIM2 in narrow range) doesn't justify the added complexity, especially given worse performance on other metrics.

---

## Future: Custom SA Tables for zenjpeg

### Motivation

Harvard's approach (simulated annealing for quantization table optimization) is sound, but their tables were:
- Optimized for the wrong metric (FSIM instead of SSIMULACRA2/Butteraugli)
- Trained on different image corpus
- Designed for baseline JPEG, not hybrid mozjpeg/jpegli strategies

**Custom SA tables optimized for modern metrics could close zenjpeg's remaining gap vs jpegli.**

### Design Approach

#### 1. Target Metrics
Multi-objective optimization:
- **Primary**: SSIMULACRA2 (what jpegli excels at)
- **Secondary**: Butteraugli, DSSIM
- **Constraint**: Must not degrade any metric by >5% vs best single-metric table

#### 2. Quality Ranges
Focus on ranges where zenjpeg currently lags:
- **Q70-Q85**: Where jpegli's adaptive quantization dominates
- **Q50-Q70**: Transition zone between mozjpeg trellis and jpegli AQ

#### 3. Table Variants
Generate multiple table sets:
- `SA_SSIM2_QXX`: Optimized purely for SSIMULACRA2
- `SA_BALANCED_QXX`: Multi-metric optimization (SSIM2 + BA + DSSIM)
- `SA_HYBRID_QXX`: Tables designed for use WITH trellis quantization

#### 4. Training Corpus
Use same corpus as benchmark:
- codec-corpus images (diverse content)
- CID22 (standard reference)
- Consider adding specific challenging cases (textures, gradients, etc.)

#### 5. Optimization Framework

**Required components**:
1. **Simulated annealing implementation**:
   - Rust crate: `rand`, `rayon` for parallel evaluation
   - Temperature schedule: exponential decay
   - Perturbation strategy: random ±1-5 on random table entries

2. **Metric evaluation**:
   - SSIMULACRA2: `ssimulacra2` crate (already have)
   - Butteraugli: `butteraugli` crate (already have)
   - DSSIM: `dssim` crate (already have)

3. **JPEG encoding pipeline**:
   - Use `mozjpeg-oxide` with custom quantization tables
   - Need efficient batch encoding (encode same image with many table variations)

4. **Objective function**:
   ```
   score = w1 * ssim2_gain + w2 * butteraugli_gain + w3 * dssim_gain
   where gain = (baseline_metric - new_metric) normalized by baseline
   ```

**Estimated effort**: 2-3 weeks for framework + 1 week of compute time for optimization

### Success Criteria

Custom SA tables would be considered successful if:
1. **At Q75**: Close >50% of gap between current zenjpeg and jpegli on SSIM2
2. **At matched BPP**: Achieve >3 SSIM2 improvement over standard tables
3. **Multi-metric**: Not degrade Butteraugli or DSSIM by >5%

### Implementation Plan (Future)

**Phase 1: Evaluation framework** (3 days)
- [ ] Script to batch-encode with custom quantization tables
- [ ] Multi-metric evaluation pipeline
- [ ] Baseline measurements for standard/mozjpeg/jpegli tables

**Phase 2: SA optimizer** (1 week)
- [ ] Simulated annealing implementation
- [ ] Perturbation strategies (random, gradient-guided, frequency-based)
- [ ] Parallel evaluation of candidate tables
- [ ] Convergence criteria and checkpointing

**Phase 3: Table generation** (1 week compute)
- [ ] Generate tables for Q50, Q60, Q70, Q75, Q80, Q85
- [ ] Test both luma and chroma table optimization
- [ ] Validate on held-out test images

**Phase 4: Integration** (3 days)
- [ ] Add custom tables to `src/sa_tables.rs` or `src/custom_quant.rs`
- [ ] Update quality selection logic
- [ ] Benchmark vs jpegli/mozjpeg

### Open Questions

1. **Should we optimize luma and chroma together or separately?**
   - Harvard only optimized luma
   - jpegli uses different chroma strategies

2. **Should tables be designed to work WITH or WITHOUT trellis?**
   - Could optimize tables assuming trellis will further refine
   - Or optimize as final quantization

3. **How to handle subsampling?**
   - Separate tables for 4:4:4, 4:2:0?
   - Or universal tables?

4. **Training time vs quality tradeoff?**
   - Harvard paper doesn't specify how long optimization took
   - May need GPU acceleration for faster iteration

---

## References

- Harvard SA tables: https://www.eecs.harvard.edu/~michaelm/SimAnneal/
- Benchmark data: `/home/lilith/work/zenjpeg/comparison_outputs/sa_tables/`
- Analysis scripts: `/home/lilith/work/zenjpeg/examples/analyze_sa_*.py`
