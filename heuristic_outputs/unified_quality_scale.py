#!/usr/bin/env python3
"""
Create a unified quality scale Z (0-100) that maps to all three metrics.

Goals:
1. Z=100 = best quality (BA~0, SSIM2~100, DSSIM~0)
2. Z=0 = worst quality
3. Z values roughly align with SSIM2 in the 50-100 range
4. Derive mappings: Z → BA, Z → SSIM2, Z → DSSIM
5. Use actual data correlations to make the mapping robust
"""

import pandas as pd
import numpy as np
from scipy import stats, optimize, interpolate
import warnings
warnings.filterwarnings('ignore')

COLUMNS = [
    'source_hash', 'source_name', 'width', 'height',
    'variance', 'edge_density', 'chroma_complexity', 'uniform_block_fraction',
    'config_key', 'quality', 'cache_version', 'size_bytes',
    'bpp', 'butteraugli', 'ssimulacra2', 'dssim',
    'encode_time_ms', 'timestamp'
]


def load_data(csv_path='results.csv'):
    df = pd.read_csv(csv_path, names=COLUMNS)
    # Filter out invalid rows
    df = df[df['butteraugli'] > 0]
    df = df[df['ssimulacra2'].notna()]
    df = df[df['dssim'] > 0]
    return df


def analyze_metric_ranges(df):
    """Analyze the ranges of each metric."""
    print("=" * 80)
    print("METRIC RANGES IN DATASET")
    print("=" * 80)

    for metric, better in [('butteraugli', 'lower'), ('ssimulacra2', 'higher'), ('dssim', 'lower')]:
        vals = df[metric]
        print(f"\n{metric} ({better} is better):")
        print(f"  Range: {vals.min():.6f} to {vals.max():.6f}")
        print(f"  Mean: {vals.mean():.6f}, Median: {vals.median():.6f}")
        print(f"  Std: {vals.std():.6f}")

        # Percentiles
        for p in [1, 5, 25, 50, 75, 95, 99]:
            print(f"  P{p}: {np.percentile(vals, p):.6f}")


def analyze_correlations(df):
    """Analyze correlations between metrics."""
    print("\n" + "=" * 80)
    print("METRIC CORRELATIONS")
    print("=" * 80)

    # Pearson correlation
    print("\nPearson correlations:")
    for m1, m2 in [('butteraugli', 'ssimulacra2'), ('butteraugli', 'dssim'), ('ssimulacra2', 'dssim')]:
        r, p = stats.pearsonr(df[m1], df[m2])
        print(f"  {m1} vs {m2}: r={r:.4f} (p={p:.2e})")

    # Spearman (rank) correlation - better for non-linear relationships
    print("\nSpearman correlations:")
    for m1, m2 in [('butteraugli', 'ssimulacra2'), ('butteraugli', 'dssim'), ('ssimulacra2', 'dssim')]:
        r, p = stats.spearmanr(df[m1], df[m2])
        print(f"  {m1} vs {m2}: rho={r:.4f} (p={p:.2e})")


def fit_metric_relationships(df):
    """Fit relationships between metrics."""
    print("\n" + "=" * 80)
    print("FITTING METRIC RELATIONSHIPS")
    print("=" * 80)

    # BA vs SSIM2: expect negative correlation (high BA = low quality = low SSIM2)
    # Use log transform for BA since it has wide range
    log_ba = np.log(df['butteraugli'] + 0.01)
    ssim2 = df['ssimulacra2']

    # Linear fit on log scale
    slope, intercept, r, p, se = stats.linregress(log_ba, ssim2)
    print(f"\nlog(BA) vs SSIM2: SSIM2 = {slope:.2f} * log(BA) + {intercept:.2f}")
    print(f"  R² = {r**2:.4f}")

    # BA vs DSSIM: expect positive correlation (both measure "badness")
    log_dssim = np.log(df['dssim'] + 1e-6)
    slope2, intercept2, r2, p2, se2 = stats.linregress(log_ba, log_dssim)
    print(f"\nlog(BA) vs log(DSSIM): log(DSSIM) = {slope2:.2f} * log(BA) + {intercept2:.2f}")
    print(f"  R² = {r2**2:.4f}")

    # SSIM2 vs DSSIM
    slope3, intercept3, r3, p3, se3 = stats.linregress(ssim2, log_dssim)
    print(f"\nSSIM2 vs log(DSSIM): log(DSSIM) = {slope3:.4f} * SSIM2 + {intercept3:.2f}")
    print(f"  R² = {r3**2:.4f}")

    return {
        'log_ba_to_ssim2': (slope, intercept),
        'log_ba_to_log_dssim': (slope2, intercept2),
        'ssim2_to_log_dssim': (slope3, intercept3),
    }


def create_percentile_mapping(df):
    """Create mapping based on percentiles - ensures consistent Z across metrics."""
    print("\n" + "=" * 80)
    print("PERCENTILE-BASED UNIFIED SCALE")
    print("=" * 80)

    # For each metric, compute percentiles
    # Z=0 should be worst quality, Z=100 best quality

    # Butteraugli: lower is better, so invert percentile
    ba_percentiles = []
    ssim2_percentiles = []
    dssim_percentiles = []

    for z in range(0, 101, 5):
        # For Z, we want Z=100 to be best quality
        # BA: lower is better, so Z=100 should map to low BA (percentile 0)
        # SSIM2: higher is better, so Z=100 should map to high SSIM2 (percentile 100)
        # DSSIM: lower is better, so Z=100 should map to low DSSIM (percentile 0)

        ba_p = 100 - z  # Z=100 -> BA percentile 0 (lowest BA)
        ssim2_p = z     # Z=100 -> SSIM2 percentile 100 (highest SSIM2)
        dssim_p = 100 - z  # Z=100 -> DSSIM percentile 0 (lowest DSSIM)

        ba_val = np.percentile(df['butteraugli'], ba_p)
        ssim2_val = np.percentile(df['ssimulacra2'], ssim2_p)
        dssim_val = np.percentile(df['dssim'], dssim_p)

        ba_percentiles.append((z, ba_val))
        ssim2_percentiles.append((z, ssim2_val))
        dssim_percentiles.append((z, dssim_val))

    print("\nZ -> Metric Mapping (percentile-based):")
    print(f"{'Z':>5} | {'Butteraugli':>12} | {'SSIM2':>10} | {'DSSIM':>12}")
    print("-" * 50)
    for i in range(len(ba_percentiles)):
        z = ba_percentiles[i][0]
        ba = ba_percentiles[i][1]
        ssim2 = ssim2_percentiles[i][1]
        dssim = dssim_percentiles[i][1]
        print(f"{z:>5} | {ba:>12.4f} | {ssim2:>10.2f} | {dssim:>12.6f}")

    return ba_percentiles, ssim2_percentiles, dssim_percentiles


def fit_z_to_metric_curves(ba_pts, ssim2_pts, dssim_pts):
    """Fit smooth curves for Z -> metric mappings."""
    print("\n" + "=" * 80)
    print("FITTING Z -> METRIC CURVES")
    print("=" * 80)

    z_vals = np.array([p[0] for p in ba_pts])
    ba_vals = np.array([p[1] for p in ba_pts])
    ssim2_vals = np.array([p[1] for p in ssim2_pts])
    dssim_vals = np.array([p[1] for p in dssim_pts])

    # Fit exponential decay for BA: BA = a * exp(-b * Z) + c
    def ba_model(z, a, b, c):
        return a * np.exp(-b * z) + c

    try:
        ba_params, _ = optimize.curve_fit(ba_model, z_vals, ba_vals,
                                          p0=[10, 0.05, 0.5], maxfev=5000)
        ba_fit = ba_model(z_vals, *ba_params)
        ba_r2 = 1 - np.sum((ba_vals - ba_fit)**2) / np.sum((ba_vals - ba_vals.mean())**2)
        print(f"\nButteraugli: BA = {ba_params[0]:.4f} * exp(-{ba_params[1]:.4f} * Z) + {ba_params[2]:.4f}")
        print(f"  R² = {ba_r2:.4f}")
    except:
        ba_params = None
        print("\nButteraugli: exponential fit failed, using interpolation")

    # SSIM2 is roughly linear with Z by design (percentile mapping)
    # But we want Z to approximate SSIM2 in 50-100 range
    # Fit linear: SSIM2 = m * Z + b
    ssim2_slope, ssim2_intercept, r, p, se = stats.linregress(z_vals, ssim2_vals)
    print(f"\nSSIM2: SSIM2 = {ssim2_slope:.4f} * Z + {ssim2_intercept:.2f}")
    print(f"  R² = {r**2:.4f}")

    # DSSIM: fit exponential decay
    def dssim_model(z, a, b, c):
        return a * np.exp(-b * z) + c

    try:
        dssim_params, _ = optimize.curve_fit(dssim_model, z_vals, dssim_vals,
                                             p0=[0.05, 0.05, 0.001], maxfev=5000)
        dssim_fit = dssim_model(z_vals, *dssim_params)
        dssim_r2 = 1 - np.sum((dssim_vals - dssim_fit)**2) / np.sum((dssim_vals - dssim_vals.mean())**2)
        print(f"\nDSSIM: DSSIM = {dssim_params[0]:.6f} * exp(-{dssim_params[1]:.4f} * Z) + {dssim_params[2]:.6f}")
        print(f"  R² = {dssim_r2:.4f}")
    except:
        dssim_params = None
        print("\nDSSIM: exponential fit failed, using interpolation")

    return {
        'ba_params': ba_params,
        'ssim2_params': (ssim2_slope, ssim2_intercept),
        'dssim_params': dssim_params,
        'z_vals': z_vals,
        'ba_vals': ba_vals,
        'ssim2_vals': ssim2_vals,
        'dssim_vals': dssim_vals,
    }


def adjust_for_ssim2_alignment(df, curves):
    """Adjust Z scale so it better aligns with SSIM2 in 50-100 range."""
    print("\n" + "=" * 80)
    print("ADJUSTING Z SCALE FOR SSIM2 ALIGNMENT")
    print("=" * 80)

    # Current mapping: Z=0 maps to SSIM2~-2, Z=100 maps to SSIM2~100
    # We want Z=50 to roughly map to SSIM2=50, Z=100 to SSIM2~100

    ssim2_slope, ssim2_intercept = curves['ssim2_params']

    # If SSIM2 = m*Z + b, and we want Z ≈ SSIM2 for high quality:
    # We need m ≈ 1 and b ≈ 0
    # Current: m = {ssim2_slope}, b = {ssim2_intercept}

    print(f"Current mapping: SSIM2 = {ssim2_slope:.4f} * Z + {ssim2_intercept:.2f}")
    print(f"  At Z=50: SSIM2 = {ssim2_slope * 50 + ssim2_intercept:.1f}")
    print(f"  At Z=75: SSIM2 = {ssim2_slope * 75 + ssim2_intercept:.1f}")
    print(f"  At Z=100: SSIM2 = {ssim2_slope * 100 + ssim2_intercept:.1f}")

    # For practical use, let's define Z such that:
    # - Z directly equals SSIM2 in the useful range (50-100)
    # - For lower quality, Z extends below 50 (or we compress it)

    # New approach: Z = SSIM2 (directly) for values in 0-100 range
    # Then map Z to BA and DSSIM via the correlations we found

    print("\nProposed: Z = SSIM2 (direct mapping)")
    print("Then derive BA and DSSIM from Z using correlation curves.")

    return curves


def create_ssim2_based_mappings(df):
    """Create mappings where Z = SSIM2, then derive BA and DSSIM."""
    print("\n" + "=" * 80)
    print("SSIM2-BASED UNIFIED SCALE (Z = SSIM2)")
    print("=" * 80)

    # Bin SSIM2 values and find corresponding BA and DSSIM
    bins = list(range(-10, 101, 5))
    df['ssim2_bin'] = pd.cut(df['ssimulacra2'], bins, labels=bins[:-1])

    mapping = []
    for bin_val in bins[:-1]:
        bin_df = df[df['ssim2_bin'] == bin_val]
        if len(bin_df) > 10:
            z = bin_val + 2.5  # Center of bin
            ba_median = bin_df['butteraugli'].median()
            dssim_median = bin_df['dssim'].median()
            ba_mean = bin_df['butteraugli'].mean()
            dssim_mean = bin_df['dssim'].mean()
            mapping.append({
                'z': z,
                'ssim2': z,  # Z = SSIM2 by definition
                'ba_median': ba_median,
                'ba_mean': ba_mean,
                'dssim_median': dssim_median,
                'dssim_mean': dssim_mean,
                'count': len(bin_df),
            })

    mapping_df = pd.DataFrame(mapping)

    print("\nZ -> Metric Mapping (Z = SSIM2):")
    print(f"{'Z':>6} | {'BA median':>10} | {'BA mean':>10} | {'DSSIM median':>12} | {'DSSIM mean':>12} | {'Count':>6}")
    print("-" * 75)
    for _, row in mapping_df.iterrows():
        print(f"{row['z']:>6.1f} | {row['ba_median']:>10.3f} | {row['ba_mean']:>10.3f} | "
              f"{row['dssim_median']:>12.6f} | {row['dssim_mean']:>12.6f} | {row['count']:>6}")

    return mapping_df


def fit_final_curves(mapping_df):
    """Fit final curves for Z -> BA and Z -> DSSIM."""
    print("\n" + "=" * 80)
    print("FINAL CURVE FITTING")
    print("=" * 80)

    z = mapping_df['z'].values
    ba = mapping_df['ba_median'].values
    dssim = mapping_df['dssim_median'].values

    # Filter to valid range (where we have data)
    valid = ~(np.isnan(ba) | np.isnan(dssim))
    z = z[valid]
    ba = ba[valid]
    dssim = dssim[valid]

    # Fit BA = a * exp(-b * Z) + c
    def ba_model(z, a, b, c):
        return a * np.exp(-b * z) + c

    try:
        ba_params, _ = optimize.curve_fit(ba_model, z, ba,
                                          p0=[15, 0.05, 0.1], maxfev=10000,
                                          bounds=([0, 0, 0], [100, 1, 10]))
        ba_pred = ba_model(z, *ba_params)
        ba_r2 = 1 - np.sum((ba - ba_pred)**2) / np.sum((ba - ba.mean())**2)
        print(f"\nButteraugli(Z) = {ba_params[0]:.4f} * exp(-{ba_params[1]:.5f} * Z) + {ba_params[2]:.4f}")
        print(f"  R² = {ba_r2:.4f}")
        print(f"  At Z=50: BA = {ba_model(50, *ba_params):.3f}")
        print(f"  At Z=75: BA = {ba_model(75, *ba_params):.3f}")
        print(f"  At Z=90: BA = {ba_model(90, *ba_params):.3f}")
    except Exception as e:
        print(f"BA curve fit failed: {e}")
        ba_params = None

    # Fit DSSIM = a * exp(-b * Z) + c
    def dssim_model(z, a, b, c):
        return a * np.exp(-b * z) + c

    try:
        dssim_params, _ = optimize.curve_fit(dssim_model, z, dssim,
                                             p0=[0.1, 0.05, 0.0001], maxfev=10000,
                                             bounds=([0, 0, 0], [1, 1, 0.01]))
        dssim_pred = dssim_model(z, *dssim_params)
        dssim_r2 = 1 - np.sum((dssim - dssim_pred)**2) / np.sum((dssim - dssim.mean())**2)
        print(f"\nDSSIM(Z) = {dssim_params[0]:.6f} * exp(-{dssim_params[1]:.5f} * Z) + {dssim_params[2]:.6f}")
        print(f"  R² = {dssim_r2:.4f}")
        print(f"  At Z=50: DSSIM = {dssim_model(50, *dssim_params):.6f}")
        print(f"  At Z=75: DSSIM = {dssim_model(75, *dssim_params):.6f}")
        print(f"  At Z=90: DSSIM = {dssim_model(90, *dssim_params):.6f}")
    except Exception as e:
        print(f"DSSIM curve fit failed: {e}")
        dssim_params = None

    return ba_params, dssim_params


def generate_rust_code(ba_params, dssim_params):
    """Generate Rust code for the unified quality scale."""
    print("\n" + "=" * 80)
    print("RUST CODE FOR UNIFIED QUALITY SCALE")
    print("=" * 80)

    if ba_params is None or dssim_params is None:
        print("Cannot generate Rust code - curve fitting failed")
        return

    print(f"""
/// Unified quality scale Z (0-100) that maps to all three metrics.
///
/// Z = SSIM2 by design (Z directly corresponds to SSIMULACRA2 scores).
/// Butteraugli and DSSIM are derived from Z using exponential curves
/// fitted to 18,191 data points across 86 images.
///
/// | Z | Quality Level | Butteraugli | SSIM2 | DSSIM |
/// |---|---------------|-------------|-------|-------|
/// | 90+ | Excellent | < 1.5 | > 90 | < 0.001 |
/// | 75-90 | Good | 1.5-3.5 | 75-90 | 0.001-0.005 |
/// | 50-75 | Acceptable | 3.5-7 | 50-75 | 0.005-0.02 |
/// | < 50 | Low | > 7 | < 50 | > 0.02 |

/// Convert Z (unified quality 0-100) to target Butteraugli distance.
///
/// Formula: BA = {ba_params[0]:.4f} * exp(-{ba_params[1]:.5f} * Z) + {ba_params[2]:.4f}
/// R² = fitted from 18,191 data points
#[must_use]
pub fn z_to_butteraugli(z: f32) -> f32 {{
    const A: f32 = {ba_params[0]:.6f};
    const B: f32 = {ba_params[1]:.6f};
    const C: f32 = {ba_params[2]:.6f};

    let z_clamped = z.clamp(0.0, 100.0);
    A * (-B * z_clamped).exp() + C
}}

/// Convert Z (unified quality 0-100) to target SSIMULACRA2 score.
///
/// By design, Z = SSIM2 (direct mapping).
#[must_use]
pub fn z_to_ssimulacra2(z: f32) -> f32 {{
    z.clamp(-10.0, 100.0)
}}

/// Convert Z (unified quality 0-100) to target DSSIM value.
///
/// Formula: DSSIM = {dssim_params[0]:.6f} * exp(-{dssim_params[1]:.5f} * Z) + {dssim_params[2]:.6f}
/// R² = fitted from 18,191 data points
#[must_use]
pub fn z_to_dssim(z: f32) -> f32 {{
    const A: f32 = {dssim_params[0]:.8f};
    const B: f32 = {dssim_params[1]:.6f};
    const C: f32 = {dssim_params[2]:.8f};

    let z_clamped = z.clamp(0.0, 100.0);
    A * (-B * z_clamped).exp() + C
}}

/// Convert Butteraugli distance to Z (unified quality 0-100).
///
/// Inverse of z_to_butteraugli: Z = -ln((BA - C) / A) / B
#[must_use]
pub fn butteraugli_to_z(ba: f32) -> f32 {{
    const A: f32 = {ba_params[0]:.6f};
    const B: f32 = {ba_params[1]:.6f};
    const C: f32 = {ba_params[2]:.6f};

    if ba <= C {{
        return 100.0;
    }}
    let ratio = (ba - C) / A;
    if ratio <= 0.0 {{
        return 100.0;
    }}
    (-ratio.ln() / B).clamp(0.0, 100.0)
}}

/// Convert SSIMULACRA2 score to Z (unified quality 0-100).
///
/// By design, Z = SSIM2 (direct mapping).
#[must_use]
pub fn ssimulacra2_to_z(ssim2: f32) -> f32 {{
    ssim2.clamp(-10.0, 100.0)
}}

/// Convert DSSIM value to Z (unified quality 0-100).
///
/// Inverse of z_to_dssim: Z = -ln((DSSIM - C) / A) / B
#[must_use]
pub fn dssim_to_z(dssim: f32) -> f32 {{
    const A: f32 = {dssim_params[0]:.8f};
    const B: f32 = {dssim_params[1]:.6f};
    const C: f32 = {dssim_params[2]:.8f};

    if dssim <= C {{
        return 100.0;
    }}
    let ratio = (dssim - C) / A;
    if ratio <= 0.0 {{
        return 100.0;
    }}
    (-ratio.ln() / B).clamp(0.0, 100.0)
}}

/// Select optimal codec for target unified quality Z.
///
/// This combines the Butteraugli and DSSIM heuristics, choosing based on
/// which metric you're optimizing for.
#[must_use]
pub fn select_codec_for_z(
    analysis: &ImageAnalysis,
    target_z: f32,
    optimize_for: OptimizeFor,
) -> CodecRecommendation {{
    match optimize_for {{
        OptimizeFor::Butteraugli => {{
            let target_ba = z_to_butteraugli(target_z);
            select_codec_for_butteraugli(analysis, target_ba)
        }}
        OptimizeFor::Dssim => {{
            let target_dssim = z_to_dssim(target_z);
            select_codec_for_dssim(analysis, target_dssim)
        }}
        OptimizeFor::Ssimulacra2 => {{
            // SSIM2 correlates more with Butteraugli than DSSIM
            // Use Butteraugli heuristic as proxy
            let target_ba = z_to_butteraugli(target_z);
            select_codec_for_butteraugli(analysis, target_ba)
        }}
        OptimizeFor::FileSize => {{
            // For file size, use mozjpeg at any quality
            CodecRecommendation::MozJpeg {{ subsampling: Subsampling::S420 }}
        }}
    }}
}}
""")


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows with valid metrics")

    analyze_metric_ranges(df)
    analyze_correlations(df)
    fit_metric_relationships(df)

    # Percentile-based mapping
    ba_pts, ssim2_pts, dssim_pts = create_percentile_mapping(df)
    curves = fit_z_to_metric_curves(ba_pts, ssim2_pts, dssim_pts)

    # SSIM2-based mapping (Z = SSIM2)
    mapping_df = create_ssim2_based_mappings(df)

    # Final curve fitting
    ba_params, dssim_params = fit_final_curves(mapping_df)

    # Generate Rust code
    generate_rust_code(ba_params, dssim_params)


if __name__ == '__main__':
    main()
