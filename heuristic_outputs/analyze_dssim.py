#!/usr/bin/env python3
"""
Analyze codec selection heuristics targeting DSSIM.

DSSIM: 0 = identical, higher = worse quality.
Typical ranges:
  - 0.001 = very high quality (visually lossless)
  - 0.005 = high quality
  - 0.01  = medium-high quality
  - 0.02  = medium quality
  - 0.03+ = low quality
"""

import pandas as pd
import numpy as np
from scipy import interpolate
import warnings
warnings.filterwarnings('ignore')

COLUMNS = [
    'source_hash', 'source_name', 'width', 'height',
    'variance', 'edge_density', 'chroma_complexity', 'uniform_block_fraction',
    'config_key', 'quality', 'cache_version', 'size_bytes',
    'bpp', 'butteraugli', 'ssimulacra2', 'dssim',
    'encode_time_ms', 'timestamp'
]

FEATURES = ['variance', 'edge_density', 'chroma_complexity', 'uniform_block_fraction']
CONFIGS = ['jpegli-420', 'jpegli-444', 'mozjpeg-420', 'mozjpeg-444']

# DSSIM targets (higher = worse quality)
DSSIM_TARGETS = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]


def load_data(csv_path='results.csv'):
    return pd.read_csv(csv_path, names=COLUMNS)


def interpolate_bpp_at_dssim(group, target_dssim):
    """Interpolate BPP needed to achieve target DSSIM."""
    sorted_group = group.sort_values('dssim')
    dssim_values = sorted_group['dssim'].values
    bpp_values = sorted_group['bpp'].values

    # Need DSSIM <= target (better or equal quality)
    if target_dssim < dssim_values.min():
        return None  # Can't achieve this quality
    if target_dssim > dssim_values.max():
        # Can easily achieve, use lowest BPP that's below target
        valid = sorted_group[sorted_group['dssim'] <= target_dssim]
        if len(valid) > 0:
            return valid['bpp'].min()
        return None

    try:
        f = interpolate.interp1d(dssim_values, bpp_values, kind='linear')
        return float(f(target_dssim))
    except:
        return None


def get_config_bpp(img_df, target_dssim, configs=CONFIGS):
    """Get BPP for each config to achieve target DSSIM."""
    config_bpp = {}
    for config_key, config_group in img_df.groupby('config_key'):
        if config_key not in configs:
            continue
        # Check if config can achieve this DSSIM (has samples at or below target)
        min_dssim = config_group['dssim'].min()
        if min_dssim <= target_dssim:
            bpp = interpolate_bpp_at_dssim(config_group, target_dssim)
            if bpp is not None and bpp > 0:
                config_bpp[config_key] = bpp
    return config_bpp


def build_dataset(df, target_values):
    """Build dataset with optimal config per image/target."""
    data = []
    for source_hash, img_group in df.groupby('source_hash'):
        chars = img_group[FEATURES].iloc[0].to_dict()
        chars['source_hash'] = source_hash

        for target in target_values:
            config_bpp = get_config_bpp(img_group, target, CONFIGS)

            if len(config_bpp) < 1:
                continue

            optimal_config = min(config_bpp, key=config_bpp.get)
            optimal_bpp = config_bpp[optimal_config]

            sample = chars.copy()
            sample['target_dssim'] = target
            sample['optimal_config'] = optimal_config
            sample['optimal_bpp'] = optimal_bpp

            for config in CONFIGS:
                if config in config_bpp:
                    sample[f'bpp_{config}'] = config_bpp[config]
                    sample[f'regret_{config}'] = (config_bpp[config] - optimal_bpp) / optimal_bpp * 100
                else:
                    sample[f'bpp_{config}'] = np.nan
                    sample[f'regret_{config}'] = np.nan

            data.append(sample)

    return pd.DataFrame(data)


def analyze_winners_by_target(df):
    """Analyze which codec wins at each DSSIM target."""
    print("=" * 80)
    print("OPTIMAL CODEC BY DSSIM TARGET")
    print("=" * 80)

    for target in DSSIM_TARGETS:
        target_df = df[df['target_dssim'] == target]
        if len(target_df) == 0:
            continue

        winners = target_df['optimal_config'].value_counts()
        total = len(target_df)

        print(f"\nDSSIM <= {target}:")
        for config, count in winners.items():
            print(f"  {config}: {count}/{total} ({count/total*100:.1f}%)")


def analyze_achievability(df_raw):
    """Analyze which configs can achieve each DSSIM target."""
    print("\n" + "=" * 80)
    print("ACHIEVABILITY BY DSSIM TARGET")
    print("=" * 80)

    for target in DSSIM_TARGETS:
        print(f"\nDSSIM <= {target}:")
        for config in CONFIGS:
            config_df = df_raw[df_raw['config_key'] == config]
            min_dssim_per_image = config_df.groupby('source_hash')['dssim'].min()
            can_achieve = (min_dssim_per_image <= target).sum()
            total = len(min_dssim_per_image)
            print(f"  {config}: {can_achieve}/{total} images ({can_achieve/total*100:.1f}%)")


def analyze_baseline_strategies(df):
    """Analyze regret for baseline strategies."""
    print("\n" + "=" * 80)
    print("BASELINE STRATEGY REGRET (vs optimal)")
    print("=" * 80)

    for config in CONFIGS:
        regrets = df[f'regret_{config}'].dropna()
        if len(regrets) > 0:
            print(f"\nAlways {config}:")
            print(f"  Samples: {len(regrets)}")
            print(f"  Mean regret: {regrets.mean():.2f}%")
            print(f"  Median: {np.median(regrets):.2f}%")
            print(f"  95th pct: {np.percentile(regrets, 95):.2f}%")
            print(f"  Max: {regrets.max():.2f}%")


def test_heuristics(df):
    """Test various DSSIM-targeted heuristics."""
    print("\n" + "=" * 80)
    print("HEURISTIC TESTS")
    print("=" * 80)

    def evaluate(heuristic_fn, name):
        regrets = []
        correct = 0
        total = 0

        for _, row in df.iterrows():
            pred = heuristic_fn(row)
            regret_col = f'regret_{pred}'
            if pd.notna(row.get(regret_col)):
                regrets.append(row[regret_col])
                total += 1
                if pred == row['optimal_config']:
                    correct += 1

        regrets = np.array(regrets)
        print(f"\n{name}:")
        print(f"  Accuracy: {correct/total*100:.1f}%")
        print(f"  Mean regret: {regrets.mean():.2f}%")
        print(f"  Median: {np.median(regrets):.2f}%")
        print(f"  95th pct: {np.percentile(regrets, 95):.2f}%")
        return regrets.mean()

    # Test 1: Quality-based (DSSIM thresholds)
    def quality_based(row):
        dssim = row['target_dssim']
        if dssim <= 0.002:
            return 'jpegli-444'
        if dssim <= 0.01:
            return 'jpegli-420'
        return 'mozjpeg-420'
    evaluate(quality_based, "DSSIM<=0.002: 444, <=0.01: j420, else m420")

    # Test 2: More aggressive jpegli
    def jpegli_heavy(row):
        dssim = row['target_dssim']
        if dssim <= 0.003:
            return 'jpegli-444'
        return 'jpegli-420'
    evaluate(jpegli_heavy, "DSSIM<=0.003: 444, else jpegli-420")

    # Test 3: mozjpeg for low quality
    def mozjpeg_low(row):
        dssim = row['target_dssim']
        if dssim <= 0.002:
            return 'jpegli-444'
        if dssim <= 0.005:
            return 'jpegli-420'
        if dssim <= 0.02:
            return 'mozjpeg-420'
        return 'mozjpeg-420'
    evaluate(mozjpeg_low, "<=0.002: j444, <=0.005: j420, else m420")

    # Test 4: Feature-aware
    def feature_aware(row):
        dssim = row['target_dssim']
        ed = row['edge_density']
        cc = row['chroma_complexity']

        # Very high quality: 444
        if dssim <= 0.002:
            return 'jpegli-444'

        # High quality with high chroma: 444
        if dssim <= 0.005 and cc > 0.15:
            return 'jpegli-444'

        # Medium quality
        if dssim <= 0.015:
            return 'jpegli-420'

        # Low quality: mozjpeg
        if cc > 0.18:
            return 'mozjpeg-444'
        return 'mozjpeg-420'
    evaluate(feature_aware, "Feature-aware heuristic")

    # Test 5: Edge-density aware
    def edge_aware(row):
        dssim = row['target_dssim']
        ed = row['edge_density']
        cc = row['chroma_complexity']
        uf = row['uniform_block_fraction']

        if dssim <= 0.002:
            return 'jpegli-444'

        if dssim <= 0.005:
            if cc > 0.14 or ed <= 0.04:
                return 'jpegli-444'
            return 'jpegli-420'

        if dssim <= 0.015:
            if ed <= 0.03 and uf > 0.05:
                return 'mozjpeg-420'
            return 'jpegli-420'

        # Low quality
        if ed <= 0.04 and cc > 0.17:
            return 'mozjpeg-444'
        return 'mozjpeg-420'
    evaluate(edge_aware, "Edge-density aware heuristic")

    # Test 6: Simple BPP-proxy based
    def bpp_proxy(row):
        dssim = row['target_dssim']
        cc = row['chroma_complexity']

        # Map DSSIM to approximate BPP range
        # Lower DSSIM = higher quality = higher BPP
        if dssim <= 0.003:
            # Very high quality (high BPP) - jpegli-444
            return 'jpegli-444'
        if dssim <= 0.01:
            # High quality - jpegli-420
            if cc > 0.16:
                return 'jpegli-444'
            return 'jpegli-420'
        if dssim <= 0.025:
            # Medium quality - still jpegli territory
            return 'jpegli-420'
        # Low quality - mozjpeg
        return 'mozjpeg-420'
    evaluate(bpp_proxy, "BPP-proxy heuristic")


def analyze_dssim_limits(df_raw):
    """Analyze min achievable DSSIM (best quality) per codec."""
    print("\n" + "=" * 80)
    print("DSSIM LIMITS BY CODEC (lower = better quality)")
    print("=" * 80)

    for config in CONFIGS:
        config_df = df_raw[df_raw['config_key'] == config]

        print(f"\n{config}:")
        print(f"  Overall DSSIM range: {config_df['dssim'].min():.6f} to {config_df['dssim'].max():.6f}")

        min_dssim_per_image = config_df.groupby('source_hash')['dssim'].min()
        max_dssim_per_image = config_df.groupby('source_hash')['dssim'].max()

        print(f"  Min DSSIM per image: min={min_dssim_per_image.min():.6f}, mean={min_dssim_per_image.mean():.6f}")
        print(f"  Max DSSIM per image: max={max_dssim_per_image.max():.6f}, mean={max_dssim_per_image.mean():.6f}")


def find_crossover_point(df):
    """Find where mozjpeg becomes better than jpegli."""
    print("\n" + "=" * 80)
    print("CROSSOVER ANALYSIS: When does mozjpeg win?")
    print("=" * 80)

    for target in DSSIM_TARGETS:
        target_df = df[df['target_dssim'] == target]
        if len(target_df) == 0:
            continue

        jpegli_wins = target_df['optimal_config'].str.startswith('jpegli').sum()
        mozjpeg_wins = target_df['optimal_config'].str.startswith('mozjpeg').sum()
        total = len(target_df)

        if total > 0:
            j_pct = jpegli_wins / total * 100
            m_pct = mozjpeg_wins / total * 100
            winner = "jpegli" if jpegli_wins > mozjpeg_wins else "mozjpeg"
            print(f"DSSIM {target}: jpegli {j_pct:.0f}% vs mozjpeg {m_pct:.0f}% -> {winner}")


def generate_heuristic():
    """Generate the final DSSIM-based heuristic."""
    print("\n" + "=" * 80)
    print("FINAL DSSIM-BASED HEURISTIC")
    print("=" * 80)

    print("""
/// Select optimal codec configuration for a target DSSIM value.
///
/// DSSIM: 0 = identical, higher = worse quality.
/// Typical ranges:
///   - 0.001 = very high quality (visually lossless)
///   - 0.005 = high quality
///   - 0.01  = medium-high quality
///   - 0.02  = medium quality
///   - 0.03+ = low quality
///
/// Based on regret-minimization analysis across 86 images.
pub fn select_codec_for_dssim(
    analysis: &ImageAnalysis,
    target_dssim: f32,
) -> CodecRecommendation {
    let chroma_complexity = analysis.chroma.chroma_quality;
    let edge_density = analysis.edge_density;

    // VERY HIGH QUALITY (DSSIM <= 0.002): jpegli-444 required
    if target_dssim <= 0.002 {
        return CodecRecommendation::Jpegli { subsampling: Subsampling::S444 };
    }

    // HIGH QUALITY (DSSIM 0.002-0.005): jpegli, 444 for chroma-rich
    if target_dssim <= 0.005 {
        if chroma_complexity > 0.14 || edge_density <= 0.04 {
            return CodecRecommendation::Jpegli { subsampling: Subsampling::S444 };
        }
        return CodecRecommendation::Jpegli { subsampling: Subsampling::S420 };
    }

    // MEDIUM QUALITY (DSSIM 0.005-0.015): jpegli-420
    if target_dssim <= 0.015 {
        return CodecRecommendation::Jpegli { subsampling: Subsampling::S420 };
    }

    // LOW QUALITY (DSSIM 0.015-0.03): mozjpeg starts winning
    if target_dssim <= 0.03 {
        if edge_density <= 0.04 && chroma_complexity > 0.17 {
            return CodecRecommendation::MozJpeg { subsampling: Subsampling::S444 };
        }
        return CodecRecommendation::MozJpeg { subsampling: Subsampling::S420 };
    }

    // VERY LOW QUALITY (DSSIM > 0.03): mozjpeg required
    if chroma_complexity > 0.18 {
        return CodecRecommendation::MozJpeg { subsampling: Subsampling::S444 };
    }
    CodecRecommendation::MozJpeg { subsampling: Subsampling::S420 }
}
""")


def main():
    df_raw = load_data()
    print(f"Loaded {len(df_raw)} rows, {df_raw['source_hash'].nunique()} images")
    print(f"DSSIM range: {df_raw['dssim'].min():.6f} to {df_raw['dssim'].max():.6f}")

    regret_df = build_dataset(df_raw, DSSIM_TARGETS)
    print(f"Dataset: {len(regret_df)} samples across {len(DSSIM_TARGETS)} DSSIM targets")

    analyze_dssim_limits(df_raw)
    analyze_achievability(df_raw)
    analyze_winners_by_target(regret_df)
    analyze_baseline_strategies(regret_df)
    find_crossover_point(regret_df)
    test_heuristics(regret_df)
    generate_heuristic()


if __name__ == '__main__':
    main()
