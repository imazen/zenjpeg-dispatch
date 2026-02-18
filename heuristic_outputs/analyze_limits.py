#!/usr/bin/env python3
"""
Analyze codec limits - when does jpegli fail to achieve target BA?

jpegli has a minimum quality floor. For very low quality (high BA, low BPP),
we need mozjpeg as a fallback.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

COLUMNS = [
    'source_hash', 'source_name', 'width', 'height',
    'variance', 'edge_density', 'chroma_complexity', 'uniform_block_fraction',
    'config_key', 'quality', 'cache_version', 'size_bytes',
    'bpp', 'butteraugli', 'ssimulacra2', 'dssim',
    'encode_time_ms', 'timestamp'
]

CONFIGS = ['jpegli-420', 'jpegli-444', 'mozjpeg-420', 'mozjpeg-444']


def load_data(csv_path='results.csv'):
    return pd.read_csv(csv_path, names=COLUMNS)


def analyze_ba_limits(df):
    """Analyze max achievable Butteraugli (worst quality) per codec."""
    print("="*80)
    print("BUTTERAUGLI LIMITS BY CODEC (higher = worse quality)")
    print("="*80)

    for config in CONFIGS:
        config_df = df[df['config_key'] == config]

        print(f"\n{config}:")
        print(f"  Overall BA range: {config_df['butteraugli'].min():.2f} to {config_df['butteraugli'].max():.2f}")
        print(f"  Overall BPP range: {config_df['bpp'].min():.3f} to {config_df['bpp'].max():.3f}")

        # Per-image max BA (worst quality achievable)
        max_ba_per_image = config_df.groupby('source_hash')['butteraugli'].max()
        min_bpp_per_image = config_df.groupby('source_hash')['bpp'].min()

        print(f"  Max BA per image: min={max_ba_per_image.min():.2f}, mean={max_ba_per_image.mean():.2f}, max={max_ba_per_image.max():.2f}")
        print(f"  Min BPP per image: min={min_bpp_per_image.min():.3f}, mean={min_bpp_per_image.mean():.3f}, max={min_bpp_per_image.max():.3f}")


def analyze_achievability_by_target(df):
    """For each BA target, how many images can each codec achieve?"""
    print("\n" + "="*80)
    print("ACHIEVABILITY BY TARGET BUTTERAUGLI")
    print("="*80)

    targets = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]

    for target in targets:
        print(f"\nTarget BA >= {target}:")
        for config in CONFIGS:
            config_df = df[df['config_key'] == config]
            max_ba_per_image = config_df.groupby('source_hash')['butteraugli'].max()
            can_achieve = (max_ba_per_image >= target).sum()
            total = len(max_ba_per_image)
            print(f"  {config}: {can_achieve}/{total} images ({can_achieve/total*100:.1f}%)")


def analyze_low_bpp_region(df):
    """Analyze which codec wins at very low BPP."""
    print("\n" + "="*80)
    print("LOW BPP ANALYSIS (where mozjpeg might be needed)")
    print("="*80)

    bpp_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    for bpp_thresh in bpp_thresholds:
        low_bpp = df[df['bpp'] <= bpp_thresh]

        if len(low_bpp) == 0:
            print(f"\nBPP <= {bpp_thresh}: No data")
            continue

        print(f"\nBPP <= {bpp_thresh} ({len(low_bpp)} samples):")

        # Which configs can achieve this BPP?
        for config in CONFIGS:
            config_low = low_bpp[low_bpp['config_key'] == config]
            n_images = config_low['source_hash'].nunique()
            if len(config_low) > 0:
                avg_ba = config_low['butteraugli'].mean()
                min_ba = config_low['butteraugli'].min()
                print(f"  {config}: {n_images} images, BA range {min_ba:.1f}-{config_low['butteraugli'].max():.1f}, avg={avg_ba:.1f}")


def analyze_jpegli_floor(df):
    """Find jpegli's quality floor."""
    print("\n" + "="*80)
    print("JPEGLI QUALITY FLOOR ANALYSIS")
    print("="*80)

    for config in ['jpegli-420', 'jpegli-444']:
        config_df = df[df['config_key'] == config]

        # Find lowest quality (highest BA) achievable per image
        worst_quality = config_df.groupby('source_hash').agg({
            'butteraugli': 'max',
            'bpp': 'min',
            'quality': 'min'  # The quality parameter used
        }).reset_index()

        print(f"\n{config} worst achievable quality:")
        print(f"  BA distribution: min={worst_quality['butteraugli'].min():.2f}, "
              f"median={worst_quality['butteraugli'].median():.2f}, "
              f"max={worst_quality['butteraugli'].max():.2f}")
        print(f"  BPP distribution: min={worst_quality['bpp'].min():.3f}, "
              f"median={worst_quality['bpp'].median():.3f}, "
              f"max={worst_quality['bpp'].max():.3f}")
        print(f"  Quality param used: min={worst_quality['quality'].min()}, max={worst_quality['quality'].max()}")

        # Images where jpegli can't go below certain BA
        cant_reach_8 = (worst_quality['butteraugli'] < 8).sum()
        cant_reach_10 = (worst_quality['butteraugli'] < 10).sum()
        print(f"  Images that CAN'T reach BA>=8: {cant_reach_8}/{len(worst_quality)}")
        print(f"  Images that CAN'T reach BA>=10: {cant_reach_10}/{len(worst_quality)}")


def analyze_mozjpeg_advantage(df):
    """Find where mozjpeg is NECESSARY (jpegli can't achieve target)."""
    print("\n" + "="*80)
    print("WHERE MOZJPEG IS NECESSARY (jpegli can't reach target)")
    print("="*80)

    # Get max BA per image per config
    max_ba = df.groupby(['source_hash', 'config_key'])['butteraugli'].max().unstack()

    # Find images where mozjpeg can go lower quality than jpegli
    jpegli_best_max = max_ba[['jpegli-420', 'jpegli-444']].max(axis=1)
    mozjpeg_best_max = max_ba[['mozjpeg-420', 'mozjpeg-444']].max(axis=1)

    print("\nMax achievable BA comparison:")
    print(f"  jpegli best max BA: mean={jpegli_best_max.mean():.2f}, median={jpegli_best_max.median():.2f}")
    print(f"  mozjpeg best max BA: mean={mozjpeg_best_max.mean():.2f}, median={mozjpeg_best_max.median():.2f}")

    # Where mozjpeg can go higher BA (worse quality) than jpegli
    mozjpeg_goes_lower = mozjpeg_best_max > jpegli_best_max
    print(f"\nImages where mozjpeg can achieve WORSE quality than jpegli: {mozjpeg_goes_lower.sum()}/{len(mozjpeg_goes_lower)}")

    # For those images, how much further can mozjpeg go?
    diff = mozjpeg_best_max[mozjpeg_goes_lower] - jpegli_best_max[mozjpeg_goes_lower]
    if len(diff) > 0:
        print(f"  BA difference: mean={diff.mean():.2f}, max={diff.max():.2f}")


def find_crossover_point(df):
    """Find the BA threshold where mozjpeg becomes necessary."""
    print("\n" + "="*80)
    print("CROSSOVER ANALYSIS: When to switch to mozjpeg")
    print("="*80)

    # Get min BPP per image per config at various BA targets
    results = []

    for target_ba in [5, 6, 7, 8, 9, 10, 12]:
        # Filter to samples at or above this BA
        high_ba = df[df['butteraugli'] >= target_ba]

        if len(high_ba) == 0:
            continue

        # For each image, find min BPP to achieve this BA for each config
        for source_hash, img_group in high_ba.groupby('source_hash'):
            row = {'source_hash': source_hash, 'target_ba': target_ba}

            for config in CONFIGS:
                config_data = img_group[img_group['config_key'] == config]
                if len(config_data) > 0:
                    row[f'min_bpp_{config}'] = config_data['bpp'].min()
                else:
                    row[f'min_bpp_{config}'] = np.nan

            results.append(row)

    results_df = pd.DataFrame(results)

    # Analyze by target BA
    for target_ba in [5, 6, 7, 8, 9, 10]:
        target_df = results_df[results_df['target_ba'] == target_ba]
        if len(target_df) == 0:
            continue

        print(f"\nTarget BA >= {target_ba}:")

        # Count which config has lowest BPP
        winners = []
        for _, row in target_df.iterrows():
            bpps = {c: row.get(f'min_bpp_{c}', np.nan) for c in CONFIGS}
            valid = {k: v for k, v in bpps.items() if not np.isnan(v)}
            if valid:
                winner = min(valid, key=valid.get)
                winners.append(winner)

        if winners:
            from collections import Counter
            counts = Counter(winners)
            total = len(winners)
            for config, count in counts.most_common():
                print(f"  {config}: {count} wins ({count/total*100:.1f}%)")


def generate_heuristic_with_fallback():
    """Generate heuristic that falls back to mozjpeg for very low quality."""
    print("\n" + "="*80)
    print("HEURISTIC WITH MOZJPEG FALLBACK FOR LOW QUALITY")
    print("="*80)

    print("""
/// Select codec for target Butteraugli, with mozjpeg fallback for very low quality.
///
/// jpegli has a quality floor - it can't produce very low quality outputs.
/// For BA > 8 targets, mozjpeg may be necessary.
pub fn select_codec_with_fallback(
    edge_density: f32,
    chroma_complexity: f32,
    uniform_block_fraction: f32,
    target_butteraugli: f32,
) -> Config {
    // VERY LOW QUALITY (BA > 8): mozjpeg territory
    // jpegli often can't achieve these targets
    if target_butteraugli > 8.0 {
        if edge_density <= 0.04 {
            // Low edge density: mozjpeg-444 for chroma preservation
            if chroma_complexity > 0.18 {
                return Config::MozJpeg { subsampling: Subsampling::S444 };
            }
        }
        // Default for very low quality: mozjpeg-420
        return Config::MozJpeg { subsampling: Subsampling::S420 };
    }

    // LOW QUALITY (BA 6-8): mixed, but jpegli still viable
    if target_butteraugli > 6.0 {
        // Try jpegli first, but mozjpeg for specific patterns
        if edge_density > 0.12 && chroma_complexity < 0.19 {
            return Config::MozJpeg { subsampling: Subsampling::S420 };
        }
        if uniform_block_fraction <= 0.02 {
            return Config::MozJpeg { subsampling: Subsampling::S420 };
        }
        // Otherwise jpegli-420
        return Config::Jpegli { subsampling: Subsampling::S420 };
    }

    // HIGH QUALITY (BA <= 2): jpegli-444
    if target_butteraugli <= 2.0 {
        return Config::Jpegli { subsampling: Subsampling::S444 };
    }

    // MEDIUM-HIGH QUALITY (BA 2-3.5): jpegli-444 for chroma-rich
    if target_butteraugli <= 3.5 {
        if chroma_complexity > 0.14 {
            return Config::Jpegli { subsampling: Subsampling::S444 };
        }
    }

    // DEFAULT: jpegli-420
    Config::Jpegli { subsampling: Subsampling::S420 }
}
""")


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows, {df['source_hash'].nunique()} images")

    analyze_ba_limits(df)
    analyze_achievability_by_target(df)
    analyze_low_bpp_region(df)
    analyze_jpegli_floor(df)
    analyze_mozjpeg_advantage(df)
    find_crossover_point(df)
    generate_heuristic_with_fallback()


if __name__ == '__main__':
    main()
