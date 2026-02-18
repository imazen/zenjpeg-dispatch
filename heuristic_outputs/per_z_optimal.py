#!/usr/bin/env python3
"""
Per-Z optimal codec configuration analysis.

For each Z value (0-100), determine which codec configuration produces the
smallest file while meeting the quality target.

This analysis:
1. Uses Z = SSIM2 by design (directly compare against ssimulacra2 column)
2. For each Z, finds all results that meet or exceed that quality target
3. Among those, finds the smallest file for each image
4. Aggregates which codec wins most often and by how much
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

COLUMNS = [
    'source_hash', 'source_name', 'width', 'height',
    'variance', 'edge_density', 'chroma_complexity', 'uniform_block_fraction',
    'config_key', 'quality', 'cache_version', 'size_bytes',
    'bpp', 'butteraugli', 'ssimulacra2', 'dssim',
    'encode_time_ms', 'timestamp'
]

CONFIGS = ['jpegli-420', 'jpegli-444', 'mozjpeg-420', 'mozjpeg-444', 'mozjpeg-max-420', 'mozjpeg-max-444']


def load_data(csv_path='results_clean.csv'):
    df = pd.read_csv(csv_path, names=COLUMNS)
    # Filter out invalid rows
    df = df[df['butteraugli'] > 0]
    df = df[df['ssimulacra2'].notna()]
    df = df[df['dssim'] > 0]
    # Only analyze the 4 main configs
    df = df[df['config_key'].isin(CONFIGS)]
    return df


def analyze_per_z_optimal(df, optimize_for='ssimulacra2'):
    """
    For each Z level, find optimal codec configuration.

    optimize_for: 'ssimulacra2', 'butteraugli', or 'dssim'
    """
    results = []

    # Get unique images
    images = df['source_hash'].unique()
    print(f"Analyzing {len(images)} images across {len(df)} data points")

    # Group by image
    by_image = df.groupby('source_hash')

    for z in range(0, 101):
        # Convert Z to target metric value
        if optimize_for == 'ssimulacra2':
            # Z = SSIM2 directly
            target = z
            quality_filter = lambda d: d['ssimulacra2'] >= target
            metric_col = 'ssimulacra2'
        elif optimize_for == 'butteraugli':
            # Z → BA using fitted curve
            target = 8.942 * np.exp(-0.01411 * z)
            quality_filter = lambda d: d['butteraugli'] <= target
            metric_col = 'butteraugli'
        elif optimize_for == 'dssim':
            # Z → DSSIM using fitted curve
            target = 0.02277 * np.exp(-0.02589 * z)
            quality_filter = lambda d: d['dssim'] <= target
            metric_col = 'dssim'

        # For each image, find smallest file meeting quality target
        wins = defaultdict(int)
        sizes_by_config = defaultdict(list)
        quality_by_config = defaultdict(list)
        images_analyzed = 0

        for img_hash, img_df in by_image:
            # Filter to results meeting quality target
            meeting_target = quality_filter(img_df)
            qualified = img_df[meeting_target]

            if len(qualified) == 0:
                continue

            images_analyzed += 1

            # Find smallest file among qualified results
            smallest_idx = qualified['size_bytes'].idxmin()
            winner = qualified.loc[smallest_idx, 'config_key']
            wins[winner] += 1

            # Track sizes and quality for each config
            for config in CONFIGS:
                config_qualified = qualified[qualified['config_key'] == config]
                if len(config_qualified) > 0:
                    # Get smallest file for this config meeting target
                    min_idx = config_qualified['size_bytes'].idxmin()
                    sizes_by_config[config].append(config_qualified.loc[min_idx, 'size_bytes'])
                    quality_by_config[config].append(config_qualified.loc[min_idx, metric_col])

        # Calculate statistics
        total_wins = sum(wins.values())
        if total_wins == 0:
            results.append({
                'z': z,
                'target': target,
                'images_analyzed': 0,
                'winner': 'none',
                'winner_pct': 0,
            })
            continue

        # Find overall winner
        winner = max(wins.keys(), key=lambda k: wins[k])
        winner_pct = wins[winner] / total_wins * 100

        # Calculate mean sizes
        mean_sizes = {k: np.mean(v) if v else np.nan for k, v in sizes_by_config.items()}
        mean_quality = {k: np.mean(v) if v else np.nan for k, v in quality_by_config.items()}

        results.append({
            'z': z,
            'target': target,
            'images_analyzed': images_analyzed,
            'winner': winner,
            'winner_pct': winner_pct,
            'jpegli_420_wins': wins.get('jpegli-420', 0),
            'jpegli_444_wins': wins.get('jpegli-444', 0),
            'mozjpeg_420_wins': wins.get('mozjpeg-420', 0),
            'mozjpeg_444_wins': wins.get('mozjpeg-444', 0),
            'jpegli_420_pct': wins.get('jpegli-420', 0) / total_wins * 100,
            'jpegli_444_pct': wins.get('jpegli-444', 0) / total_wins * 100,
            'mozjpeg_420_pct': wins.get('mozjpeg-420', 0) / total_wins * 100,
            'mozjpeg_444_pct': wins.get('mozjpeg-444', 0) / total_wins * 100,
            'mozjpeg_max_420_pct': wins.get('mozjpeg-max-420', 0) / total_wins * 100,
            'mozjpeg_max_444_pct': wins.get('mozjpeg-max-444', 0) / total_wins * 100,
            'jpegli_420_mean_size': mean_sizes.get('jpegli-420', np.nan),
            'jpegli_444_mean_size': mean_sizes.get('jpegli-444', np.nan),
            'mozjpeg_420_mean_size': mean_sizes.get('mozjpeg-420', np.nan),
            'mozjpeg_444_mean_size': mean_sizes.get('mozjpeg-444', np.nan),
        })

    return pd.DataFrame(results)


def print_summary_table(results_df, metric_name):
    """Print a formatted summary table."""
    print(f"\n{'='*100}")
    print(f"PER-Z OPTIMAL CONFIG ANALYSIS (Optimizing for {metric_name})")
    print(f"{'='*100}")

    print(f"\n{'Z':>4} {'Target':>10} {'Images':>7} {'Winner':>16} {'Win%':>6} | "
          f"{'j420%':>6} {'j444%':>6} {'m420%':>6} {'m444%':>6} {'mx420%':>6} {'mx444%':>6}")
    print("-" * 120)

    # Print every 5th Z value for readability
    for _, row in results_df.iterrows():
        z = row['z']
        if z % 5 != 0:
            continue

        mx420 = row.get('mozjpeg_max_420_pct', 0.0)
        mx444 = row.get('mozjpeg_max_444_pct', 0.0)
        print(f"{z:4d} {row['target']:10.4f} {row['images_analyzed']:7d} "
              f"{row['winner']:>16} {row['winner_pct']:5.1f}% | "
              f"{row['jpegli_420_pct']:5.1f}% {row['jpegli_444_pct']:5.1f}% "
              f"{row['mozjpeg_420_pct']:5.1f}% {row['mozjpeg_444_pct']:5.1f}% "
              f"{mx420:5.1f}% {mx444:5.1f}%")


def analyze_crossover_points(results_df, metric_name):
    """Find Z values where the optimal codec changes."""
    print(f"\n{'='*80}")
    print(f"CROSSOVER POINTS ({metric_name})")
    print(f"{'='*80}")

    # Find jpegli vs mozjpeg crossover (mozjpeg includes max variants)
    results_df['jpegli_total_pct'] = results_df['jpegli_420_pct'] + results_df['jpegli_444_pct']
    results_df['mozjpeg_total_pct'] = (results_df['mozjpeg_420_pct'] + results_df['mozjpeg_444_pct'] +
                                        results_df.get('mozjpeg_max_420_pct', 0) + results_df.get('mozjpeg_max_444_pct', 0))

    prev_leader = None
    for _, row in results_df.iterrows():
        j_pct = row['jpegli_total_pct']
        m_pct = row['mozjpeg_total_pct']
        current_leader = 'jpegli' if j_pct > m_pct else 'mozjpeg'

        if prev_leader is not None and current_leader != prev_leader:
            print(f"Z={row['z']:3d}: {prev_leader} -> {current_leader} "
                  f"(jpegli {j_pct:.1f}% vs mozjpeg {m_pct:.1f}%)")

        prev_leader = current_leader

    # Summarize dominant ranges
    print(f"\nDominant codec by Z range:")

    # Find ranges where each codec dominates (>60%)
    jpegli_ranges = []
    mozjpeg_ranges = []

    start_z = 0
    current_dominant = None

    for _, row in results_df.iterrows():
        z = row['z']
        j_pct = row['jpegli_total_pct']
        m_pct = row['mozjpeg_total_pct']

        if j_pct > 60:
            dominant = 'jpegli'
        elif m_pct > 60:
            dominant = 'mozjpeg'
        else:
            dominant = 'mixed'

        if dominant != current_dominant:
            if current_dominant is not None:
                if current_dominant == 'jpegli':
                    jpegli_ranges.append((start_z, z - 1))
                elif current_dominant == 'mozjpeg':
                    mozjpeg_ranges.append((start_z, z - 1))
            start_z = z
            current_dominant = dominant

    # Close final range
    if current_dominant == 'jpegli':
        jpegli_ranges.append((start_z, 100))
    elif current_dominant == 'mozjpeg':
        mozjpeg_ranges.append((start_z, 100))

    print(f"  jpegli dominant (>60%): {jpegli_ranges}")
    print(f"  mozjpeg dominant (>60%): {mozjpeg_ranges}")


def generate_rust_heuristic(results_df, metric_name, optimize_for):
    """Generate Rust code for the optimal heuristic."""
    print(f"\n{'='*80}")
    print(f"RUST HEURISTIC CODE ({metric_name})")
    print(f"{'='*80}")

    # Simplify to key decision points
    # Find Z ranges with clear winners

    # Group Z values by winner
    winner_ranges = []
    current_winner = None
    start_z = 0

    for _, row in results_df.iterrows():
        z = int(row['z'])
        winner = row['winner']

        if winner != current_winner:
            if current_winner is not None:
                winner_ranges.append((start_z, z - 1, current_winner, row['winner_pct']))
            start_z = z
            current_winner = winner

    # Close final range
    if current_winner is not None:
        winner_ranges.append((start_z, 100, current_winner, row['winner_pct']))

    # Generate Rust code
    func_name = f"select_codec_for_{optimize_for}"
    target_type = "f32"

    print(f"""
/// Select optimal codec for target {optimize_for} based on per-Z analysis.
///
/// Generated from {len(results_df)} Z-level analyses.
#[must_use]
pub fn {func_name}_optimal(target_z: u8) -> CodecRecommendation {{
    match target_z {{""")

    for start, end, winner, pct in winner_ranges:
        # Parse winner into codec and subsampling
        parts = winner.split('-')
        codec = parts[0].capitalize()
        if len(parts) > 1:
            subsamp = f"S{parts[1].upper()}"
        else:
            subsamp = "S420"

        if start == end:
            print(f"        {start} => CodecRecommendation::{codec} {{ subsampling: Subsampling::{subsamp} }}, // {pct:.0f}%")
        else:
            print(f"        {start}..={end} => CodecRecommendation::{codec} {{ subsampling: Subsampling::{subsamp} }}, // {pct:.0f}%")

    print("""        _ => CodecRecommendation::Jpegli { subsampling: Subsampling::S420 },
    }
}
""")


def analyze_regret_per_z(df, optimize_for='ssimulacra2'):
    """
    Calculate regret for using a fixed strategy vs optimal per-image selection.

    Regret = (size_strategy - size_optimal) / size_optimal * 100
    """
    print(f"\n{'='*80}")
    print(f"REGRET ANALYSIS ({optimize_for})")
    print(f"{'='*80}")

    strategies = ['always-jpegli-420', 'always-jpegli-444',
                  'always-mozjpeg-420', 'always-mozjpeg-444',
                  'always-mozjpeg-max-420', 'always-mozjpeg-max-444']

    results = []

    for z in range(0, 101, 5):  # Every 5 Z values
        # Convert Z to target metric value
        if optimize_for == 'ssimulacra2':
            target = z
            quality_filter = lambda d: d['ssimulacra2'] >= target
        elif optimize_for == 'butteraugli':
            target = 8.942 * np.exp(-0.01411 * z)
            quality_filter = lambda d: d['butteraugli'] <= target
        elif optimize_for == 'dssim':
            target = 0.02277 * np.exp(-0.02589 * z)
            quality_filter = lambda d: d['dssim'] <= target

        by_image = df.groupby('source_hash')

        regrets = {s: [] for s in strategies}

        for img_hash, img_df in by_image:
            qualified = img_df[quality_filter(img_df)]

            if len(qualified) == 0:
                continue

            # Find optimal (smallest file meeting target)
            optimal_size = qualified['size_bytes'].min()

            # Calculate regret for each strategy
            for strategy in strategies:
                config = strategy.replace('always-', '')
                config_qualified = qualified[qualified['config_key'] == config]

                if len(config_qualified) > 0:
                    strategy_size = config_qualified['size_bytes'].min()
                    regret = (strategy_size - optimal_size) / optimal_size * 100
                    regrets[strategy].append(regret)

        # Calculate mean regret for each strategy
        row = {'z': z, 'target': target}
        for strategy in strategies:
            if regrets[strategy]:
                row[f'{strategy}_mean_regret'] = np.mean(regrets[strategy])
                row[f'{strategy}_max_regret'] = np.max(regrets[strategy])
            else:
                row[f'{strategy}_mean_regret'] = np.nan
                row[f'{strategy}_max_regret'] = np.nan

        results.append(row)

    regret_df = pd.DataFrame(results)

    print(f"\n{'Z':>4} | {'j420%':>7} {'j444%':>7} {'m420%':>7} {'m444%':>7} {'mx420%':>7} {'mx444%':>7}")
    print("-" * 70)

    for _, row in regret_df.iterrows():
        print(f"{int(row['z']):4d} | "
              f"{row.get('always-jpegli-420_mean_regret', np.nan):6.2f}% "
              f"{row.get('always-jpegli-444_mean_regret', np.nan):6.2f}% "
              f"{row.get('always-mozjpeg-420_mean_regret', np.nan):6.2f}% "
              f"{row.get('always-mozjpeg-444_mean_regret', np.nan):6.2f}% "
              f"{row.get('always-mozjpeg-max-420_mean_regret', np.nan):6.2f}% "
              f"{row.get('always-mozjpeg-max-444_mean_regret', np.nan):6.2f}%")

    # Find best single strategy
    print("\n\nBest single strategy (lowest mean regret across all Z levels):")
    for strategy in strategies:
        mean_regret = regret_df[f'{strategy}_mean_regret'].mean()
        max_regret = regret_df[f'{strategy}_max_regret'].max()
        print(f"  {strategy}: mean={mean_regret:.2f}%, max={max_regret:.2f}%")

    return regret_df


def main():
    import sys

    csv_path = 'results_clean.csv'
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    print(f"Loading data from {csv_path}...")
    df = load_data(csv_path)
    print(f"Loaded {len(df)} rows, {df['source_hash'].nunique()} unique images")

    # Analyze for each metric
    for metric in ['ssimulacra2', 'butteraugli', 'dssim']:
        print(f"\n\n{'#'*100}")
        print(f"# ANALYSIS FOR: {metric.upper()}")
        print(f"{'#'*100}")

        results = analyze_per_z_optimal(df, optimize_for=metric)
        print_summary_table(results, metric)
        analyze_crossover_points(results, metric)
        generate_rust_heuristic(results, metric, metric)
        analyze_regret_per_z(df, optimize_for=metric)

        # Save results
        results.to_csv(f'per_z_optimal_{metric}.csv', index=False)
        print(f"\nSaved to per_z_optimal_{metric}.csv")


if __name__ == '__main__':
    main()
