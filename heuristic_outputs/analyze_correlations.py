#!/usr/bin/env python3
"""Analyze benchmark results and compute correlations between image characteristics and encoder wins."""

import pandas as pd
import numpy as np
from scipy import stats

# Column names for the CSV (it has no header)
COLUMNS = [
    'source_hash', 'source_name', 'width', 'height',
    'variance', 'edge_density', 'chroma_complexity', 'uniform_block_fraction',
    'config_key', 'quality', 'cache_version', 'size_bytes',
    'bpp', 'butteraugli', 'ssimulacra2', 'dssim',
    'encode_time_ms', 'timestamp'
]

def load_data(csv_path='results.csv'):
    """Load and prepare the benchmark data."""
    df = pd.read_csv(csv_path, names=COLUMNS)
    print(f"Loaded {len(df)} rows")
    print(f"Images: {df['source_hash'].nunique()}")
    print(f"Configs: {df['config_key'].unique().tolist()}")
    return df

def find_winner(group, metric='ssimulacra2', lower_is_better=False):
    """Find the config with best quality at the lowest BPP."""
    if lower_is_better:
        best_idx = group[metric].idxmin()
    else:
        best_idx = group[metric].idxmax()
    return group.loc[best_idx, 'config_key']

def analyze_by_bpp_range(df, bpp_ranges):
    """Analyze winners in each BPP range."""
    results = []

    for bpp_min, bpp_max, range_name in bpp_ranges:
        range_df = df[(df['bpp'] >= bpp_min) & (df['bpp'] < bpp_max)]

        if len(range_df) == 0:
            continue

        # For each image in this range, find the best config at similar BPP
        for metric in ['ssimulacra2', 'butteraugli', 'dssim']:
            lower_is_better = metric in ['butteraugli', 'dssim']

            # Group by image and find winner for each
            winners_by_image = {}
            for source_hash, img_group in range_df.groupby('source_hash'):
                # For each config, get the best quality in this BPP range
                config_scores = {}
                for config_key, config_group in img_group.groupby('config_key'):
                    if len(config_group) == 0:
                        continue
                    if lower_is_better:
                        best_score = config_group[metric].min()
                    else:
                        best_score = config_group[metric].max()
                    config_scores[config_key] = best_score

                if config_scores:
                    if lower_is_better:
                        winner = min(config_scores, key=config_scores.get)
                    else:
                        winner = max(config_scores, key=config_scores.get)
                    winners_by_image[source_hash] = winner

            # Count wins by config
            win_counts = {}
            for winner in winners_by_image.values():
                win_counts[winner] = win_counts.get(winner, 0) + 1

            total = len(winners_by_image)
            if total == 0:
                continue

            results.append({
                'bpp_range': range_name,
                'metric': metric,
                'total_images': total,
                'wins': win_counts
            })

    return results

def compute_correlations(df, bpp_ranges):
    """Compute point-biserial correlations between image characteristics and encoder wins."""
    correlations = []

    image_chars = ['variance', 'edge_density', 'chroma_complexity', 'uniform_block_fraction']

    for bpp_min, bpp_max, range_name in bpp_ranges:
        range_df = df[(df['bpp'] >= bpp_min) & (df['bpp'] < bpp_max)]

        if len(range_df) == 0:
            continue

        for metric in ['ssimulacra2', 'butteraugli']:
            lower_is_better = metric == 'butteraugli'

            # Get winner for each image
            winners_data = []
            for source_hash, img_group in range_df.groupby('source_hash'):
                # Get image characteristics (same for all rows of this image)
                chars = img_group[image_chars].iloc[0].to_dict()
                chars['source_hash'] = source_hash

                # Find winner
                config_scores = {}
                for config_key, config_group in img_group.groupby('config_key'):
                    if lower_is_better:
                        best_score = config_group[metric].min()
                    else:
                        best_score = config_group[metric].max()
                    config_scores[config_key] = best_score

                if config_scores:
                    if lower_is_better:
                        winner = min(config_scores, key=config_scores.get)
                    else:
                        winner = max(config_scores, key=config_scores.get)
                    chars['winner'] = winner
                    winners_data.append(chars)

            if len(winners_data) < 10:
                continue

            winners_df = pd.DataFrame(winners_data)

            # For each pair of configs, compute correlations
            configs = winners_df['winner'].unique()
            for config_a in configs:
                for config_b in configs:
                    if config_a >= config_b:
                        continue

                    # Binary variable: 1 if config_a wins, 0 if config_b wins
                    subset = winners_df[winners_df['winner'].isin([config_a, config_b])]
                    if len(subset) < 10:
                        continue

                    wins_a = (subset['winner'] == config_a).astype(int)

                    for char in image_chars:
                        try:
                            corr, pval = stats.pointbiserialr(wins_a, subset[char])
                            if not np.isnan(corr):
                                correlations.append({
                                    'bpp_range': range_name,
                                    'metric': metric,
                                    'config_a': config_a,
                                    'config_b': config_b,
                                    'characteristic': char,
                                    'correlation': corr,
                                    'p_value': pval,
                                    'n': len(subset),
                                    'wins_a': wins_a.sum(),
                                    'wins_b': len(subset) - wins_a.sum()
                                })
                        except:
                            pass

    return pd.DataFrame(correlations)

def main():
    df = load_data()

    print("\n" + "="*80)
    print("BPP Range Analysis")
    print("="*80)

    bpp_ranges = [
        (0.2, 0.5, "Very Low (0.2-0.5)"),
        (0.5, 1.0, "Low (0.5-1.0)"),
        (1.0, 1.5, "Medium (1.0-1.5)"),
        (1.5, 2.0, "High (1.5-2.0)"),
        (2.0, 3.0, "Very High (2.0-3.0)")
    ]

    results = analyze_by_bpp_range(df, bpp_ranges)

    for r in results:
        print(f"\n{r['bpp_range']} - {r['metric']} ({r['total_images']} images):")
        for config, wins in sorted(r['wins'].items(), key=lambda x: -x[1]):
            pct = 100 * wins / r['total_images']
            print(f"  {config}: {wins} wins ({pct:.1f}%)")

    print("\n" + "="*80)
    print("Point-Biserial Correlations (significant at p<0.05)")
    print("="*80)

    corr_df = compute_correlations(df, bpp_ranges)

    if len(corr_df) > 0:
        # Filter to significant correlations
        sig_corr = corr_df[corr_df['p_value'] < 0.05].sort_values('p_value')

        if len(sig_corr) > 0:
            print("\nSignificant correlations found:")
            for _, row in sig_corr.iterrows():
                direction = "higher" if row['correlation'] > 0 else "lower"
                print(f"\n  {row['bpp_range']} | {row['metric']}")
                print(f"    {row['config_a']} vs {row['config_b']}")
                print(f"    {row['characteristic']}: r={row['correlation']:.3f}, p={row['p_value']:.4f}")
                print(f"    ({direction} {row['characteristic']} favors {row['config_a'] if row['correlation'] > 0 else row['config_b']})")
                print(f"    n={row['n']}, wins: {row['config_a']}={row['wins_a']}, {row['config_b']}={row['wins_b']}")
        else:
            print("\nNo significant correlations found (p<0.05)")
            print("\nAll correlations:")
            print(corr_df.to_string())
    else:
        print("\nNot enough data for correlation analysis")

    # Save correlations to CSV
    if len(corr_df) > 0:
        corr_df.to_csv('correlations.csv', index=False)
        print("\nCorrelations saved to correlations.csv")

if __name__ == '__main__':
    main()
