#!/usr/bin/env python3
"""
Statistical analysis of quality mapping between mozjpeg and jpegli.
Checks variance, confidence intervals, and subsampling effects.
"""

import pandas as pd
import numpy as np
from scipy import stats

def load_data(csv_path='results_clean.csv'):
    """Load benchmark results."""
    df = pd.read_csv(csv_path, header=None, names=[
        'run_hash', 'image', 'width', 'height', 'luma_complexity',
        'edge_density', 'flat_block_pct', 'chroma_quality',
        'config', 'quality', 'version', 'size', 'bpp',
        'butteraugli', 'ssim2', 'dssim', 'encode_ms', 'timestamp'
    ])
    return df

def analyze_variance(df, config, metric):
    """Analyze variance of metric values at each quality level."""
    subset = df[df['config'] == config]

    stats_by_q = subset.groupby('quality')[metric].agg(['count', 'mean', 'median', 'std',
                                                         lambda x: x.quantile(0.25),
                                                         lambda x: x.quantile(0.75)])
    stats_by_q.columns = ['count', 'mean', 'median', 'std', 'q25', 'q75']
    stats_by_q['iqr'] = stats_by_q['q75'] - stats_by_q['q25']
    stats_by_q['cv'] = stats_by_q['std'] / stats_by_q['mean'].abs()  # Coefficient of variation

    return stats_by_q

def check_mapping_consistency(df, source_config, target_config, metric):
    """Check if mapping is consistent across individual images."""

    # Get unique images that have data for both configs
    source_images = set(df[df['config'] == source_config]['image'].unique())
    target_images = set(df[df['config'] == target_config]['image'].unique())
    common_images = source_images & target_images

    print(f"\nImages with both {source_config} and {target_config}: {len(common_images)}")

    # For each image, find the mapping at specific quality levels
    mappings_by_image = []

    for img in list(common_images)[:50]:  # Sample 50 images for speed
        source_data = df[(df['config'] == source_config) & (df['image'] == img)]
        target_data = df[(df['config'] == target_config) & (df['image'] == img)]

        for source_q in [50, 70, 80, 90]:
            source_row = source_data[source_data['quality'] == source_q]
            if len(source_row) == 0:
                continue

            source_val = source_row[metric].values[0]

            # Find closest target quality
            target_data_sorted = target_data.copy()
            target_data_sorted['diff'] = (target_data_sorted[metric] - source_val).abs()
            if len(target_data_sorted) == 0:
                continue

            best_match = target_data_sorted.loc[target_data_sorted['diff'].idxmin()]

            mappings_by_image.append({
                'image': img,
                'source_q': source_q,
                'target_q': best_match['quality'],
                'source_val': source_val,
                'target_val': best_match[metric]
            })

    mappings_df = pd.DataFrame(mappings_by_image)

    if len(mappings_df) == 0:
        print("No mappings found!")
        return None

    # Analyze consistency
    print(f"\n{'Source Q':>10} | {'Target Q Mean':>13} | {'Target Q Std':>12} | {'Target Q Range':>14} | {'N':>5}")
    print("-" * 65)

    for source_q in [50, 70, 80, 90]:
        subset = mappings_df[mappings_df['source_q'] == source_q]
        if len(subset) == 0:
            continue

        target_mean = subset['target_q'].mean()
        target_std = subset['target_q'].std()
        target_min = subset['target_q'].min()
        target_max = subset['target_q'].max()

        print(f"{source_q:>10} | {target_mean:>13.1f} | {target_std:>12.1f} | {target_min:>3.0f} - {target_max:<3.0f}     | {len(subset):>5}")

    return mappings_df

def compare_subsampling_modes(df, metric):
    """Compare mappings for 420 vs 444 subsampling."""

    configs = [
        ('mozjpeg-420', 'jpegli-420', '4:2:0'),
        ('mozjpeg-444', 'jpegli-444', '4:4:4'),
    ]

    print(f"\n{'='*80}")
    print(f"SUBSAMPLING COMPARISON FOR {metric.upper()}")
    print(f"{'='*80}")

    results = {}

    for source, target, label in configs:
        source_metrics = df[df['config'] == source].groupby('quality')[metric].median().to_dict()
        target_metrics = df[df['config'] == target].groupby('quality')[metric].median().to_dict()

        mappings = {}
        for q in [50, 60, 70, 80, 85, 90, 95]:
            if q not in source_metrics:
                continue

            source_val = source_metrics[q]

            # Find equivalent target Q
            best_q = None
            best_diff = float('inf')
            for tq, tv in target_metrics.items():
                diff = abs(tv - source_val)
                if diff < best_diff:
                    best_diff = diff
                    best_q = tq

            mappings[q] = best_q

        results[label] = mappings

    # Print comparison table
    print(f"\n{'mozjpeg Q':>10} | {'jpegli equiv (420)':>18} | {'jpegli equiv (444)':>18} | {'Diff':>6}")
    print("-" * 60)

    for q in [50, 60, 70, 80, 85, 90, 95]:
        q_420 = results.get('4:2:0', {}).get(q, '-')
        q_444 = results.get('4:4:4', {}).get(q, '-')

        if isinstance(q_420, (int, float)) and isinstance(q_444, (int, float)):
            diff = q_444 - q_420
            diff_str = f"{diff:+d}"
        else:
            diff_str = "-"

        print(f"{q:>10} | {str(q_420):>18} | {str(q_444):>18} | {diff_str:>6}")

    return results

def statistical_significance_test(df, metric):
    """Test if jpegli produces significantly different metric values than mozjpeg at same Q."""

    print(f"\n{'='*80}")
    print(f"STATISTICAL SIGNIFICANCE TEST: mozjpeg vs jpegli at same Q ({metric})")
    print(f"{'='*80}")

    configs = [('mozjpeg-420', 'jpegli-420'), ('mozjpeg-444', 'jpegli-444')]

    for moz_config, jpegli_config in configs:
        print(f"\n{moz_config} vs {jpegli_config}:")
        print(f"{'Q':>5} | {'mozjpeg mean':>12} | {'jpegli mean':>12} | {'t-stat':>10} | {'p-value':>10} | {'Significant?':>12}")
        print("-" * 75)

        for q in [50, 70, 80, 90]:
            moz_vals = df[(df['config'] == moz_config) & (df['quality'] == q)][metric].values
            jpegli_vals = df[(df['config'] == jpegli_config) & (df['quality'] == q)][metric].values

            if len(moz_vals) < 5 or len(jpegli_vals) < 5:
                continue

            # Paired t-test would require same images, use independent t-test
            t_stat, p_val = stats.ttest_ind(moz_vals, jpegli_vals)

            moz_mean = np.mean(moz_vals)
            jpegli_mean = np.mean(jpegli_vals)
            sig = "YES" if p_val < 0.05 else "no"

            print(f"{q:>5} | {moz_mean:>12.4f} | {jpegli_mean:>12.4f} | {t_stat:>10.2f} | {p_val:>10.4f} | {sig:>12}")

def main():
    print("Loading benchmark data...")
    df = load_data()
    print(f"Loaded {len(df)} results from {df['image'].nunique()} unique images")

    # 1. Check variance at each quality level
    print("\n" + "="*80)
    print("VARIANCE ANALYSIS: How much does Butteraugli vary at each Q?")
    print("="*80)

    for config in ['mozjpeg-420', 'jpegli-420']:
        print(f"\n{config}:")
        stats = analyze_variance(df, config, 'butteraugli')
        print(stats[stats.index.isin([50, 70, 80, 90])].to_string())

    # 2. Check mapping consistency across images
    print("\n" + "="*80)
    print("MAPPING CONSISTENCY: Does the Q mapping vary by image?")
    print("="*80)

    print("\nButteraugli mapping consistency (mozjpeg-420 -> jpegli-420):")
    check_mapping_consistency(df, 'mozjpeg-420', 'jpegli-420', 'butteraugli')

    print("\nSSIMULACRA2 mapping consistency (mozjpeg-420 -> jpegli-420):")
    check_mapping_consistency(df, 'mozjpeg-420', 'jpegli-420', 'ssim2')

    # 3. Compare 420 vs 444 mappings
    for metric in ['butteraugli', 'ssim2', 'dssim']:
        compare_subsampling_modes(df, metric)

    # 4. Statistical significance
    for metric in ['butteraugli', 'ssim2']:
        statistical_significance_test(df, metric)

if __name__ == '__main__':
    main()
