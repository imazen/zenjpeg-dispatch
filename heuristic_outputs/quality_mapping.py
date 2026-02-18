#!/usr/bin/env python3
"""
Generate quality mapping tables between mozjpeg and jpegli
based on equivalent perceptual quality (Butteraugli, SSIM2, DSSIM).
"""

import pandas as pd
import numpy as np
from collections import defaultdict

def load_data(csv_path='results_clean.csv'):
    """Load benchmark results."""
    df = pd.read_csv(csv_path, header=None, names=[
        'run_hash', 'image', 'width', 'height', 'luma_complexity',
        'edge_density', 'flat_block_pct', 'chroma_quality',
        'config', 'quality', 'version', 'size', 'bpp',
        'butteraugli', 'ssim2', 'dssim', 'encode_ms', 'timestamp'
    ])
    return df

def get_median_metric_by_quality(df, config, metric):
    """Get median metric value for each quality level."""
    subset = df[df['config'] == config]
    return subset.groupby('quality')[metric].median().to_dict()

def find_equivalent_quality(source_metrics, target_metrics, source_q):
    """Find the target quality that gives closest metric value to source_q."""
    if source_q not in source_metrics:
        return None

    source_value = source_metrics[source_q]

    best_q = None
    best_diff = float('inf')

    for target_q, target_value in target_metrics.items():
        diff = abs(target_value - source_value)
        if diff < best_diff:
            best_diff = diff
            best_q = target_q

    return best_q, source_value, target_metrics.get(best_q)

def generate_mapping_table(df, source_config, target_config, metric, metric_name):
    """Generate a mapping table from source to target config based on metric."""
    source_metrics = get_median_metric_by_quality(df, source_config, metric)
    target_metrics = get_median_metric_by_quality(df, target_config, metric)

    print(f"\n{'='*70}")
    print(f"{source_config} -> {target_config} Quality Mapping (by {metric_name})")
    print(f"{'='*70}")
    print(f"{'Source Q':>10} {'Target Q':>10} {metric_name:>12} {'Target '+metric_name:>14}")
    print("-" * 50)

    mappings = []
    for q in range(10, 100, 5):
        result = find_equivalent_quality(source_metrics, target_metrics, q)
        if result:
            target_q, source_val, target_val = result
            print(f"{q:>10} {target_q:>10} {source_val:>12.4f} {target_val:>14.4f}")
            mappings.append((q, target_q, source_val, target_val))

    return mappings

def generate_bidirectional_table(df, config1, config2, metric, metric_name):
    """Generate a bidirectional mapping table."""
    metrics1 = get_median_metric_by_quality(df, config1, metric)
    metrics2 = get_median_metric_by_quality(df, config2, metric)

    print(f"\n{'='*80}")
    print(f"Bidirectional Quality Mapping: {config1} <-> {config2}")
    print(f"Based on: {metric_name}")
    print(f"{'='*80}")

    # Find common quality range
    qualities = sorted(set(metrics1.keys()) & set(metrics2.keys()))

    print(f"\n{'Q':>5} | {config1:>15} | {config2:>15} | {'Equiv '+config2+' Q':>18}")
    print("-" * 65)

    for q in qualities[::5]:  # Every 5th quality
        val1 = metrics1.get(q)
        val2 = metrics2.get(q)

        # Find equivalent quality in config2 for config1's metric value
        equiv_q = None
        if val1:
            best_diff = float('inf')
            for q2, v2 in metrics2.items():
                diff = abs(v2 - val1)
                if diff < best_diff:
                    best_diff = diff
                    equiv_q = q2

        if val1 and val2:
            print(f"{q:>5} | {val1:>15.4f} | {val2:>15.4f} | {equiv_q:>18}")

def generate_compact_mapping(df):
    """Generate compact mapping tables for practical use."""

    configs = {
        'mozjpeg': 'mozjpeg-420',
        'mozjpeg-max': 'mozjpeg-max-420',
        'jpegli': 'jpegli-420'
    }

    metrics = {
        'butteraugli': ('butteraugli', 'lower is better'),
        'ssim2': ('ssim2', 'higher is better'),
        'dssim': ('dssim', 'lower is better')
    }

    for metric_key, (metric_col, direction) in metrics.items():
        print(f"\n{'#'*80}")
        print(f"# QUALITY MAPPING BY {metric_key.upper()} ({direction})")
        print(f"{'#'*80}")

        # Get metrics for each config
        all_metrics = {}
        for name, config in configs.items():
            all_metrics[name] = get_median_metric_by_quality(df, config, metric_col)

        # Print header
        print(f"\n{'mozjpeg Q':>10} | {'jpegli equiv':>12} | {'mozjpeg-max equiv':>17} | {'mozjpeg '+metric_key:>15}")
        print("-" * 65)

        mozjpeg_metrics = all_metrics['mozjpeg']
        jpegli_metrics = all_metrics['jpegli']
        mozmax_metrics = all_metrics['mozjpeg-max']

        for q in range(10, 100, 5):
            if q not in mozjpeg_metrics:
                continue

            moz_val = mozjpeg_metrics[q]

            # Find equivalent jpegli Q
            jpegli_q = None
            best_diff = float('inf')
            for jq, jv in jpegli_metrics.items():
                diff = abs(jv - moz_val)
                if diff < best_diff:
                    best_diff = diff
                    jpegli_q = jq

            # Find equivalent mozjpeg-max Q
            mozmax_q = None
            best_diff = float('inf')
            for mq, mv in mozmax_metrics.items():
                diff = abs(mv - moz_val)
                if diff < best_diff:
                    best_diff = diff
                    mozmax_q = mq

            print(f"{q:>10} | {jpegli_q:>12} | {mozmax_q:>17} | {moz_val:>15.4f}")

        # Reverse mapping: jpegli -> mozjpeg
        print(f"\n{'jpegli Q':>10} | {'mozjpeg equiv':>13} | {'jpegli '+metric_key:>15}")
        print("-" * 45)

        for q in range(10, 100, 5):
            if q not in jpegli_metrics:
                continue

            jpegli_val = jpegli_metrics[q]

            # Find equivalent mozjpeg Q
            moz_q = None
            best_diff = float('inf')
            for mq, mv in mozjpeg_metrics.items():
                diff = abs(mv - jpegli_val)
                if diff < best_diff:
                    best_diff = diff
                    moz_q = mq

            print(f"{q:>10} | {moz_q:>13} | {jpegli_val:>15.4f}")

def main():
    print("Loading benchmark data...")
    df = load_data()

    print(f"Loaded {len(df)} results")
    print(f"Configs: {df['config'].unique()}")

    # Check which configs have enough data
    config_counts = df['config'].value_counts()
    print(f"\nResults per config:\n{config_counts}")

    # Generate compact mapping tables
    generate_compact_mapping(df)

    # Also generate detailed bidirectional tables
    for metric, name in [('butteraugli', 'Butteraugli'), ('ssim2', 'SSIMULACRA2'), ('dssim', 'DSSIM')]:
        generate_bidirectional_table(df, 'mozjpeg-420', 'jpegli-420', metric, name)

if __name__ == '__main__':
    main()
