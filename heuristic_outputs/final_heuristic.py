#!/usr/bin/env python3
"""
Final overfit heuristic based on deep analysis.

Key patterns found:
1. At BPP < 0.3: mozjpeg competitive (48% of wins at ultra-low BPP)
2. mozjpeg-420 wins: edge_density > 0.04, target BA > 4.75, low uniform_block_fraction
3. mozjpeg-444 wins: low edge_density < 0.05, high chroma_complexity > 0.18, target BA > 6
4. jpegli-444 wins: target BA <= 2.75 OR (low edge_density + high chroma/uniform)
5. jpegli-420 wins: everything else (the default)
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
BUTTERAUGLI_TARGETS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0]


def load_data(csv_path='results.csv'):
    df = pd.read_csv(csv_path, names=COLUMNS)
    return df


def interpolate_bpp_at_butteraugli(group, target_ba):
    sorted_group = group.sort_values('butteraugli')
    ba_values = sorted_group['butteraugli'].values
    bpp_values = sorted_group['bpp'].values
    if target_ba < ba_values.min() or target_ba > ba_values.max():
        return None
    try:
        f = interpolate.interp1d(ba_values, bpp_values, kind='linear')
        return float(f(target_ba))
    except:
        return None


def get_all_config_bpp(img_df, target_ba):
    config_bpp = {}
    for config_key, config_group in img_df.groupby('config_key'):
        min_ba = config_group['butteraugli'].min()
        if min_ba <= target_ba:
            bpp = interpolate_bpp_at_butteraugli(config_group, target_ba)
            if bpp is not None and bpp > 0:
                config_bpp[config_key] = bpp
    return config_bpp


def build_dataset(df, target_values):
    data = []
    for source_hash, img_group in df.groupby('source_hash'):
        chars = img_group[FEATURES].iloc[0].to_dict()
        chars['source_hash'] = source_hash

        for target in target_values:
            config_bpp = get_all_config_bpp(img_group, target)
            if len(config_bpp) < 1:
                continue

            optimal_config = min(config_bpp, key=config_bpp.get)
            optimal_bpp = config_bpp[optimal_config]

            sample = chars.copy()
            sample['target_butteraugli'] = target
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


def overfit_heuristic_v1(row):
    """First attempt at overfit heuristic."""
    ba = row['target_butteraugli']
    ed = row['edge_density']
    cc = row['chroma_complexity']
    uf = row['uniform_block_fraction']
    var = row['variance']

    # Rule 1: High quality (low BA) -> jpegli-444
    if ba <= 2.0:
        return 'jpegli-444'

    # Rule 2: Medium-high quality with specific characteristics
    if ba <= 3.0:
        if cc > 0.14 or uf > 0.03:
            return 'jpegli-444'
        return 'jpegli-420'

    # Rule 3: Low quality (high BA > 6) - mozjpeg territory
    if ba > 6.0:
        if ed <= 0.04:
            return 'mozjpeg-444'
        if ed > 0.12 and cc < 0.19:
            return 'mozjpeg-420'
        if uf <= 0.02:
            return 'mozjpeg-420'

    # Rule 4: Very low quality (BA > 8)
    if ba > 8.0:
        if ed > 0.05 and uf < 0.1:
            return 'mozjpeg-420'

    return 'jpegli-420'


def overfit_heuristic_v2(row):
    """More aggressive overfit based on decision tree patterns."""
    ba = row['target_butteraugli']
    ed = row['edge_density']
    cc = row['chroma_complexity']
    uf = row['uniform_block_fraction']
    var = row['variance']

    # HIGH QUALITY: BA <= 2.75
    if ba <= 2.75:
        if cc <= 0.14:
            if ed <= 0.05:
                if ba <= 1.25:
                    if var <= 3781 and uf <= 0.49:
                        return 'jpegli-444'
                    return 'jpegli-420'
                else:
                    return 'jpegli-420'
            else:
                if uf > 0.03:
                    return 'jpegli-444'
                return 'jpegli-420'
        else:  # cc > 0.14
            if var <= 1633:
                if ba <= 1.75:
                    if ed <= 0.06:
                        return 'jpegli-444'
                    return 'jpegli-420'
                return 'jpegli-444'
            else:
                if ed <= 0.05:
                    return 'jpegli-444'
                if uf > 0.1:
                    return 'jpegli-444'
                return 'jpegli-420'

    # MEDIUM QUALITY: BA 2.75 - 4.75
    if ba <= 4.75:
        if cc <= 0.17:
            if var <= 2438:
                if var > 1773:
                    return 'mozjpeg-420'
                return 'jpegli-420'
            return 'jpegli-420'
        else:  # cc > 0.17
            if ed <= 0.04:
                if cc <= 0.46:
                    return 'jpegli-444'
                return 'jpegli-420'
            if ed > 0.07:
                return 'mozjpeg-420'
            return 'jpegli-420'

    # LOW QUALITY: BA > 4.75
    if ed <= 0.04:
        if ed <= 0.02:
            return 'jpegli-444'
        if ba <= 5.75:
            if cc > 0.35:
                return 'jpegli-444'
            if uf > 0.08:
                return 'mozjpeg-420'
            return 'jpegli-420'
        if ba > 6.5:
            if cc <= 0.18:
                return 'jpegli-444'
            return 'mozjpeg-444'
        return 'jpegli-420'

    if ed <= 0.12:
        if uf <= 0.02:
            if ba > 8.5:
                return 'mozjpeg-444'
            return 'mozjpeg-420'
        if ed > 0.07:
            return 'mozjpeg-420'
        return 'jpegli-420'

    # ed > 0.12
    if cc <= 0.19:
        return 'mozjpeg-420'
    if ba > 8.5:
        return 'mozjpeg-444'
    return 'jpegli-420'


def overfit_heuristic_v3(row):
    """Simplified but targeted heuristic."""
    ba = row['target_butteraugli']
    ed = row['edge_density']
    cc = row['chroma_complexity']
    uf = row['uniform_block_fraction']
    var = row['variance']

    # RULE 1: High quality always needs jpegli-444
    if ba <= 2.0:
        return 'jpegli-444'

    # RULE 2: Medium-high quality (2.0-3.5) - jpegli-444 for chroma-rich or uniform
    if ba <= 3.5:
        if cc > 0.14:
            return 'jpegli-444'
        if uf > 0.1 and ed <= 0.05:
            return 'jpegli-444'
        return 'jpegli-420'

    # RULE 3: Medium quality (3.5-5.0) - jpegli-420 dominates, some mozjpeg
    if ba <= 5.0:
        if ed <= 0.04 and cc > 0.35:
            return 'jpegli-444'
        if ed > 0.07 and cc > 0.17:
            return 'mozjpeg-420'
        return 'jpegli-420'

    # RULE 4: Low quality (5.0-7.0) - mozjpeg starts winning
    if ba <= 7.0:
        if ed <= 0.02:
            return 'jpegli-444'
        if ed <= 0.04:
            if cc > 0.18:
                return 'mozjpeg-444'
            return 'jpegli-420'
        if ed > 0.12 and cc < 0.19:
            return 'mozjpeg-420'
        if uf <= 0.02:
            return 'mozjpeg-420'
        return 'jpegli-420'

    # RULE 5: Very low quality (BA > 7.0) - mozjpeg competitive
    if ed <= 0.04:
        if cc > 0.18:
            return 'mozjpeg-444'
        return 'jpegli-444'
    if ed > 0.12:
        if cc < 0.19:
            return 'mozjpeg-420'
        if ba > 8.5:
            return 'mozjpeg-444'
    if uf <= 0.02:
        return 'mozjpeg-420'
    if ed > 0.07:
        return 'mozjpeg-420'

    return 'jpegli-420'


def evaluate_heuristic(df, heuristic_fn, name):
    """Evaluate a heuristic function."""
    df = df.copy()
    df['predicted'] = df.apply(heuristic_fn, axis=1)

    # Accuracy
    accuracy = (df['predicted'] == df['optimal_config']).mean()

    # Per-config accuracy
    per_config = {}
    for config in CONFIGS:
        config_samples = df[df['optimal_config'] == config]
        if len(config_samples) > 0:
            correct = (config_samples['predicted'] == config).sum()
            per_config[config] = (correct, len(config_samples), correct/len(config_samples)*100)

    # Regret
    regrets = []
    for _, row in df.iterrows():
        pred = row['predicted']
        regret_col = f'regret_{pred}'
        if pd.notna(row.get(regret_col)):
            regrets.append(row[regret_col])
    regrets = np.array(regrets)

    print(f"\n{'='*60}")
    print(f"{name}")
    print('='*60)
    print(f"Overall accuracy: {accuracy*100:.1f}%")
    print(f"\nPer-config accuracy:")
    for config, (correct, total, pct) in per_config.items():
        print(f"  {config}: {correct}/{total} ({pct:.1f}%)")
    print(f"\nRegret statistics:")
    print(f"  Mean: {regrets.mean():.2f}%")
    print(f"  Median: {np.median(regrets):.2f}%")
    print(f"  95th pct: {np.percentile(regrets, 95):.2f}%")
    print(f"  Max: {regrets.max():.2f}%")

    return accuracy, regrets.mean()


def generate_rust_code():
    """Generate final Rust heuristic."""
    print("\n" + "="*80)
    print("FINAL RUST HEURISTIC")
    print("="*80)

    print("""
/// Select optimal codec for target Butteraugli distance.
///
/// This heuristic is overfit to 86 images (CID22 + CLIC) across 15 quality targets.
/// Training accuracy: ~85%, Mean regret: ~2%
///
/// Arguments:
/// - variance: Image luma variance
/// - edge_density: Fraction of edge pixels (Sobel gradient > 30)
/// - chroma_complexity: Chroma channel complexity
/// - uniform_block_fraction: Fraction of 8x8 blocks with variance < 100
/// - target_butteraugli: Target perceptual quality (lower = better, 1.0 = visually lossless)
pub fn select_codec_butteraugli(
    variance: f32,
    edge_density: f32,
    chroma_complexity: f32,
    uniform_block_fraction: f32,
    target_butteraugli: f32,
) -> Config {
    // HIGH QUALITY (BA <= 2.0): jpegli-444 required for achievability
    if target_butteraugli <= 2.0 {
        return Config::Jpegli { subsampling: Subsampling::S444 };
    }

    // MEDIUM-HIGH QUALITY (BA 2.0-3.5): jpegli-444 for chroma-rich images
    if target_butteraugli <= 3.5 {
        if chroma_complexity > 0.14 {
            return Config::Jpegli { subsampling: Subsampling::S444 };
        }
        if uniform_block_fraction > 0.1 && edge_density <= 0.05 {
            return Config::Jpegli { subsampling: Subsampling::S444 };
        }
        return Config::Jpegli { subsampling: Subsampling::S420 };
    }

    // MEDIUM QUALITY (BA 3.5-5.0): jpegli-420 dominates
    if target_butteraugli <= 5.0 {
        if edge_density <= 0.04 && chroma_complexity > 0.35 {
            return Config::Jpegli { subsampling: Subsampling::S444 };
        }
        if edge_density > 0.07 && chroma_complexity > 0.17 {
            return Config::MozJpeg { subsampling: Subsampling::S420 };
        }
        return Config::Jpegli { subsampling: Subsampling::S420 };
    }

    // LOW QUALITY (BA 5.0-7.0): mozjpeg becomes competitive
    if target_butteraugli <= 7.0 {
        if edge_density <= 0.02 {
            return Config::Jpegli { subsampling: Subsampling::S444 };
        }
        if edge_density <= 0.04 && chroma_complexity > 0.18 {
            return Config::MozJpeg { subsampling: Subsampling::S444 };
        }
        if edge_density > 0.12 && chroma_complexity < 0.19 {
            return Config::MozJpeg { subsampling: Subsampling::S420 };
        }
        if uniform_block_fraction <= 0.02 {
            return Config::MozJpeg { subsampling: Subsampling::S420 };
        }
        return Config::Jpegli { subsampling: Subsampling::S420 };
    }

    // VERY LOW QUALITY (BA > 7.0): mozjpeg often wins
    if edge_density <= 0.04 {
        if chroma_complexity > 0.18 {
            return Config::MozJpeg { subsampling: Subsampling::S444 };
        }
        return Config::Jpegli { subsampling: Subsampling::S444 };
    }
    if edge_density > 0.12 && chroma_complexity < 0.19 {
        return Config::MozJpeg { subsampling: Subsampling::S420 };
    }
    if target_butteraugli > 8.5 && edge_density > 0.12 {
        return Config::MozJpeg { subsampling: Subsampling::S444 };
    }
    if uniform_block_fraction <= 0.02 || edge_density > 0.07 {
        return Config::MozJpeg { subsampling: Subsampling::S420 };
    }

    Config::Jpegli { subsampling: Subsampling::S420 }
}
""")


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows")

    regret_df = build_dataset(df, BUTTERAUGLI_TARGETS)
    print(f"Dataset: {len(regret_df)} samples")

    # Baseline
    def always_jpegli_420(row):
        return 'jpegli-420'

    evaluate_heuristic(regret_df, always_jpegli_420, "BASELINE: Always jpegli-420")

    # Test heuristics
    evaluate_heuristic(regret_df, overfit_heuristic_v1, "HEURISTIC V1 (Simple)")
    evaluate_heuristic(regret_df, overfit_heuristic_v2, "HEURISTIC V2 (Decision Tree Port)")
    evaluate_heuristic(regret_df, overfit_heuristic_v3, "HEURISTIC V3 (Balanced)")

    # Generate Rust code
    generate_rust_code()


if __name__ == '__main__':
    main()
