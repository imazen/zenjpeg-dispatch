#!/usr/bin/env python3
"""
Simplified approach: Only choose between jpegli-420 and jpegli-444.

Key insight from previous analysis:
- Always jpegli-420 has 5.13% mean regret
- Complex heuristics that try to pick mozjpeg have HIGHER regret
- Maybe mozjpeg is too risky - when it's wrong, it's catastrophically wrong

New approach: Only pick between jpegli subsampling modes.
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
JPEGLI_CONFIGS = ['jpegli-420', 'jpegli-444']
BUTTERAUGLI_TARGETS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0]


def load_data(csv_path='results.csv'):
    return pd.read_csv(csv_path, names=COLUMNS)


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


def get_config_bpp(img_df, target_ba, configs=CONFIGS):
    config_bpp = {}
    for config_key, config_group in img_df.groupby('config_key'):
        if config_key not in configs:
            continue
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
            # Get ALL config BPPs
            all_config_bpp = get_config_bpp(img_group, target, CONFIGS)
            # Get jpegli-only BPPs
            jpegli_bpp = get_config_bpp(img_group, target, JPEGLI_CONFIGS)

            if len(all_config_bpp) < 1 or len(jpegli_bpp) < 1:
                continue

            # Overall optimal
            optimal_config = min(all_config_bpp, key=all_config_bpp.get)
            optimal_bpp = all_config_bpp[optimal_config]

            # Jpegli-only optimal
            jpegli_optimal = min(jpegli_bpp, key=jpegli_bpp.get)
            jpegli_optimal_bpp = jpegli_bpp[jpegli_optimal]

            sample = chars.copy()
            sample['target_butteraugli'] = target
            sample['optimal_config'] = optimal_config
            sample['optimal_bpp'] = optimal_bpp
            sample['jpegli_optimal'] = jpegli_optimal
            sample['jpegli_optimal_bpp'] = jpegli_optimal_bpp

            # Regret vs TRUE optimal (including mozjpeg)
            for config in CONFIGS:
                if config in all_config_bpp:
                    sample[f'bpp_{config}'] = all_config_bpp[config]
                    sample[f'regret_{config}'] = (all_config_bpp[config] - optimal_bpp) / optimal_bpp * 100
                else:
                    sample[f'bpp_{config}'] = np.nan
                    sample[f'regret_{config}'] = np.nan

            data.append(sample)

    return pd.DataFrame(data)


def analyze_jpegli_vs_mozjpeg(df):
    """Analyze when mozjpeg beats jpegli."""
    print("\n" + "="*80)
    print("JPEGLI VS MOZJPEG ANALYSIS")
    print("="*80)

    mozjpeg_wins = df[df['optimal_config'].isin(['mozjpeg-420', 'mozjpeg-444'])]
    jpegli_wins = df[df['optimal_config'].isin(['jpegli-420', 'jpegli-444'])]

    print(f"\nTotal samples: {len(df)}")
    print(f"Mozjpeg optimal: {len(mozjpeg_wins)} ({len(mozjpeg_wins)/len(df)*100:.1f}%)")
    print(f"Jpegli optimal: {len(jpegli_wins)} ({len(jpegli_wins)/len(df)*100:.1f}%)")

    # When mozjpeg wins, how much worse is the best jpegli option?
    print("\nWhen mozjpeg is optimal, how much worse is best jpegli?")
    jpegli_regrets = []
    for _, row in mozjpeg_wins.iterrows():
        # Find best jpegli BPP
        j420_bpp = row.get('bpp_jpegli-420', np.nan)
        j444_bpp = row.get('bpp_jpegli-444', np.nan)

        best_jpegli_bpp = np.nanmin([j420_bpp, j444_bpp])
        if not np.isnan(best_jpegli_bpp):
            regret = (best_jpegli_bpp - row['optimal_bpp']) / row['optimal_bpp'] * 100
            jpegli_regrets.append(regret)

    jpegli_regrets = np.array(jpegli_regrets)
    print(f"  Mean regret of best jpegli: {jpegli_regrets.mean():.2f}%")
    print(f"  Median regret: {np.median(jpegli_regrets):.2f}%")
    print(f"  95th pct: {np.percentile(jpegli_regrets, 95):.2f}%")
    print(f"  Max: {jpegli_regrets.max():.2f}%")

    # When mozjpeg wins, analyze by BA range
    print("\nMozjpeg wins by Butteraugli range:")
    for ba_min, ba_max, name in [(0, 3, "high quality"), (3, 6, "medium"), (6, 10, "low")]:
        subset = mozjpeg_wins[(mozjpeg_wins['target_butteraugli'] >= ba_min) &
                              (mozjpeg_wins['target_butteraugli'] < ba_max)]
        if len(subset) > 0:
            print(f"  BA {ba_min}-{ba_max} ({name}): {len(subset)} wins")


def analyze_jpegli_only_regret(df):
    """Analyze regret when restricted to jpegli-only choices."""
    print("\n" + "="*80)
    print("JPEGLI-ONLY STRATEGY ANALYSIS")
    print("="*80)

    # Always jpegli-420
    j420_regrets = df['regret_jpegli-420'].dropna()
    print(f"\nAlways jpegli-420 (vs TRUE optimal):")
    print(f"  Mean regret: {j420_regrets.mean():.2f}%")
    print(f"  Median: {np.median(j420_regrets):.2f}%")
    print(f"  95th pct: {np.percentile(j420_regrets, 95):.2f}%")

    # Always jpegli-444
    j444_regrets = df['regret_jpegli-444'].dropna()
    print(f"\nAlways jpegli-444 (vs TRUE optimal):")
    print(f"  Mean regret: {j444_regrets.mean():.2f}%")
    print(f"  Median: {np.median(j444_regrets):.2f}%")
    print(f"  95th pct: {np.percentile(j444_regrets, 95):.2f}%")

    # Best jpegli (oracle)
    best_jpegli_regrets = []
    for _, row in df.iterrows():
        r420 = row.get('regret_jpegli-420', np.inf)
        r444 = row.get('regret_jpegli-444', np.inf)
        if np.isnan(r420):
            r420 = np.inf
        if np.isnan(r444):
            r444 = np.inf
        best_jpegli_regrets.append(min(r420, r444))

    best_jpegli_regrets = np.array([r for r in best_jpegli_regrets if r < np.inf])
    print(f"\nOracle best jpegli (vs TRUE optimal):")
    print(f"  Mean regret: {best_jpegli_regrets.mean():.2f}%")
    print(f"  Median: {np.median(best_jpegli_regrets):.2f}%")
    print(f"  95th pct: {np.percentile(best_jpegli_regrets, 95):.2f}%")


def test_jpegli_heuristics(df):
    """Test various jpegli-only heuristics."""
    print("\n" + "="*80)
    print("JPEGLI-ONLY HEURISTIC TESTS")
    print("="*80)

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
                if pred == row['jpegli_optimal']:
                    correct += 1

        regrets = np.array(regrets)
        print(f"\n{name}:")
        print(f"  Accuracy (vs best jpegli): {correct/total*100:.1f}%")
        print(f"  Mean regret (vs TRUE optimal): {regrets.mean():.2f}%")
        print(f"  Median: {np.median(regrets):.2f}%")
        print(f"  95th pct: {np.percentile(regrets, 95):.2f}%")
        return regrets.mean()

    # Test 1: Quality-based
    def quality_based(row):
        if row['target_butteraugli'] <= 2.0:
            return 'jpegli-444'
        return 'jpegli-420'
    evaluate(quality_based, "BA <= 2.0: 444, else 420")

    # Test 2: More aggressive 444
    def quality_based_3(row):
        if row['target_butteraugli'] <= 3.0:
            return 'jpegli-444'
        return 'jpegli-420'
    evaluate(quality_based_3, "BA <= 3.0: 444, else 420")

    # Test 3: Quality + chroma
    def quality_chroma(row):
        if row['target_butteraugli'] <= 2.0:
            return 'jpegli-444'
        if row['target_butteraugli'] <= 3.5 and row['chroma_complexity'] > 0.14:
            return 'jpegli-444'
        return 'jpegli-420'
    evaluate(quality_chroma, "BA<=2: 444, BA<=3.5+high_chroma: 444, else 420")

    # Test 4: Quality + chroma + uniform
    def quality_chroma_uniform(row):
        if row['target_butteraugli'] <= 2.0:
            return 'jpegli-444'
        if row['target_butteraugli'] <= 3.5:
            if row['chroma_complexity'] > 0.14:
                return 'jpegli-444'
            if row['uniform_block_fraction'] > 0.1 and row['edge_density'] <= 0.05:
                return 'jpegli-444'
        if row['target_butteraugli'] <= 5.0 and row['edge_density'] <= 0.04 and row['chroma_complexity'] > 0.35:
            return 'jpegli-444'
        return 'jpegli-420'
    evaluate(quality_chroma_uniform, "Complex jpegli-only heuristic")

    # Test 5: Based on jpegli_optimal analysis
    def optimal_patterns(row):
        ba = row['target_butteraugli']
        ed = row['edge_density']
        cc = row['chroma_complexity']
        uf = row['uniform_block_fraction']
        var = row['variance']

        # jpegli-444 wins patterns from data:
        # - BA <= 2.0 almost always
        # - BA 2-3 with high chroma or low edge
        # - Very low edge_density at any quality

        if ba <= 2.0:
            return 'jpegli-444'

        if ba <= 3.0:
            if cc > 0.14:
                return 'jpegli-444'
            if ed <= 0.05 and uf > 0.03:
                return 'jpegli-444'

        if ba <= 4.5:
            if ed <= 0.04 and cc > 0.17:
                return 'jpegli-444'

        if ed <= 0.02:
            return 'jpegli-444'

        return 'jpegli-420'

    evaluate(optimal_patterns, "Pattern-based jpegli heuristic")


def generate_final_heuristic():
    """Generate the final recommended heuristic."""
    print("\n" + "="*80)
    print("FINAL RECOMMENDED HEURISTIC (JPEGLI-ONLY)")
    print("="*80)

    print("""
/// Select optimal jpegli subsampling for target Butteraugli distance.
///
/// Key insight: Complex mozjpeg selection increases regret.
/// Simple jpegli-only selection achieves near-optimal results.
///
/// Arguments:
/// - edge_density: Fraction of edge pixels
/// - chroma_complexity: Chroma channel complexity
/// - uniform_block_fraction: Fraction of uniform 8x8 blocks
/// - target_butteraugli: Target quality (lower = better)
pub fn select_jpegli_subsampling(
    edge_density: f32,
    chroma_complexity: f32,
    uniform_block_fraction: f32,
    target_butteraugli: f32,
) -> Subsampling {
    // HIGH QUALITY (BA <= 2.0): Always use 444
    if target_butteraugli <= 2.0 {
        return Subsampling::S444;
    }

    // MEDIUM-HIGH QUALITY (BA 2.0-3.0): 444 for chroma-rich or uniform images
    if target_butteraugli <= 3.0 {
        if chroma_complexity > 0.14 {
            return Subsampling::S444;
        }
        if edge_density <= 0.05 && uniform_block_fraction > 0.03 {
            return Subsampling::S444;
        }
    }

    // MEDIUM QUALITY (BA 3.0-4.5): 444 for specific patterns
    if target_butteraugli <= 4.5 {
        if edge_density <= 0.04 && chroma_complexity > 0.17 {
            return Subsampling::S444;
        }
    }

    // Very low edge density at any quality favors 444
    if edge_density <= 0.02 {
        return Subsampling::S444;
    }

    // Default: 420 (best compression, good quality)
    Subsampling::S420
}
""")


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows")

    regret_df = build_dataset(df, BUTTERAUGLI_TARGETS)
    print(f"Dataset: {len(regret_df)} samples")

    analyze_jpegli_vs_mozjpeg(regret_df)
    analyze_jpegli_only_regret(regret_df)
    test_jpegli_heuristics(regret_df)
    generate_final_heuristic()


if __name__ == '__main__':
    main()
