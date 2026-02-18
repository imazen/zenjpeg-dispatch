#!/usr/bin/env python3
"""
Regret-based analysis v2: Account for achievability constraints.

Key insight: We can't always pick jpegli-420 if it CAN'T achieve the target quality.
At high SSIM2 targets, some codecs may not be able to reach the goal.
"""

import pandas as pd
import numpy as np
from scipy import interpolate
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
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


def load_data(csv_path='results.csv'):
    df = pd.read_csv(csv_path, names=COLUMNS)
    print(f"Loaded {len(df)} rows, {df['source_hash'].nunique()} images")
    return df


def interpolate_bpp_at_ssim2(group, target_ssim2):
    """Interpolate to find BPP needed to achieve target SSIM2."""
    sorted_group = group.sort_values('ssimulacra2')
    ssim2_values = sorted_group['ssimulacra2'].values
    bpp_values = sorted_group['bpp'].values

    # Can't achieve if target is outside achievable range
    if target_ssim2 < ssim2_values.min() or target_ssim2 > ssim2_values.max():
        return None

    try:
        f = interpolate.interp1d(ssim2_values, bpp_values, kind='linear')
        return float(f(target_ssim2))
    except:
        return None


def get_achievable_configs(img_df, target_ssim2):
    """Get configs that CAN achieve target SSIM2, with their BPP."""
    config_bpp = {}
    for config_key, config_group in img_df.groupby('config_key'):
        max_ssim2 = config_group['ssimulacra2'].max()
        if max_ssim2 >= target_ssim2:
            bpp = interpolate_bpp_at_ssim2(config_group, target_ssim2)
            if bpp is not None and bpp > 0:
                config_bpp[config_key] = bpp
    return config_bpp


def build_regret_dataset(df, target_values):
    """Build dataset tracking achievability."""
    data = []

    for source_hash, img_group in df.groupby('source_hash'):
        chars = img_group[FEATURES].iloc[0].to_dict()
        chars['source_hash'] = source_hash

        for target in target_values:
            config_bpp = get_achievable_configs(img_group, target)

            if len(config_bpp) < 1:
                continue

            optimal_config = min(config_bpp, key=config_bpp.get)
            optimal_bpp = config_bpp[optimal_config]

            sample = chars.copy()
            sample['target_ssim2'] = target
            sample['optimal_config'] = optimal_config
            sample['optimal_bpp'] = optimal_bpp
            sample['achievable_configs'] = list(config_bpp.keys())
            sample['n_achievable'] = len(config_bpp)

            # Store BPP and regret for each config
            for config in CONFIGS:
                if config in config_bpp:
                    sample[f'bpp_{config}'] = config_bpp[config]
                    sample[f'regret_{config}'] = (config_bpp[config] - optimal_bpp) / optimal_bpp * 100
                    sample[f'achievable_{config}'] = True
                else:
                    sample[f'bpp_{config}'] = np.nan
                    sample[f'regret_{config}'] = np.nan
                    sample[f'achievable_{config}'] = False

            data.append(sample)

    return pd.DataFrame(data)


def analyze_achievability(regret_df):
    """Analyze which configs can achieve which quality targets."""
    print("\n" + "="*80)
    print("ACHIEVABILITY ANALYSIS: Can each codec reach the target?")
    print("="*80)

    for target in sorted(regret_df['target_ssim2'].unique()):
        target_df = regret_df[regret_df['target_ssim2'] == target]
        n_images = len(target_df)

        print(f"\nTarget SSIM2 = {target} ({n_images} images):")
        for config in CONFIGS:
            achievable = target_df[f'achievable_{config}'].sum()
            pct = achievable / n_images * 100
            print(f"  {config}: achievable for {achievable}/{n_images} ({pct:.1f}%)")


def evaluate_with_achievability(regret_df, strategy_fn, strategy_name):
    """Evaluate a strategy, falling back when preferred choice can't achieve target."""
    regrets = []
    fallbacks = 0
    failures = 0

    for _, row in regret_df.iterrows():
        preferred = strategy_fn(row)

        # Check if preferred can achieve target
        if row.get(f'achievable_{preferred}', False):
            regret = row[f'regret_{preferred}']
        else:
            # Fall back to best achievable
            fallbacks += 1
            achievable = [c for c in CONFIGS if row.get(f'achievable_{c}', False)]
            if achievable:
                # Pick best among achievable
                best_achievable = min(achievable, key=lambda c: row[f'bpp_{c}'])
                regret = row[f'regret_{best_achievable}']
            else:
                failures += 1
                continue

        if pd.notna(regret):
            regrets.append(regret)

    regrets = np.array(regrets)

    print(f"\n{strategy_name}:")
    print(f"  Mean regret: {regrets.mean():.2f}%")
    print(f"  Median regret: {np.median(regrets):.2f}%")
    print(f"  95th percentile: {np.percentile(regrets, 95):.2f}%")
    print(f"  Optimal choice: {(regrets == 0).mean() * 100:.1f}%")
    print(f"  Fallbacks needed: {fallbacks} ({fallbacks/len(regret_df)*100:.1f}%)")

    return regrets.mean(), fallbacks


def test_edge_density_threshold(regret_df):
    """Test simple edge_density threshold heuristics."""
    print("\n" + "="*80)
    print("EDGE DENSITY THRESHOLD ANALYSIS")
    print("="*80)

    # Test various thresholds
    thresholds = [0.05, 0.07, 0.10, 0.12, 0.15]

    for thresh in thresholds:
        def strategy(row, t=thresh):
            if row['edge_density'] > t:
                return 'jpegli-420'
            else:
                # Low edge density - check quality
                if row['target_ssim2'] >= 85:
                    return 'jpegli-444'
                else:
                    return 'mozjpeg-420'

        evaluate_with_achievability(regret_df, strategy, f"edge_density > {thresh}: jpegli-420, else quality-based")


def test_combined_heuristics(regret_df):
    """Test various combined heuristics."""
    print("\n" + "="*80)
    print("COMBINED HEURISTIC STRATEGIES")
    print("="*80)

    # Strategy 1: Always jpegli-420 (with fallback)
    evaluate_with_achievability(
        regret_df,
        lambda row: 'jpegli-420',
        "Always jpegli-420 (fallback if needed)"
    )

    # Strategy 2: Quality-based
    def quality_based(row):
        if row['target_ssim2'] >= 85:
            return 'jpegli-444'
        else:
            return 'jpegli-420'

    evaluate_with_achievability(regret_df, quality_based, "SSIM2 >= 85: jpegli-444, else jpegli-420")

    # Strategy 3: Quality + edge density
    def quality_edge(row):
        if row['target_ssim2'] >= 85:
            return 'jpegli-444'
        elif row['edge_density'] > 0.12:
            return 'jpegli-420'
        else:
            return 'mozjpeg-420'

    evaluate_with_achievability(regret_df, quality_edge, "SSIM2>=85: 444, edge>0.12: jpegli-420, else mozjpeg-420")

    # Strategy 4: Refined based on decision tree insights
    def refined(row):
        if row['target_ssim2'] >= 85:
            return 'jpegli-444'
        elif row['edge_density'] <= 0.12 and row['uniform_block_fraction'] > 0.70:
            return 'mozjpeg-420'
        else:
            return 'jpegli-420'

    evaluate_with_achievability(regret_df, refined, "SSIM2>=85: 444, low-edge+uniform: mozjpeg, else jpegli-420")


def analyze_high_quality_separately(regret_df):
    """Analyze high quality targets separately."""
    print("\n" + "="*80)
    print("HIGH QUALITY (SSIM2 >= 85) ANALYSIS")
    print("="*80)

    high_q = regret_df[regret_df['target_ssim2'] >= 85]
    print(f"\nSamples at SSIM2 >= 85: {len(high_q)}")

    # Winner distribution at high quality
    print("\nOptimal codec distribution:")
    print(high_q['optimal_config'].value_counts())

    # Can jpegli-420 achieve these targets?
    achievable_420 = high_q['achievable_jpegli-420'].sum()
    print(f"\njpegli-420 can achieve target: {achievable_420}/{len(high_q)} ({achievable_420/len(high_q)*100:.1f}%)")

    # When jpegli-420 CAN achieve, what's the regret vs jpegli-444?
    can_achieve = high_q[high_q['achievable_jpegli-420']]
    if len(can_achieve) > 0:
        regret_420 = can_achieve['regret_jpegli-420'].mean()
        regret_444 = can_achieve['regret_jpegli-444'].dropna().mean()
        print(f"\nWhen both achievable:")
        print(f"  jpegli-420 mean regret: {regret_420:.2f}%")
        print(f"  jpegli-444 mean regret: {regret_444:.2f}%")


def generate_final_heuristic(regret_df):
    """Generate final recommended heuristic."""
    print("\n" + "="*80)
    print("FINAL RECOMMENDED HEURISTIC")
    print("="*80)

    print("""
/// Select codec to achieve target SSIMULACRA2 at minimum file size.
///
/// Based on regret-minimization analysis of 86 images across quality targets.
pub fn select_codec_for_target_ssim2(
    edge_density: f32,
    uniform_block_fraction: f32,
    target_ssim2: f32,
) -> Config {
    if target_ssim2 >= 85.0 {
        // High quality: jpegli-444 often required and optimal
        Config::Jpegli { subsampling: Subsampling::S444 }
    } else if edge_density <= 0.12 && uniform_block_fraction > 0.70 {
        // Very uniform, low-edge images: mozjpeg-420 wins
        Config::MozJpeg { subsampling: Subsampling::S420 }
    } else {
        // Default: jpegli-420 (optimal 67% of cases, low regret otherwise)
        Config::Jpegli { subsampling: Subsampling::S420 }
    }
}
""")


def main():
    df = load_data()

    target_values = [40, 50, 60, 70, 75, 80, 85, 90]
    print(f"\nBuilding regret dataset for targets: {target_values}")
    regret_df = build_regret_dataset(df, target_values)
    print(f"Regret dataset: {len(regret_df)} samples")

    # Analyze achievability
    analyze_achievability(regret_df)

    # Analyze high quality separately
    analyze_high_quality_separately(regret_df)

    # Test edge density thresholds
    test_edge_density_threshold(regret_df)

    # Test combined strategies
    test_combined_heuristics(regret_df)

    # Generate final heuristic
    generate_final_heuristic(regret_df)


if __name__ == '__main__':
    main()
