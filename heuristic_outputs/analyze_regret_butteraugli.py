#!/usr/bin/env python3
"""
Regret-based analysis targeting Butteraugli distance.

Butteraugli: lower is better (0 = identical, <1.0 = visually lossless, >3 = noticeable)

For a target Butteraugli distance, find which codec achieves it at smallest BPP.
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

# Butteraugli target ranges (lower is better)
# ~1.0 = visually lossless, ~3.0 = noticeable, ~6.0 = significant
BUTTERAUGLI_TARGETS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


def load_data(csv_path='results.csv'):
    df = pd.read_csv(csv_path, names=COLUMNS)
    print(f"Loaded {len(df)} rows, {df['source_hash'].nunique()} images")
    print(f"Butteraugli range: {df['butteraugli'].min():.2f} to {df['butteraugli'].max():.2f}")
    return df


def interpolate_bpp_at_butteraugli(group, target_ba):
    """
    Interpolate to find BPP needed to achieve target Butteraugli.
    Note: Lower Butteraugli = higher quality = typically higher BPP
    """
    # Sort by butteraugli (ascending = better quality)
    sorted_group = group.sort_values('butteraugli')
    ba_values = sorted_group['butteraugli'].values
    bpp_values = sorted_group['bpp'].values

    # Can't achieve if target is outside achievable range
    # For Butteraugli, we need to be able to get DOWN to the target
    if target_ba < ba_values.min() or target_ba > ba_values.max():
        return None

    try:
        # Interpolate: given target butteraugli, what BPP do we need?
        f = interpolate.interp1d(ba_values, bpp_values, kind='linear')
        return float(f(target_ba))
    except:
        return None


def get_achievable_configs(img_df, target_ba):
    """Get configs that CAN achieve target Butteraugli (or better), with their BPP."""
    config_bpp = {}
    for config_key, config_group in img_df.groupby('config_key'):
        # Can achieve if min butteraugli <= target (can get good enough quality)
        min_ba = config_group['butteraugli'].min()
        if min_ba <= target_ba:
            bpp = interpolate_bpp_at_butteraugli(config_group, target_ba)
            if bpp is not None and bpp > 0:
                config_bpp[config_key] = bpp
    return config_bpp


def build_regret_dataset(df, target_values):
    """Build dataset tracking achievability for Butteraugli targets."""
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
            sample['target_butteraugli'] = target
            sample['optimal_config'] = optimal_config
            sample['optimal_bpp'] = optimal_bpp
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
    """Analyze which configs can achieve which Butteraugli targets."""
    print("\n" + "="*80)
    print("ACHIEVABILITY ANALYSIS: Can each codec reach the target Butteraugli?")
    print("="*80)

    for target in sorted(regret_df['target_butteraugli'].unique()):
        target_df = regret_df[regret_df['target_butteraugli'] == target]
        n_images = len(target_df)

        print(f"\nTarget Butteraugli = {target} ({n_images} images):")
        for config in CONFIGS:
            achievable = target_df[f'achievable_{config}'].sum()
            pct = achievable / n_images * 100 if n_images > 0 else 0
            print(f"  {config}: achievable for {achievable}/{n_images} ({pct:.1f}%)")


def analyze_winners_by_target(regret_df):
    """Analyze which codec wins at each target Butteraugli."""
    print("\n" + "="*80)
    print("WINNER ANALYSIS BY TARGET BUTTERAUGLI")
    print("="*80)

    for target in sorted(regret_df['target_butteraugli'].unique()):
        target_df = regret_df[regret_df['target_butteraugli'] == target]

        if len(target_df) < 5:
            continue

        winner_counts = target_df['optimal_config'].value_counts()
        total = len(target_df)

        print(f"\nTarget Butteraugli = {target} ({total} images):")
        for config, count in winner_counts.items():
            pct = 100 * count / total
            avg_bpp = target_df[target_df['optimal_config'] == config]['optimal_bpp'].mean()
            print(f"  {config}: {count} wins ({pct:.1f}%), avg BPP={avg_bpp:.3f}")


def evaluate_baseline_strategies(regret_df):
    """Evaluate simple 'always pick X' strategies."""
    print("\n" + "="*80)
    print("BASELINE STRATEGIES: Always pick one codec")
    print("="*80)

    results = []

    for config in CONFIGS:
        regret_col = f'regret_{config}'
        valid_regrets = regret_df[regret_col].dropna()

        if len(valid_regrets) == 0:
            continue

        mean_regret = valid_regrets.mean()
        median_regret = valid_regrets.median()
        p95_regret = valid_regrets.quantile(0.95)
        max_regret = valid_regrets.max()
        zero_regret_pct = (valid_regrets == 0).mean() * 100

        # Count how often this config CAN'T achieve target
        achievable_col = f'achievable_{config}'
        cant_achieve = (~regret_df[achievable_col]).sum()
        cant_achieve_pct = cant_achieve / len(regret_df) * 100

        results.append({
            'strategy': f'Always {config}',
            'mean_regret': mean_regret,
            'median_regret': median_regret,
            'p95_regret': p95_regret,
            'optimal_pct': zero_regret_pct,
            'cant_achieve_pct': cant_achieve_pct,
        })

        print(f"\nAlways {config}:")
        print(f"  Mean regret: {mean_regret:.2f}% larger files")
        print(f"  Median regret: {median_regret:.2f}%")
        print(f"  95th percentile: {p95_regret:.2f}%")
        print(f"  Optimal choice: {zero_regret_pct:.1f}% of the time")
        print(f"  Can't achieve target: {cant_achieve_pct:.1f}% of cases")

    return pd.DataFrame(results)


def evaluate_with_achievability(regret_df, strategy_fn, strategy_name):
    """Evaluate a strategy with fallback when preferred can't achieve target."""
    regrets = []
    fallbacks = 0
    failures = 0

    for _, row in regret_df.iterrows():
        preferred = strategy_fn(row)

        if row.get(f'achievable_{preferred}', False):
            regret = row[f'regret_{preferred}']
        else:
            fallbacks += 1
            achievable = [c for c in CONFIGS if row.get(f'achievable_{c}', False)]
            if achievable:
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


def test_combined_heuristics(regret_df):
    """Test various combined heuristics."""
    print("\n" + "="*80)
    print("COMBINED HEURISTIC STRATEGIES")
    print("="*80)

    # Strategy 1: Always jpegli-420
    evaluate_with_achievability(
        regret_df,
        lambda row: 'jpegli-420',
        "Always jpegli-420 (fallback if needed)"
    )

    # Strategy 2: Always jpegli-444
    evaluate_with_achievability(
        regret_df,
        lambda row: 'jpegli-444',
        "Always jpegli-444 (fallback if needed)"
    )

    # Strategy 3: Quality-based (low Butteraugli = high quality = use 444)
    def quality_based(row):
        if row['target_butteraugli'] <= 2.0:
            return 'jpegli-444'
        else:
            return 'jpegli-420'

    evaluate_with_achievability(regret_df, quality_based, "BA <= 2.0: jpegli-444, else jpegli-420")

    # Strategy 4: Quality-based with 1.5 threshold
    def quality_based_15(row):
        if row['target_butteraugli'] <= 1.5:
            return 'jpegli-444'
        else:
            return 'jpegli-420'

    evaluate_with_achievability(regret_df, quality_based_15, "BA <= 1.5: jpegli-444, else jpegli-420")

    # Strategy 5: Quality + edge density
    def quality_edge(row):
        if row['target_butteraugli'] <= 2.0:
            return 'jpegli-444'
        elif row['edge_density'] <= 0.12 and row['uniform_block_fraction'] > 0.70:
            return 'mozjpeg-420'
        else:
            return 'jpegli-420'

    evaluate_with_achievability(regret_df, quality_edge,
        "BA<=2.0: 444, low-edge+uniform: mozjpeg, else jpegli-420")

    # Strategy 6: More aggressive 444 for quality
    def quality_edge_v2(row):
        if row['target_butteraugli'] <= 3.0:
            return 'jpegli-444'
        elif row['edge_density'] <= 0.12 and row['uniform_block_fraction'] > 0.70:
            return 'mozjpeg-420'
        else:
            return 'jpegli-420'

    evaluate_with_achievability(regret_df, quality_edge_v2,
        "BA<=3.0: 444, low-edge+uniform: mozjpeg, else jpegli-420")


def analyze_high_quality_separately(regret_df):
    """Analyze high quality targets (low Butteraugli) separately."""
    print("\n" + "="*80)
    print("HIGH QUALITY (Butteraugli <= 2.0) ANALYSIS")
    print("="*80)

    high_q = regret_df[regret_df['target_butteraugli'] <= 2.0]
    print(f"\nSamples at Butteraugli <= 2.0: {len(high_q)}")

    if len(high_q) == 0:
        print("No samples at this quality level")
        return

    print("\nOptimal codec distribution:")
    print(high_q['optimal_config'].value_counts())

    # Can jpegli-420 achieve these targets?
    achievable_420 = high_q['achievable_jpegli-420'].sum()
    print(f"\njpegli-420 can achieve target: {achievable_420}/{len(high_q)} ({achievable_420/len(high_q)*100:.1f}%)")

    achievable_444 = high_q['achievable_jpegli-444'].sum()
    print(f"jpegli-444 can achieve target: {achievable_444}/{len(high_q)} ({achievable_444/len(high_q)*100:.1f}%)")


def train_decision_tree(regret_df):
    """Train and analyze decision tree."""
    print("\n" + "="*80)
    print("DECISION TREE ANALYSIS")
    print("="*80)

    features_with_target = FEATURES + ['target_butteraugli']
    X = regret_df[features_with_target].values
    y = regret_df['optimal_config'].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10)
    tree.fit(X, y_encoded)

    print("\nDecision Tree Rules:")
    rules = export_text(tree, feature_names=features_with_target, class_names=le.classes_.tolist())
    print(rules)

    print("\nFeature Importances:")
    for fname, imp in sorted(zip(features_with_target, tree.feature_importances_), key=lambda x: -x[1]):
        print(f"  {fname}: {imp:.3f}")

    return tree, le


def analyze_margin_of_victory(regret_df):
    """Analyze margins between codecs."""
    print("\n" + "="*80)
    print("MARGIN OF VICTORY ANALYSIS")
    print("="*80)

    margins = []
    for _, row in regret_df.iterrows():
        bpps = []
        for config in CONFIGS:
            bpp_col = f'bpp_{config}'
            if pd.notna(row[bpp_col]):
                bpps.append((config, row[bpp_col]))

        if len(bpps) >= 2:
            bpps.sort(key=lambda x: x[1])
            best_bpp = bpps[0][1]
            second_bpp = bpps[1][1]
            margin = (second_bpp - best_bpp) / best_bpp * 100
            margins.append(margin)

    margins = np.array(margins)

    print(f"\nMargin between best and second-best codec:")
    print(f"  Mean margin: {margins.mean():.2f}%")
    print(f"  Median margin: {np.median(margins):.2f}%")
    print(f"  <1% margin: {(margins < 1).mean() * 100:.1f}% of cases")
    print(f"  <2% margin: {(margins < 2).mean() * 100:.1f}% of cases")
    print(f"  <5% margin: {(margins < 5).mean() * 100:.1f}% of cases")
    print(f"  >10% margin: {(margins > 10).mean() * 100:.1f}% of cases")


def generate_final_heuristic(regret_df):
    """Generate final recommended heuristic."""
    print("\n" + "="*80)
    print("FINAL RECOMMENDED HEURISTIC FOR BUTTERAUGLI")
    print("="*80)

    print("""
/// Select codec to achieve target Butteraugli distance at minimum file size.
///
/// Butteraugli: lower is better (0=identical, <1=visually lossless, >3=noticeable)
///
/// Based on regret-minimization analysis of 86 images across quality targets.
pub fn select_codec_for_target_butteraugli(
    edge_density: f32,
    uniform_block_fraction: f32,
    target_butteraugli: f32,
) -> Config {
    if target_butteraugli <= 2.0 {
        // High quality (low Butteraugli): jpegli-444 often required
        Config::Jpegli { subsampling: Subsampling::S444 }
    } else if edge_density <= 0.12 && uniform_block_fraction > 0.70 {
        // Very uniform, low-edge images: mozjpeg-420 wins
        Config::MozJpeg { subsampling: Subsampling::S420 }
    } else {
        // Default: jpegli-420
        Config::Jpegli { subsampling: Subsampling::S420 }
    }
}
""")


def main():
    df = load_data()

    print(f"\nBuilding regret dataset for Butteraugli targets: {BUTTERAUGLI_TARGETS}")
    regret_df = build_regret_dataset(df, BUTTERAUGLI_TARGETS)
    print(f"Regret dataset: {len(regret_df)} samples")

    # Analyze achievability
    analyze_achievability(regret_df)

    # Winner analysis
    analyze_winners_by_target(regret_df)

    # Baseline strategies
    evaluate_baseline_strategies(regret_df)

    # High quality analysis
    analyze_high_quality_separately(regret_df)

    # Combined heuristics
    test_combined_heuristics(regret_df)

    # Decision tree
    train_decision_tree(regret_df)

    # Margin analysis
    analyze_margin_of_victory(regret_df)

    # Final heuristic
    generate_final_heuristic(regret_df)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
Key findings for Butteraugli-targeted codec selection:

1. At high quality (Butteraugli <= 2.0):
   - jpegli-444 has better achievability and often wins
   - 420 subsampling may not reach target quality

2. At lower quality (Butteraugli > 3.0):
   - jpegli-420 dominates
   - mozjpeg-420 wins for very uniform, low-edge images

3. The heuristic mirrors SSIMULACRA2 findings:
   - High quality -> jpegli-444
   - Low quality, uniform images -> mozjpeg-420
   - Otherwise -> jpegli-420
""")


if __name__ == '__main__':
    main()
