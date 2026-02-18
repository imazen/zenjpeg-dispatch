#!/usr/bin/env python3
"""
Regret-based analysis for codec selection heuristics.

Instead of optimizing win rate, we minimize REGRET - the % file size increase
when picking a suboptimal codec vs the true optimal for each image.

This approach:
1. Measures actual cost of wrong decisions (not just counting wins)
2. Compares heuristics against "always pick X" baselines
3. Only recommends complex rules if they meaningfully reduce regret
"""

import pandas as pd
import numpy as np
from scipy import interpolate
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
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
    """Load benchmark data."""
    df = pd.read_csv(csv_path, names=COLUMNS)
    print(f"Loaded {len(df)} rows, {df['source_hash'].nunique()} images")
    return df


def interpolate_bpp_at_ssim2(group, target_ssim2):
    """Interpolate to find BPP needed to achieve target SSIM2."""
    sorted_group = group.sort_values('ssimulacra2')
    ssim2_values = sorted_group['ssimulacra2'].values
    bpp_values = sorted_group['bpp'].values

    if target_ssim2 < ssim2_values.min() or target_ssim2 > ssim2_values.max():
        return None

    try:
        f = interpolate.interp1d(ssim2_values, bpp_values, kind='linear', fill_value='extrapolate')
        return float(f(target_ssim2))
    except:
        return None


def get_bpp_for_all_configs(img_df, target_ssim2):
    """Get interpolated BPP for each config at target SSIM2."""
    config_bpp = {}
    for config_key, config_group in img_df.groupby('config_key'):
        bpp = interpolate_bpp_at_ssim2(config_group, target_ssim2)
        if bpp is not None and bpp > 0:
            config_bpp[config_key] = bpp
    return config_bpp


def compute_regret(predicted_config, config_bpp):
    """
    Compute regret: % file size increase vs optimal choice.
    Returns 0 if predicted is optimal, positive value otherwise.
    """
    if not config_bpp or predicted_config not in config_bpp:
        return None

    optimal_bpp = min(config_bpp.values())
    predicted_bpp = config_bpp[predicted_config]

    if optimal_bpp <= 0:
        return None

    regret = (predicted_bpp - optimal_bpp) / optimal_bpp * 100
    return regret


def build_regret_dataset(df, target_ssim2_values):
    """Build dataset with regret for each image at each target quality."""
    data = []

    for source_hash, img_group in df.groupby('source_hash'):
        chars = img_group[FEATURES].iloc[0].to_dict()
        chars['source_hash'] = source_hash

        min_ssim2 = img_group['ssimulacra2'].min()
        max_ssim2 = img_group['ssimulacra2'].max()

        for target in target_ssim2_values:
            if target < min_ssim2 + 5 or target > max_ssim2 - 5:
                continue

            config_bpp = get_bpp_for_all_configs(img_group, target)

            if len(config_bpp) < 2:
                continue

            optimal_config = min(config_bpp, key=config_bpp.get)
            optimal_bpp = config_bpp[optimal_config]

            sample = chars.copy()
            sample['target_ssim2'] = target
            sample['optimal_config'] = optimal_config
            sample['optimal_bpp'] = optimal_bpp

            # Store BPP and regret for each config
            for config in CONFIGS:
                if config in config_bpp:
                    sample[f'bpp_{config}'] = config_bpp[config]
                    sample[f'regret_{config}'] = (config_bpp[config] - optimal_bpp) / optimal_bpp * 100
                else:
                    sample[f'bpp_{config}'] = np.nan
                    sample[f'regret_{config}'] = np.nan

            data.append(sample)

    return pd.DataFrame(data)


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

        results.append({
            'strategy': f'Always {config}',
            'mean_regret': mean_regret,
            'median_regret': median_regret,
            'p95_regret': p95_regret,
            'max_regret': max_regret,
            'optimal_pct': zero_regret_pct,
            'n_samples': len(valid_regrets)
        })

        print(f"\nAlways {config}:")
        print(f"  Mean regret: {mean_regret:.2f}% larger files")
        print(f"  Median regret: {median_regret:.2f}%")
        print(f"  95th percentile: {p95_regret:.2f}%")
        print(f"  Max regret: {max_regret:.2f}%")
        print(f"  Optimal choice: {zero_regret_pct:.1f}% of the time")

    return pd.DataFrame(results)


def evaluate_quality_based_strategy(regret_df):
    """Evaluate strategy that switches codec based on target quality."""
    print("\n" + "="*80)
    print("QUALITY-BASED STRATEGY")
    print("="*80)

    # Simple rule: jpegli-420 for SSIM2 < 85, jpegli-444 for SSIM2 >= 85
    def simple_rule(row):
        if row['target_ssim2'] < 85:
            return 'jpegli-420'
        else:
            return 'jpegli-444'

    regret_df['predicted_simple'] = regret_df.apply(simple_rule, axis=1)

    regrets = []
    for _, row in regret_df.iterrows():
        pred = row['predicted_simple']
        regret_col = f'regret_{pred}'
        if pd.notna(row[regret_col]):
            regrets.append(row[regret_col])

    regrets = np.array(regrets)

    print(f"\nSimple rule (jpegli-420 if SSIM2<85, else jpegli-444):")
    print(f"  Mean regret: {regrets.mean():.2f}%")
    print(f"  Median regret: {np.median(regrets):.2f}%")
    print(f"  95th percentile: {np.percentile(regrets, 95):.2f}%")
    print(f"  Max regret: {regrets.max():.2f}%")
    print(f"  Optimal choice: {(regrets == 0).mean() * 100:.1f}%")

    return regrets.mean()


def evaluate_decision_tree_strategy(regret_df):
    """Train decision tree and evaluate its regret."""
    print("\n" + "="*80)
    print("DECISION TREE STRATEGY")
    print("="*80)

    features_with_target = FEATURES + ['target_ssim2']
    X = regret_df[features_with_target].values
    y = regret_df['optimal_config'].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train decision tree
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10)
    tree.fit(X, y_encoded)

    # Predict and compute regret
    predictions = le.inverse_transform(tree.predict(X))

    regrets = []
    for i, (_, row) in enumerate(regret_df.iterrows()):
        pred = predictions[i]
        regret_col = f'regret_{pred}'
        if pd.notna(row[regret_col]):
            regrets.append(row[regret_col])

    regrets = np.array(regrets)

    print(f"\nDecision Tree (depth=4):")
    print(f"  Mean regret: {regrets.mean():.2f}%")
    print(f"  Median regret: {np.median(regrets):.2f}%")
    print(f"  95th percentile: {np.percentile(regrets, 95):.2f}%")
    print(f"  Max regret: {regrets.max():.2f}%")
    print(f"  Optimal choice: {(regrets == 0).mean() * 100:.1f}%")

    # Cross-validation accuracy
    cv_scores = cross_val_score(tree, X, y_encoded, cv=5)
    print(f"  CV accuracy: {cv_scores.mean():.3f}")

    # Feature importance
    print(f"\n  Feature Importances:")
    for fname, imp in sorted(zip(features_with_target, tree.feature_importances_), key=lambda x: -x[1]):
        print(f"    {fname}: {imp:.3f}")

    return regrets.mean(), tree, le


def analyze_margin_of_victory(regret_df):
    """Analyze how often the winning codec wins by a significant margin."""
    print("\n" + "="*80)
    print("MARGIN OF VICTORY ANALYSIS")
    print("="*80)

    # For each sample, compute margin between best and second-best
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

    return margins


def compute_value_of_heuristic(baseline_regret, heuristic_regret):
    """Compute how much value a heuristic adds vs baseline."""
    reduction = baseline_regret - heuristic_regret
    pct_reduction = reduction / baseline_regret * 100 if baseline_regret > 0 else 0
    return reduction, pct_reduction


def main():
    df = load_data()

    # Build regret dataset
    target_values = [40, 50, 60, 70, 75, 80, 85, 90]
    print(f"\nBuilding regret dataset for targets: {target_values}")
    regret_df = build_regret_dataset(df, target_values)
    print(f"Regret dataset: {len(regret_df)} samples")

    # Evaluate baselines
    baseline_results = evaluate_baseline_strategies(regret_df)

    # Find best baseline
    best_baseline = baseline_results.loc[baseline_results['mean_regret'].idxmin()]
    print(f"\n*** Best baseline: {best_baseline['strategy']} with {best_baseline['mean_regret']:.2f}% mean regret ***")

    # Evaluate quality-based strategy
    quality_regret = evaluate_quality_based_strategy(regret_df)

    # Evaluate decision tree
    tree_regret, tree, le = evaluate_decision_tree_strategy(regret_df)

    # Analyze margins
    margins = analyze_margin_of_victory(regret_df)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Is a complex heuristic worth it?")
    print("="*80)

    best_baseline_regret = best_baseline['mean_regret']

    print(f"\nBest simple baseline: {best_baseline['strategy']}")
    print(f"  Mean regret: {best_baseline_regret:.2f}%")

    print(f"\nQuality-based rule (SSIM2 threshold):")
    print(f"  Mean regret: {quality_regret:.2f}%")
    reduction, pct = compute_value_of_heuristic(best_baseline_regret, quality_regret)
    print(f"  Improvement over baseline: {reduction:.2f}% ({pct:.1f}% reduction)")

    print(f"\nDecision tree heuristic:")
    print(f"  Mean regret: {tree_regret:.2f}%")
    reduction, pct = compute_value_of_heuristic(best_baseline_regret, tree_regret)
    print(f"  Improvement over baseline: {reduction:.2f}% ({pct:.1f}% reduction)")

    print(f"\nMargin analysis:")
    print(f"  In {(margins < 2).mean()*100:.1f}% of cases, margin is <2%")
    print(f"  This means even 'wrong' choices often cost little")

    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    if tree_regret < best_baseline_regret - 1.0:
        print("\nComplex heuristic IS worth it:")
        print(f"  Saves {best_baseline_regret - tree_regret:.2f}% average file size")
    elif quality_regret < best_baseline_regret - 0.5:
        print("\nSimple quality-based rule is sufficient:")
        print(f"  Saves {best_baseline_regret - quality_regret:.2f}% average file size")
        print("  Much simpler than decision tree")
    else:
        print(f"\nJust use '{best_baseline['strategy'].replace('Always ', '')}':")
        print(f"  Complex heuristics don't improve enough to justify complexity")
        print(f"  Mean regret of {best_baseline_regret:.2f}% is acceptable")


if __name__ == '__main__':
    main()
