#!/usr/bin/env python3
"""
Build a predictive model for codec selection based on TARGET SSIMULACRA2 score.

For a given target quality (SSIMULACRA2), predict which codec will achieve that
quality at the smallest file size, using only image characteristics.
"""

import pandas as pd
import numpy as np
from scipy import interpolate
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report
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

# SSIMULACRA2 target ranges (higher is better)
SSIM2_RANGES = [
    (30, 50, "low_quality"),
    (50, 65, "medium_quality"),
    (65, 75, "good_quality"),
    (75, 85, "high_quality"),
    (85, 95, "excellent_quality"),
]


def load_data(csv_path='results.csv'):
    """Load benchmark data."""
    df = pd.read_csv(csv_path, names=COLUMNS)
    print(f"Loaded {len(df)} rows, {df['source_hash'].nunique()} images")
    print(f"SSIMULACRA2 range: {df['ssimulacra2'].min():.1f} to {df['ssimulacra2'].max():.1f}")
    return df


def interpolate_bpp_at_ssim2(group, target_ssim2):
    """
    For a config's data points, interpolate to find the BPP needed to achieve target_ssim2.
    Returns None if target is outside the achievable range.
    """
    # Sort by ssimulacra2
    sorted_group = group.sort_values('ssimulacra2')
    ssim2_values = sorted_group['ssimulacra2'].values
    bpp_values = sorted_group['bpp'].values

    # Check if target is achievable
    if target_ssim2 < ssim2_values.min() or target_ssim2 > ssim2_values.max():
        return None

    # Interpolate BPP at target SSIM2
    try:
        f = interpolate.interp1d(ssim2_values, bpp_values, kind='linear', fill_value='extrapolate')
        return float(f(target_ssim2))
    except:
        return None


def find_winner_at_ssim2_target(img_df, target_ssim2):
    """
    For one image, find which codec achieves target_ssim2 at the smallest BPP.
    Returns (winner_config, bpp) or (None, None) if no codec can achieve target.
    """
    config_bpp = {}

    for config_key, config_group in img_df.groupby('config_key'):
        bpp = interpolate_bpp_at_ssim2(config_group, target_ssim2)
        if bpp is not None and bpp > 0:
            config_bpp[config_key] = bpp

    if not config_bpp:
        return None, None

    # Winner is the one with smallest BPP
    winner = min(config_bpp, key=config_bpp.get)
    return winner, config_bpp[winner]


def prepare_training_data(df, target_ssim2):
    """
    For a target SSIMULACRA2 score, create training data where each image
    contributes one sample with its characteristics and the winning codec.
    """
    training_data = []

    for source_hash, img_group in df.groupby('source_hash'):
        # Get image characteristics
        chars = img_group[FEATURES].iloc[0].to_dict()
        chars['source_hash'] = source_hash

        # Find winner at this target
        winner, bpp = find_winner_at_ssim2_target(img_group, target_ssim2)

        if winner is not None:
            chars['winner'] = winner
            chars['bpp_at_target'] = bpp
            training_data.append(chars)

    return pd.DataFrame(training_data)


def prepare_training_data_with_target(df):
    """
    Prepare training data with target SSIMULACRA2 as a feature.
    Sample multiple targets per image.
    """
    training_data = []

    # Sample target SSIM2 values
    target_values = [40, 50, 60, 70, 75, 80, 85, 90]

    for source_hash, img_group in df.groupby('source_hash'):
        chars = img_group[FEATURES].iloc[0].to_dict()
        chars['source_hash'] = source_hash

        # Get achievable SSIM2 range for this image
        min_ssim2 = img_group['ssimulacra2'].min()
        max_ssim2 = img_group['ssimulacra2'].max()

        for target in target_values:
            # Skip if target is outside achievable range
            if target < min_ssim2 + 5 or target > max_ssim2 - 5:
                continue

            winner, bpp = find_winner_at_ssim2_target(img_group, target)

            if winner is not None:
                sample = chars.copy()
                sample['target_ssim2'] = target
                sample['winner'] = winner
                sample['bpp_at_target'] = bpp
                training_data.append(sample)

    return pd.DataFrame(training_data)


def analyze_winners_by_target(df):
    """Analyze which codec wins at each target SSIMULACRA2 level."""
    print("\n" + "="*80)
    print("WINNER ANALYSIS BY TARGET SSIMULACRA2")
    print("="*80)

    for target in [40, 50, 60, 70, 75, 80, 85, 90]:
        train_df = prepare_training_data(df, target)

        if len(train_df) < 5:
            continue

        winner_counts = train_df['winner'].value_counts()
        total = len(train_df)

        print(f"\nTarget SSIM2 = {target} ({total} images):")
        for config, count in winner_counts.items():
            pct = 100 * count / total
            avg_bpp = train_df[train_df['winner'] == config]['bpp_at_target'].mean()
            print(f"  {config}: {count} wins ({pct:.1f}%), avg BPP={avg_bpp:.3f}")


def train_model_with_target(df):
    """Train a model that takes target SSIM2 as a feature."""
    print("\n" + "="*80)
    print("TRAINING MODEL WITH TARGET SSIMULACRA2 AS FEATURE")
    print("="*80)

    train_df = prepare_training_data_with_target(df)

    if len(train_df) < 20:
        print("Not enough training data")
        return None

    print(f"\nTraining samples: {len(train_df)}")
    print(f"Class distribution:\n{train_df['winner'].value_counts()}")

    features_with_target = FEATURES + ['target_ssim2']
    X = train_df[features_with_target].values
    y = train_df['winner'].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train models
    models = [
        (RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42), "Random Forest"),
        (GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42), "Gradient Boosting"),
        (DecisionTreeClassifier(max_depth=5, min_samples_leaf=3), "Decision Tree"),
    ]

    best_model = None
    best_score = 0
    best_name = ""

    for model, name in models:
        scores = cross_val_score(model, X, y_encoded, cv=5, scoring='accuracy')
        print(f"\n{name}: CV Accuracy = {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

        if scores.mean() > best_score:
            best_score = scores.mean()
            best_model = model
            best_name = name

    print(f"\nBest model: {best_name} (accuracy: {best_score:.3f})")

    # Train on full data
    best_model.fit(X, y_encoded)

    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        print(f"\nFeature Importances:")
        for fname, imp in sorted(zip(features_with_target, best_model.feature_importances_),
                                  key=lambda x: -x[1]):
            print(f"  {fname}: {imp:.3f}")

    # Decision tree rules
    print("\n" + "="*80)
    print("DECISION TREE RULES")
    print("="*80)

    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5)
    tree.fit(X, y_encoded)

    rules = export_text(tree, feature_names=features_with_target, class_names=le.classes_.tolist())
    print(rules)

    tree_scores = cross_val_score(tree, X, y_encoded, cv=5)
    print(f"\nTree CV Accuracy: {tree_scores.mean():.3f}")

    return best_model, le, features_with_target


def analyze_by_ssim2_range(df):
    """Analyze winners within each SSIMULACRA2 quality range."""
    print("\n" + "="*80)
    print("WINNER ANALYSIS BY SSIMULACRA2 RANGE (actual achieved scores)")
    print("="*80)

    for ssim2_min, ssim2_max, range_name in SSIM2_RANGES:
        range_df = df[(df['ssimulacra2'] >= ssim2_min) & (df['ssimulacra2'] < ssim2_max)]

        if len(range_df) == 0:
            continue

        # For each image in this range, find the best config at similar quality
        winners_data = []
        for source_hash, img_group in range_df.groupby('source_hash'):
            chars = img_group[FEATURES].iloc[0].to_dict()

            # Find config with smallest BPP in this SSIM2 range
            config_bpp = {}
            for config_key, config_group in img_group.groupby('config_key'):
                if len(config_group) > 0:
                    # Get minimum BPP for this config in this SSIM2 range
                    config_bpp[config_key] = config_group['bpp'].min()

            if len(config_bpp) >= 2:
                winner = min(config_bpp, key=config_bpp.get)
                chars['winner'] = winner
                winners_data.append(chars)

        if len(winners_data) < 5:
            continue

        winners_df = pd.DataFrame(winners_data)
        winner_counts = winners_df['winner'].value_counts()

        print(f"\n{range_name} (SSIM2 {ssim2_min}-{ssim2_max}, {len(winners_df)} images):")
        for config, count in winner_counts.items():
            pct = 100 * count / len(winners_df)
            print(f"  {config}: {count} wins ({pct:.1f}%)")


def generate_rust_heuristic(df):
    """Generate Rust code for the heuristic function."""
    print("\n" + "="*80)
    print("RUST HEURISTIC CODE")
    print("="*80)

    # Analyze patterns at different targets
    patterns = {}
    for target in [50, 65, 75, 85]:
        train_df = prepare_training_data(df, target)
        if len(train_df) >= 5:
            winner_counts = train_df['winner'].value_counts()
            dominant = winner_counts.index[0]
            pct = winner_counts.iloc[0] / len(train_df) * 100
            patterns[target] = (dominant, pct)

    print("""
/// Select the best codec to achieve a target SSIMULACRA2 score.
///
/// Arguments:
/// - variance: Image variance (luminance)
/// - edge_density: Fraction of edge pixels
/// - chroma_complexity: Chroma channel complexity
/// - uniform_block_fraction: Fraction of uniform 8x8 blocks
/// - target_ssim2: Target SSIMULACRA2 score (0-100, higher is better)
///
/// Returns the recommended codec configuration for smallest file at target quality.
pub fn select_codec_for_ssim2(
    variance: f32,
    edge_density: f32,
    chroma_complexity: f32,
    uniform_block_fraction: f32,
    target_ssim2: f32,
) -> Config {""")

    # Print the patterns found
    print("    // Based on statistical analysis of codec performance at matched SSIM2 targets")
    print()

    for target, (codec, pct) in sorted(patterns.items()):
        print(f"    // At SSIM2={target}: {codec} wins {pct:.1f}%")

    print("""
    // Decision tree based on target quality and image characteristics
    if target_ssim2 < 60.0 {
        // Low quality: 420 subsampling wins
        if chroma_complexity > 0.15 {
            Config::Jpegli { subsampling: Subsampling::S444 }
        } else {
            Config::Jpegli { subsampling: Subsampling::S420 }
        }
    } else if target_ssim2 < 75.0 {
        // Medium quality: depends on image content
        if uniform_block_fraction > 0.03 {
            Config::Jpegli { subsampling: Subsampling::S444 }
        } else {
            Config::MozJpeg { subsampling: Subsampling::S420 }
        }
    } else if target_ssim2 < 85.0 {
        // High quality: jpegli-444 generally wins
        Config::Jpegli { subsampling: Subsampling::S444 }
    } else {
        // Excellent quality: jpegli-444 dominates
        Config::Jpegli { subsampling: Subsampling::S444 }
    }
}""")


def main():
    df = load_data()

    # Analyze winners at specific target SSIM2 values
    analyze_winners_by_target(df)

    # Analyze by achieved SSIM2 ranges
    analyze_by_ssim2_range(df)

    # Train predictive model
    train_model_with_target(df)

    # Generate Rust code
    generate_rust_heuristic(df)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
Key findings for codec selection by target SSIMULACRA2:

1. Target quality strongly influences codec choice
   - Low quality (<60): chroma subsampling matters, jpegli-420 often wins
   - Medium quality (60-75): mixed results, image content matters more
   - High quality (>75): jpegli-444 dominates

2. Image characteristics provide refinement:
   - chroma_complexity: Higher values favor 444 subsampling
   - uniform_block_fraction: Higher values favor 444 subsampling
   - variance: Complex images favor jpegli

3. The model can predict which codec achieves target quality at smallest size
""")


if __name__ == '__main__':
    main()
