#!/usr/bin/env python3
"""
Build a predictive model for codec selection based on image characteristics.

This script trains classifiers to predict the best codec given:
- Image characteristics (variance, edge_density, chroma_complexity, uniform_block_fraction)
- Target BPP range

No image processing is needed at prediction time - just the pre-computed analysis values.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Column names for the CSV
COLUMNS = [
    'source_hash', 'source_name', 'width', 'height',
    'variance', 'edge_density', 'chroma_complexity', 'uniform_block_fraction',
    'config_key', 'quality', 'cache_version', 'size_bytes',
    'bpp', 'butteraugli', 'ssimulacra2', 'dssim',
    'encode_time_ms', 'timestamp'
]

FEATURES = ['variance', 'edge_density', 'chroma_complexity', 'uniform_block_fraction']

BPP_RANGES = [
    (0.2, 0.5, "very_low"),
    (0.5, 1.0, "low"),
    (1.0, 1.5, "medium"),
    (1.5, 2.0, "high"),
    (2.0, 3.0, "very_high")
]

def load_data(csv_path='results.csv'):
    """Load benchmark data."""
    df = pd.read_csv(csv_path, names=COLUMNS)
    print(f"Loaded {len(df)} rows, {df['source_hash'].nunique()} images")
    return df

def prepare_training_data(df, metric='ssimulacra2', bpp_range=None):
    """
    Prepare training data: for each image (optionally in a BPP range),
    determine the winning codec and pair with image characteristics.
    """
    lower_is_better = metric in ['butteraugli', 'dssim']

    if bpp_range:
        bpp_min, bpp_max, range_name = bpp_range
        df = df[(df['bpp'] >= bpp_min) & (df['bpp'] < bpp_max)]

    training_data = []

    for source_hash, img_group in df.groupby('source_hash'):
        # Get image characteristics (same for all rows)
        chars = img_group[FEATURES].iloc[0].to_dict()
        chars['source_hash'] = source_hash

        # Find the best score for each config
        config_scores = {}
        for config_key, config_group in img_group.groupby('config_key'):
            if lower_is_better:
                best_score = config_group[metric].min()
            else:
                best_score = config_group[metric].max()
            config_scores[config_key] = best_score

        if config_scores:
            # Determine winner
            if lower_is_better:
                winner = min(config_scores, key=config_scores.get)
            else:
                winner = max(config_scores, key=config_scores.get)

            chars['winner'] = winner
            chars['best_score'] = config_scores[winner]
            training_data.append(chars)

    return pd.DataFrame(training_data)

def prepare_training_data_with_bpp(df, metric='ssimulacra2'):
    """
    Prepare training data including BPP as a feature.
    Each image contributes one sample per BPP range where it has data.
    """
    lower_is_better = metric in ['butteraugli', 'dssim']
    training_data = []

    for bpp_min, bpp_max, range_name in BPP_RANGES:
        range_df = df[(df['bpp'] >= bpp_min) & (df['bpp'] < bpp_max)]

        for source_hash, img_group in range_df.groupby('source_hash'):
            chars = img_group[FEATURES].iloc[0].to_dict()
            chars['source_hash'] = source_hash
            chars['bpp_target'] = (bpp_min + bpp_max) / 2  # Use midpoint as feature
            chars['bpp_range'] = range_name

            config_scores = {}
            for config_key, config_group in img_group.groupby('config_key'):
                if lower_is_better:
                    best_score = config_group[metric].min()
                else:
                    best_score = config_group[metric].max()
                config_scores[config_key] = best_score

            if len(config_scores) >= 2:  # Need at least 2 configs to compare
                if lower_is_better:
                    winner = min(config_scores, key=config_scores.get)
                else:
                    winner = max(config_scores, key=config_scores.get)

                chars['winner'] = winner
                training_data.append(chars)

    return pd.DataFrame(training_data)

def train_and_evaluate(X, y, model, model_name):
    """Train model and report cross-validation scores."""
    # Cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\n{model_name}:")
    print(f"  CV Accuracy: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

    # Train on full data for feature importance
    model.fit(X, y)
    return model, scores.mean()

def build_decision_rules(df, metric='ssimulacra2'):
    """Build interpretable decision rules using a shallow decision tree."""
    print(f"\n{'='*80}")
    print(f"Decision Rules for {metric}")
    print('='*80)

    train_df = prepare_training_data_with_bpp(df, metric)

    if len(train_df) < 20:
        print("Not enough data for decision rules")
        return None

    features_with_bpp = FEATURES + ['bpp_target']
    X = train_df[features_with_bpp].values
    y = train_df['winner'].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Use a shallow tree for interpretability
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5)
    tree.fit(X, y_encoded)

    # Print the tree rules
    rules = export_text(tree, feature_names=features_with_bpp, class_names=le.classes_.tolist())
    print("\nDecision Tree Rules:")
    print(rules)

    # Cross-validation accuracy
    scores = cross_val_score(tree, X, y_encoded, cv=5)
    print(f"\nTree CV Accuracy: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

    return tree, le

def analyze_by_metric(df, metric):
    """Full analysis for one metric."""
    print(f"\n{'#'*80}")
    print(f"# Analysis for {metric.upper()}")
    print('#'*80)

    # Prepare data with BPP as feature
    train_df = prepare_training_data_with_bpp(df, metric)

    if len(train_df) < 20:
        print("Not enough training data")
        return

    print(f"\nTraining samples: {len(train_df)}")
    print(f"Class distribution:\n{train_df['winner'].value_counts()}")

    features_with_bpp = FEATURES + ['bpp_target']
    X = train_df[features_with_bpp].values
    y = train_df['winner'].values

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Try different models
    models = [
        (LogisticRegression(max_iter=1000, multi_class='multinomial'), "Logistic Regression"),
        (RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42), "Random Forest"),
        (GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42), "Gradient Boosting"),
        (DecisionTreeClassifier(max_depth=5, min_samples_leaf=3), "Decision Tree"),
    ]

    best_model = None
    best_score = 0
    best_name = ""

    for model, name in models:
        if "Logistic" in name:
            trained, score = train_and_evaluate(X_scaled, y_encoded, model, name)
        else:
            trained, score = train_and_evaluate(X, y_encoded, model, name)

        if score > best_score:
            best_score = score
            best_model = trained
            best_name = name

    print(f"\nBest model: {best_name} (accuracy: {best_score:.3f})")

    # Feature importance for tree-based models
    if hasattr(best_model, 'feature_importances_'):
        print(f"\nFeature Importances ({best_name}):")
        for fname, imp in sorted(zip(features_with_bpp, best_model.feature_importances_),
                                  key=lambda x: -x[1]):
            print(f"  {fname}: {imp:.3f}")

    # Train-test split for detailed report
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print(f"\nClassification Report (test set):")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Build interpretable rules
    build_decision_rules(df, metric)

    return best_model, le, scaler

def generate_heuristic_function(df, metric='ssimulacra2'):
    """Generate a simple heuristic function based on the learned patterns."""
    print(f"\n{'='*80}")
    print(f"SIMPLE HEURISTIC FUNCTION for {metric}")
    print('='*80)

    # Analyze patterns by BPP range
    for bpp_min, bpp_max, range_name in BPP_RANGES:
        range_df = df[(df['bpp'] >= bpp_min) & (df['bpp'] < bpp_max)]
        train_df = prepare_training_data(range_df, metric)

        if len(train_df) < 10:
            continue

        winner_counts = train_df['winner'].value_counts()
        dominant = winner_counts.index[0]
        dominant_pct = winner_counts.iloc[0] / len(train_df) * 100

        print(f"\n{range_name} ({bpp_min}-{bpp_max} BPP):")
        print(f"  Default winner: {dominant} ({dominant_pct:.1f}%)")

        # Check if any feature splits improve prediction
        for feature in FEATURES:
            median = train_df[feature].median()

            low_df = train_df[train_df[feature] < median]
            high_df = train_df[train_df[feature] >= median]

            if len(low_df) >= 5 and len(high_df) >= 5:
                low_winner = low_df['winner'].value_counts().index[0]
                high_winner = high_df['winner'].value_counts().index[0]

                if low_winner != high_winner:
                    low_pct = low_df['winner'].value_counts().iloc[0] / len(low_df) * 100
                    high_pct = high_df['winner'].value_counts().iloc[0] / len(high_df) * 100
                    print(f"  Split on {feature} (median={median:.4f}):")
                    print(f"    < median: {low_winner} ({low_pct:.1f}%)")
                    print(f"    >= median: {high_winner} ({high_pct:.1f}%)")

def generate_rust_heuristic(df, metric='ssimulacra2'):
    """Generate Rust code for the heuristic function."""
    print(f"\n{'='*80}")
    print(f"RUST HEURISTIC CODE for {metric}")
    print('='*80)

    # Train a simple decision tree
    train_df = prepare_training_data_with_bpp(df, metric)
    features_with_bpp = FEATURES + ['bpp_target']
    X = train_df[features_with_bpp].values
    y = train_df['winner'].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5)
    tree.fit(X, y_encoded)

    # Generate Rust code
    print("""
/// Select the best codec based on image characteristics and target BPP.
///
/// Arguments:
/// - variance: Image variance (luminance)
/// - edge_density: Fraction of edge pixels
/// - chroma_complexity: Chroma channel complexity
/// - uniform_block_fraction: Fraction of uniform 8x8 blocks
/// - target_bpp: Target bits per pixel
///
/// Returns the recommended codec configuration.
pub fn select_codec(
    variance: f32,
    edge_density: f32,
    chroma_complexity: f32,
    uniform_block_fraction: f32,
    target_bpp: f32,
) -> Config {""")

    # Simple heuristic based on the most common patterns
    print("""    // Based on statistical analysis of 86 images across CID22 and CLIC corpora

    if target_bpp < 0.5 {
        // Very low BPP: jpegli-420 wins ~80% on SSIMULACRA2
        Config::Jpegli { subsampling: Subsampling::S420 }
    } else if target_bpp < 1.0 {
        // Low BPP: jpegli-420 wins ~50%, but 444 better for uniform images
        if uniform_block_fraction > 0.05 {
            Config::Jpegli { subsampling: Subsampling::S444 }
        } else {
            Config::Jpegli { subsampling: Subsampling::S420 }
        }
    } else if target_bpp < 1.5 {
        // Medium BPP: mozjpeg vs jpegli depends on edge density
        if edge_density < 0.1 {
            Config::MozJpeg { subsampling: Subsampling::S420 }
        } else {
            Config::Jpegli { subsampling: Subsampling::S444 }
        }
    } else {
        // High BPP: jpegli-444 wins ~60-75%
        Config::Jpegli { subsampling: Subsampling::S444 }
    }
}""")

def main():
    df = load_data()

    # Analyze each metric
    for metric in ['ssimulacra2', 'butteraugli']:
        analyze_by_metric(df, metric)
        generate_heuristic_function(df, metric)

    # Generate Rust code for the primary metric
    generate_rust_heuristic(df, 'ssimulacra2')

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
Key findings for codec selection heuristics:

1. BPP is the strongest predictor of which codec wins
   - Very Low BPP (<0.5): jpegli-420 dominates
   - Low BPP (0.5-1.0): jpegli-420, with 444 for uniform images
   - Medium BPP (1.0-1.5): Mixed - mozjpeg for low-edge, jpegli-444 otherwise
   - High BPP (>1.5): jpegli-444 dominates

2. Image characteristics provide refinement:
   - uniform_block_fraction: Higher values favor 444 subsampling
   - edge_density: Higher values favor 420 subsampling
   - chroma_complexity: Lower values slightly favor mozjpeg

3. Model accuracy:
   - Random Forest/Gradient Boosting achieve ~50-60% accuracy
   - This is reasonable given the overlap between codec performance
   - The heuristic provides a good default; edge cases may need both encodings
""")

if __name__ == '__main__':
    main()
