"""
Titanic survival prediction model training.

Trains multiple classifiers and selects the best by ROC-AUC.
Uses stratified 5-fold CV to prevent overfitting on small dataset.
"""
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.processing import process_data

warnings.filterwarnings('ignore')


def get_models():
    """Return dict of model name -> model instance."""
    return {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, class_weight='balanced',
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=5,
            random_state=42, class_weight='balanced', n_jobs=-1,
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=5, random_state=42,
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_child_weight=5, random_state=42,
            eval_metric='logloss', verbosity=0,
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_child_samples=5, random_state=42, verbose=-1,
        ),
    }


def train_and_evaluate(data_path='Titanic-Dataset.csv', cv_folds=5):
    """Train all models with CV and evaluate on held-out test set."""
    X_train, X_test, y_train, y_test, pipeline = process_data(data_path)

    models = get_models()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    results = []
    trained_models = {}

    for name, model in models.items():
        start = time.time()

        # Cross-validation
        cv_scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring=['accuracy', 'f1', 'roc_auc', 'precision', 'recall'],
            return_train_score=False,
        )

        # Train on full training set
        model.fit(X_train, y_train)
        fit_time = time.time() - start

        # Evaluate on test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        result = {
            'model': name,
            'cv_accuracy': cv_scores['test_accuracy'].mean(),
            'cv_f1': cv_scores['test_f1'].mean(),
            'cv_roc_auc': cv_scores['test_roc_auc'].mean(),
            'cv_precision': cv_scores['test_precision'].mean(),
            'cv_recall': cv_scores['test_recall'].mean(),
            'cv_accuracy_std': cv_scores['test_accuracy'].std(),
            'cv_roc_auc_std': cv_scores['test_roc_auc'].std(),
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred),
            'test_recall': recall_score(y_test, y_pred),
            'test_f1': f1_score(y_test, y_pred),
            'test_roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else None,
            'fit_time': round(fit_time, 2),
        }
        results.append(result)
        trained_models[name] = model
        print(f"  {name}: CV AUC={result['cv_roc_auc']:.4f}±{result['cv_roc_auc_std']:.4f}, "
              f"Test AUC={result['test_roc_auc']:.4f}, Test F1={result['test_f1']:.4f}")

    # Select best model by CV ROC-AUC
    results_df = pd.DataFrame(results).sort_values('cv_roc_auc', ascending=False)
    best_name = results_df.iloc[0]['model']
    best_model = trained_models[best_name]

    print(f"\nBest model: {best_name} (CV AUC: {results_df.iloc[0]['cv_roc_auc']:.4f})")

    return results_df, best_model, trained_models, pipeline, X_test, y_test


def save_model(model, pipeline, results_df, output_dir='models'):
    """Save the best model, pipeline, and results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    joblib.dump(model, output_path / 'best_model.joblib')
    joblib.dump(pipeline, output_path / 'pipeline.joblib')
    results_df.to_csv(output_path / 'model_comparison.csv', index=False)

    # Save feature names
    feature_names = [
        'Sex_binary', 'Title_encoded', 'Pclass', 'HasCabin', 'Deck_encoded',
        'Fare_log1p', 'FareBin', 'FamilySize', 'IsAlone', 'FamilyType',
        'Age_imputed', 'AgeBin', 'IsChild', 'IsMother',
        'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Sex_Pclass',
    ]
    with open(output_path / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f)

    print(f"Model artifacts saved to {output_path}/")


def generate_evaluation_artifacts(best_model, pipeline, X_test, y_test, output_dir='reports'):
    """Generate evaluation plots and metrics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=['Perished', 'Survived'],
        cmap='Blues', ax=ax,
    )
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    fig.savefig(output_path / 'confusion_matrix.png', dpi=150)
    plt.close()

    # ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name='Best Model')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax.set_title('ROC Curve')
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path / 'roc_curve.png', dpi=150)
    plt.close()

    # Feature importance
    feature_names = [
        'Sex_binary', 'Title_encoded', 'Pclass', 'HasCabin', 'Deck_encoded',
        'Fare_log1p', 'FareBin', 'FamilySize', 'IsAlone', 'FamilyType',
        'Age_imputed', 'AgeBin', 'IsChild', 'IsMother',
        'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Sex_Pclass',
    ]

    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = np.abs(best_model.coef_[0])
    else:
        importances = np.zeros(len(feature_names))

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
    }).sort_values('importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#117A65' if i >= len(importance_df) - 5 else '#566573'
              for i in range(len(importance_df))]
    ax.barh(importance_df['feature'], importance_df['importance'], color=colors)
    ax.set_title('Feature Importance (Top 5 highlighted)')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    fig.savefig(output_path / 'feature_importance.png', dpi=150)
    plt.close()

    # Save classification report
    report = classification_report(y_test, y_pred, target_names=['Perished', 'Survived'])
    with open(output_path / 'classification_report.txt', 'w') as f:
        f.write(report)

    # Save metrics JSON
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_prob)),
        'confusion_matrix': cm.tolist(),
        'feature_importance': dict(zip(feature_names, importances.tolist())),
        'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
        'classification_report': report,
    }
    with open(output_path / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Evaluation artifacts saved to {output_path}/")
    return metrics


if __name__ == '__main__':
    print("Training models...")
    results_df, best_model, trained_models, pipeline, X_test, y_test = train_and_evaluate()

    print("\nSaving model artifacts...")
    save_model(best_model, pipeline, results_df)

    print("\nGenerating evaluation artifacts...")
    metrics = generate_evaluation_artifacts(best_model, pipeline, X_test, y_test)

    print(f"\n{'='*50}")
    print(f"Best Model Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"{'='*50}")
