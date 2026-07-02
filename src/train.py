from pathlib import Path
from datetime import datetime
import argparse
import joblib
import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)


def find_best_threshold(y_true, probs):
    best_threshold = 0.5
    best_acc = 0.0
    best_f1 = 0.0

    for threshold in np.arange(0.05, 0.96, 0.01):
        preds = (probs >= threshold).astype(int)

        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds)

        # Choose by accuracy first, then F1
        if acc > best_acc or (acc == best_acc and f1 > best_f1):
            best_acc = acc
            best_f1 = f1
            best_threshold = threshold

    return float(best_threshold), float(best_acc), float(best_f1)


def make_logistic_model():
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )

    return model


def save_cv_report(report_path: Path, df, feature_cols, cv_results):
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Recaptured Photo Detection - Cross Validation Report\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Dataset\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total images:  {len(df)}\n")
        f.write(f"Real images:   {int((df['label'] == 0).sum())}\n")
        f.write(f"Screen images: {int((df['label'] == 1).sum())}\n")
        f.write(f"Total features: {len(feature_cols)}\n\n")

        f.write("Model\n")
        f.write("-" * 70 + "\n")
        f.write("Model type: Logistic Regression\n")
        f.write("Feature scaling: StandardScaler\n")
        f.write("Class weight: balanced\n")
        f.write("Solver: liblinear\n")
        f.write("Cross validation: 5-fold StratifiedKFold\n\n")

        f.write("Cross-Validation Fold Averages at Threshold 0.50\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"Mean fold accuracy: {cv_results['cv_fold_acc_mean']:.4f} "
            f"± {cv_results['cv_fold_acc_std']:.4f}\n"
        )
        f.write(
            f"Mean fold F1:       {cv_results['cv_fold_f1_mean']:.4f} "
            f"± {cv_results['cv_fold_f1_std']:.4f}\n"
        )
        f.write(
            f"Mean fold AUC:      {cv_results['cv_fold_auc_mean']:.4f} "
            f"± {cv_results['cv_fold_auc_std']:.4f}\n\n"
        )

        f.write("Out-of-Fold Results at Best Threshold\n")
        f.write("-" * 70 + "\n")
        f.write(f"Best threshold: {cv_results['cv_best_threshold']:.2f}\n")
        f.write(f"OOF accuracy:   {cv_results['cv_oof_accuracy']:.4f}\n")
        f.write(f"OOF F1:         {cv_results['cv_oof_f1']:.4f}\n")
        f.write(f"OOF ROC-AUC:    {cv_results['cv_oof_auc']:.4f}\n\n")

        f.write("OOF Confusion Matrix at Best Threshold\n")
        f.write("-" * 70 + "\n")
        f.write("Rows = true labels, columns = predicted labels\n")
        f.write("[[true_real_as_real, true_real_as_screen],\n")
        f.write(" [true_screen_as_real, true_screen_as_screen]]\n\n")
        f.write(str(cv_results["best_confusion"]))
        f.write("\n\n")

        f.write("OOF Classification Report at Best Threshold\n")
        f.write("-" * 70 + "\n")
        f.write(cv_results["best_classification_report"])
        f.write("\n\n")

        f.write("Notes\n")
        f.write("-" * 70 + "\n")
        f.write(
            "The reported numbers are from 5-fold stratified cross-validation "
            "on the current self-collected dataset. After cross-validation, "
            "the final Logistic Regression model is retrained on the full "
            "available dataset before saving. Hidden-test performance may vary "
            "with different phones, displays, lighting, compression, and "
            "recapture styles.\n"
        )

    print(f"\nSaved CV report to: {report_path}")


def run_cross_validation(model_template, X, y, n_splits=5):
    print("\nRunning 5-fold cross-validation...")

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    oof_probs = np.zeros(len(y), dtype=np.float32)

    fold_accs = []
    fold_f1s = []
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        model = clone(model_template)
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)

        oof_probs[val_idx] = probs

        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds)
        auc = roc_auc_score(y_val, probs)

        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_aucs.append(auc)

        print(
            f"Fold {fold}: "
            f"accuracy={acc:.4f}, "
            f"f1={f1:.4f}, "
            f"auc={auc:.4f}"
        )

    default_preds = (oof_probs >= 0.5).astype(int)

    default_acc = accuracy_score(y, default_preds)
    default_f1 = f1_score(y, default_preds)
    default_auc = roc_auc_score(y, oof_probs)

    best_threshold, best_acc, best_f1 = find_best_threshold(y, oof_probs)
    best_preds = (oof_probs >= best_threshold).astype(int)

    best_confusion = confusion_matrix(y, best_preds)
    best_class_report = classification_report(
        y,
        best_preds,
        target_names=["real", "screen"],
    )

    print("\nCross-validation fold averages at threshold 0.50:")
    print(f"Mean accuracy: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    print(f"Mean F1:       {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print(f"Mean AUC:      {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    print("\nOut-of-fold results at threshold 0.50:")
    print(f"OOF accuracy: {default_acc:.4f}")
    print(f"OOF F1:       {default_f1:.4f}")
    print(f"OOF AUC:      {default_auc:.4f}")

    print("\nOOF confusion matrix at threshold 0.50:")
    print(confusion_matrix(y, default_preds))

    print("\nBest threshold from out-of-fold predictions:")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"OOF accuracy:   {best_acc:.4f}")
    print(f"OOF F1:         {best_f1:.4f}")
    print(f"OOF AUC:        {default_auc:.4f}")

    print("\nOOF confusion matrix at best threshold:")
    print(best_confusion)

    print("\nOOF classification report at best threshold:")
    print(best_class_report)

    cv_results = {
        "cv_best_threshold": best_threshold,
        "cv_oof_accuracy": best_acc,
        "cv_oof_f1": best_f1,
        "cv_oof_auc": float(default_auc),
        "cv_fold_acc_mean": float(np.mean(fold_accs)),
        "cv_fold_acc_std": float(np.std(fold_accs)),
        "cv_fold_f1_mean": float(np.mean(fold_f1s)),
        "cv_fold_f1_std": float(np.std(fold_f1s)),
        "cv_fold_auc_mean": float(np.mean(fold_aucs)),
        "cv_fold_auc_std": float(np.std(fold_aucs)),
        "best_confusion": best_confusion,
        "best_classification_report": best_class_report,
    }

    return cv_results


def train_model(features_csv: Path, model_out: Path, report_out: Path):
    df = pd.read_csv(features_csv)

    if "label" not in df.columns:
        raise ValueError("features.csv must contain a label column")

    drop_cols = ["image_path", "label"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df["label"]

    print(f"Rows: {len(df)}")
    print(f"Features: {len(feature_cols)}")
    print("\nLabel counts:")
    print(y.value_counts())

    model_template = make_logistic_model()

    cv_results = run_cross_validation(
        model_template=model_template,
        X=X,
        y=y,
        n_splits=5,
    )

    save_cv_report(
        report_path=report_out,
        df=df,
        feature_cols=feature_cols,
        cv_results=cv_results,
    )

    print("\nTraining final Logistic Regression on full dataset...")
    final_model = make_logistic_model()
    final_model.fit(X, y)

    model_out.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": final_model,
        "feature_cols": feature_cols,
        "threshold": float(cv_results["cv_best_threshold"]),
        "model_name": "logistic_regression",
        "cv_oof_accuracy": cv_results["cv_oof_accuracy"],
        "cv_oof_f1": cv_results["cv_oof_f1"],
        "cv_oof_auc": cv_results["cv_oof_auc"],
        "cv_fold_acc_mean": cv_results["cv_fold_acc_mean"],
        "cv_fold_acc_std": cv_results["cv_fold_acc_std"],
        "cv_fold_f1_mean": cv_results["cv_fold_f1_mean"],
        "cv_fold_f1_std": cv_results["cv_fold_f1_std"],
        "cv_fold_auc_mean": cv_results["cv_fold_auc_mean"],
        "cv_fold_auc_std": cv_results["cv_fold_auc_std"],
    }

    joblib.dump(bundle, model_out)

    print(f"\nSaved final model to: {model_out}")
    print(f"Saved threshold: {cv_results['cv_best_threshold']:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=str,
        default="outputs/features.csv",
    )
    parser.add_argument(
        "--model_out",
        type=str,
        default="models/recapture_logistic.joblib",
    )
    parser.add_argument(
        "--report_out",
        type=str,
        default="outputs/cv_results.txt",
    )

    args = parser.parse_args()

    train_model(
        features_csv=Path(args.features),
        model_out=Path(args.model_out),
        report_out=Path(args.report_out),
    )


if __name__ == "__main__":
    main()