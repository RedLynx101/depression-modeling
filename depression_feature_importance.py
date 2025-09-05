# Title: depression_feature_importance.py
# 
# Author: Noah Hicks
# Date: 2025-09-04
# Note: AI was used to help with the creation and editing of this code. 
# You may assume that AI wrote all the text, and I have approved it. 
# Models used: GPT 5.
#
# Summary: Compute feature-wise R^2 (univariate) against a depression target
# and the overall multivariate model R^2. Saves a CSV ranking and prints a
# concise report. Works with the Student Stress dataset next to this script.
#
# Run:
#   python depression_feature_importance.py \
#       --save-csv eda_output/depression_r2_by_feature.csv \
#       --save-dir eda_output \
#       --target auto
#
# Notes:
# - Auto-detects the dataset via stress_eda.py helpers and picks the first
#   column containing 'depress' for the target (override via --target).
# - Handles numeric and categorical predictors; categorical are one-hot encoded
#   per-feature for univariate R^2 and via ColumnTransformer for overall R^2.

from __future__ import annotations

import os
import re
import argparse
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Local helper module for loading/cleaning
import stress_eda as se


def detect_depression_target(df: pd.DataFrame, user_target: Optional[str] = None) -> str:
    """Select a depression-like target column.
    Prefers user_target if provided; otherwise picks the first column name
    matching 'depress' (case-insensitive). Raises if not found or non-numeric.
    """
    if user_target:
        tgt = user_target
        if tgt not in df.columns:
            raise ValueError(f"Target '{tgt}' not found in columns.")
    else:
        patt = re.compile(r"depress", re.IGNORECASE)
        candidates = [c for c in df.columns if patt.search(c)]
        if not candidates:
            raise ValueError("Could not auto-detect a depression target column. Use --target to specify.")
        tgt = candidates[0]

    # Coerce to numeric if possible; otherwise raise
    if not pd.api.types.is_numeric_dtype(df[tgt]):
        try:
            df[tgt] = pd.to_numeric(df[tgt], errors="coerce")
        except Exception:
            pass
    if not pd.api.types.is_numeric_dtype(df[tgt]):
        raise TypeError(f"Target '{tgt}' must be numeric after cleaning.")
    return tgt


def r2_univariate(df: pd.DataFrame, feature: str, target: str) -> Tuple[float, int]:
    """Compute R^2 for a single feature predicting the target.
    Handles numeric vs categorical via one-hot encoding. Returns (r2, n_used).
    Drops rows with NA in feature or target for a fair comparison.
    """
    sub = df[[feature, target]].dropna()
    n = len(sub)
    if n < 3:
        return (np.nan, n)

    X = sub[[feature]]
    y = sub[target].values

    if pd.api.types.is_numeric_dtype(X[feature]):
        model = LinearRegression()
        model.fit(X.values, y)
        r2 = model.score(X.values, y)
        return (float(r2), n)
    else:
        # One-hot encode this single categorical feature
        enc = OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False)
        X_enc = enc.fit_transform(X[[feature]])
        model = LinearRegression()
        model.fit(X_enc, y)
        r2 = model.score(X_enc, y)
        return (float(r2), n)


def r2_overall(df: pd.DataFrame, target: str, exclude: Optional[List[str]] = None) -> Tuple[Tuple[float, int], Pipeline]:
    """Compute overall multivariate R^2 using all predictors except `exclude`.
    Builds a sklearn Pipeline with preprocessing and LinearRegression.
    Returns (r2, n_used, pipeline).
    """
    exclude = exclude or []
    features = [c for c in df.columns if c != target and c not in exclude]
    if not features:
        raise ValueError("No features available for overall model.")

    # Build dataset and drop rows with NA in target
    data = df[features + [target]].dropna(subset=[target]).copy()
    n = len(data)

    X = data[features]
    y = data[target].values

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Include imputers for robustness on missing data
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )

    pipe = Pipeline([
        ("pre", pre),
        ("lr", LinearRegression()),
    ])

    pipe.fit(X, y)
    r2 = pipe.score(X, y)
    return (float(r2), n), pipe


def _contains_any(name: str, patterns: List[str]) -> bool:
    """Return True if name contains any of the substrings in patterns (case-insensitive)."""
    lower = name.lower()
    return any(p.lower() in lower for p in patterns)


def compute_feature_r2_table(df: pd.DataFrame, target: str, exclude_patterns: Optional[List[str]] = None) -> pd.DataFrame:
    """Compute univariate R^2 for each feature vs target and return a DataFrame.

    Features containing any of exclude_patterns are skipped.
    """
    exclude_patterns = exclude_patterns or []
    records = []
    for feat in df.columns:
        if feat == target or _contains_any(feat, exclude_patterns):
            continue
        r2, n = r2_univariate(df, feat, target)
        ftype = "numeric" if pd.api.types.is_numeric_dtype(df[feat]) else "categorical"
        records.append({"feature": feat, "type": ftype, "n": n, "r2": r2})
    out = pd.DataFrame.from_records(records)
    out = out.sort_values(by=["r2"], ascending=False, na_position="last").reset_index(drop=True)
    return out


def _prepare_data(df: pd.DataFrame, target: str, exclude_patterns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Prepare X, y, and column lists, excluding features containing patterns like 'stress'."""
    exclude_patterns = exclude_patterns or []
    features = [c for c in df.columns if c != target and not _contains_any(c, exclude_patterns)]
    data = df[features + [target]].copy()
    # Keep rows where target is present; feature imputation will handle NaNs later
    data = data.dropna(subset=[target])
    X = data[features]
    y = data[target]
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return X, y, num_cols, cat_cols


def _build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Create a preprocessing transformer with imputers, scaling, and one-hot encoding."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return pre


def _build_model_pipelines(pre: ColumnTransformer, random_state: int = 42) -> Dict[str, Pipeline]:
    """Define candidate regression models wrapped in a preprocessing pipeline."""
    models: Dict[str, Pipeline] = {
        "LinearRegression": Pipeline([("pre", pre), ("est", LinearRegression())]),
        "RidgeCV": Pipeline([("pre", pre), ("est", RidgeCV(alphas=np.logspace(-3, 3, 13)))]),
        "LassoCV": Pipeline([("pre", pre), ("est", LassoCV(alphas=np.logspace(-3, 1, 9), max_iter=10000, random_state=random_state))]),
        "RandomForest": Pipeline([("pre", pre), ("est", RandomForestRegressor(n_estimators=300, random_state=random_state))]),
        "GradientBoosting": Pipeline([("pre", pre), ("est", GradientBoostingRegressor(random_state=random_state))]),
    }
    return models


def _cv_evaluate(models: Dict[str, Pipeline], X: pd.DataFrame, y: pd.Series, cv_splits: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Run cross-validation for each model and return a summary DataFrame of metrics."""
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    rows = []
    for name, pipe in models.items():
        scores = cross_validate(
            pipe,
            X,
            y,
            cv=kf,
            scoring={
                "r2": "r2",
                "neg_mae": "neg_mean_absolute_error",
                "neg_mse": "neg_mean_squared_error",
            },
            n_jobs=None,
            return_train_score=False,
        )
        r2_mean = float(np.mean(scores["test_r2"]))
        r2_std = float(np.std(scores["test_r2"]))
        mae_mean = float(-np.mean(scores["test_neg_mae"]))
        rmse_mean = float(np.sqrt(np.maximum(0.0, -np.mean(scores["test_neg_mse"]))))
        rows.append({
            "model": name,
            "cv_r2_mean": r2_mean,
            "cv_r2_std": r2_std,
            "cv_mae": mae_mean,
            "cv_rmse": rmse_mean,
        })
    res = pd.DataFrame(rows).sort_values(by=["cv_r2_mean"], ascending=False).reset_index(drop=True)
    return res


def _fit_best_and_test(models: Dict[str, Pipeline], cv_results: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[str, Pipeline, Dict[str, float]]:
    """Fit the best model (by CV R^2) on train and evaluate on test. Return name, pipeline, and metrics."""
    best_name = cv_results.iloc[0]["model"]
    best_pipe = models[best_name]
    best_pipe.fit(X_train, y_train)

    y_pred = best_pipe.predict(X_test)
    test_r2 = float(r2_score(y_test, y_pred))
    test_mae = float(mean_absolute_error(y_test, y_pred))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    metrics = {"test_r2": test_r2, "test_mae": test_mae, "test_rmse": test_rmse}
    return best_name, best_pipe, metrics


def _plot_cv_results(cv_results: pd.DataFrame, save_dir: str) -> str:
    """Save a bar plot of CV R^2 by model and return the file path."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    # Draw bars first
    ax = sns.barplot(data=cv_results, x="cv_r2_mean", y="model", orient="h", ci=None)
    # Overlay error bars (robust to single-row case and NaNs)
    means = cv_results["cv_r2_mean"].astype(float).values
    stds = cv_results["cv_r2_std"].astype(float).values if "cv_r2_std" in cv_results.columns else np.zeros_like(means)
    stds = np.nan_to_num(stds, nan=0.0)
    y_positions = np.arange(len(means))
    for i, (m, s) in enumerate(zip(means, stds)):
        plt.errorbar(m, i, xerr=s, fmt='none', ecolor='black', elinewidth=1, capsize=3)
    ax.set_xlabel("CV R^2 (mean ± sd)")
    ax.set_ylabel("Model")
    ax.set_title("5-fold CV Performance (R^2)")
    plt.tight_layout()
    out_path = os.path.join(save_dir, "cv_r2_by_model.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def _plot_test_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, save_dir: str) -> Tuple[str, str]:
    """Save predicted-vs-actual and residual plots. Return their paths."""
    os.makedirs(save_dir, exist_ok=True)
    # Pred vs Actual
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", label="y = x")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Test: Predicted vs Actual")
    plt.legend()
    plt.tight_layout()
    pvspath = os.path.join(save_dir, "test_pred_vs_actual.png")
    plt.savefig(pvspath, dpi=150, bbox_inches="tight")
    plt.close()

    # Residuals
    residuals = y_true - y_pred
    plt.figure(figsize=(7, 4))
    sns.histplot(residuals, kde=True)
    plt.title("Test Residuals (Actual - Pred)")
    plt.xlabel("Residual")
    plt.tight_layout()
    residpath = os.path.join(save_dir, "test_residuals.png")
    plt.savefig(residpath, dpi=150, bbox_inches="tight")
    plt.close()
    return pvspath, residpath


def _get_ohe_feature_names(pre: ColumnTransformer, num_cols: List[str], cat_cols: List[str]) -> List[str]:
    """Construct output feature names from the preprocessor for num + one-hot cat.

    Falls back gracefully if names cannot be retrieved from the encoder.
    """
    names: List[str] = []
    # Numeric features: names preserved
    names.extend(list(num_cols))
    try:
        cat_pipeline = pre.named_transformers_["cat"]
        ohe = None
        if isinstance(cat_pipeline, Pipeline) and "ohe" in cat_pipeline.named_steps:
            ohe = cat_pipeline.named_steps["ohe"]
        if ohe is not None and hasattr(ohe, "get_feature_names_out"):
            ohe_names = list(ohe.get_feature_names_out(cat_cols))
        else:
            # Fallback: approximate names
            ohe_names = [f"{c}_encoded" for c in cat_cols]
        names.extend(ohe_names)
    except Exception:
        # If anything goes wrong, append generic placeholders
        names.extend([f"cat_{i}" for i in range(1, 1 + 1)])
    return names


def _plot_feature_importance(best_pipe: Pipeline, num_cols: List[str], cat_cols: List[str], save_dir: str, top_n: int = 25) -> Optional[str]:
    """Plot feature importances or absolute coefficients for the best model.

    Returns the output path if created; otherwise None.
    """
    os.makedirs(save_dir, exist_ok=True)
    est = best_pipe.named_steps.get("est")
    pre = best_pipe.named_steps.get("pre")
    if est is None or pre is None:
        return None

    feature_names = _get_ohe_feature_names(pre, num_cols, cat_cols)
    values = None
    title = None
    if hasattr(est, "feature_importances_"):
        values = np.asarray(getattr(est, "feature_importances_"))
        title = "Feature Importance (Tree-based)"
    elif hasattr(est, "coef_"):
        coef = np.asarray(getattr(est, "coef_"))
        coef = coef.ravel()
        values = np.abs(coef)
        title = "Absolute Coefficients (Linear)"
    else:
        return None

    # Align lengths if mismatch (defensive)
    k = min(len(feature_names), len(values))
    feature_names = feature_names[:k]
    values = values[:k]

    imp_df = pd.DataFrame({"feature": feature_names, "importance": values}).sort_values(
        by="importance", ascending=False
    ).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_df, x="importance", y="feature")
    plt.title(title)
    plt.xlabel("importance")
    plt.ylabel("feature")
    plt.tight_layout()
    out_path = os.path.join(save_dir, "best_model_feature_importance.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def _save_report(report_path: str, resolved: str, target: str, exclude_patterns: List[str], n_train: int, n_test: int, cv_results: pd.DataFrame, best_name: str, test_metrics: Dict[str, float], top_features_table: pd.DataFrame, model_path: str, plot_paths: List[str]) -> None:
    """Write a human-readable text report summarizing the analysis and results."""
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    lines: List[str] = []
    lines.append("Title: Depression Modeling Report")
    lines.append("")
    lines.append("Author: Noah Hicks")
    lines.append("Date: 2025-09-04")
    lines.append("Note: AI was used to help with the creation and editing of this code. ")
    lines.append("You may assume that AI wrote all the text, and I have approved it. ")
    lines.append("Models used: GPT 5 and Claude 4.1.")
    lines.append("")
    lines.append(f"Dataset: {resolved}")
    lines.append(f"Target: {target}")
    lines.append(f"Excluded features (contains): {exclude_patterns}")
    lines.append(f"Train/Test sizes: {n_train} / {n_test}")
    lines.append("")
    lines.append("Cross-Validation (5-fold) Summary (sorted by R^2):")
    lines.append(cv_results.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    lines.append("")
    lines.append(f"Best model: {best_name}")
    lines.append(f"Test R^2:  {test_metrics['test_r2']:.4f}")
    lines.append(f"Test RMSE: {test_metrics['test_rmse']:.4f}")
    lines.append(f"Test MAE:  {test_metrics['test_mae']:.4f}")
    lines.append("")
    lines.append("Top 15 features by univariate R^2 vs target:")
    lines.append(top_features_table.head(15).to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    lines.append("")
    lines.append(f"Saved model path: {model_path}")
    lines.append("Saved plots:")
    for p in plot_paths:
        lines.append(f" - {p}")
    lines.append("")
    lines.append("Notes:\n- Features containing 'stress' were excluded from modeling as requested.\n- R^2 is the primary metric; lower RMSE/MAE also indicate better fit.")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(path: Optional[str], target: Optional[str], save_csv: str, save_dir: Optional[str], report_path: str, model_out: str, test_size: float = 0.2, cv_folds: int = 5, random_state: int = 42) -> None:
    """Load data, compute per-feature R^2, run CV across models, pick best, evaluate, and save artifacts."""
    exclude_patterns = ["stress"]

    # Load and clean
    df_raw, resolved = se.load_stress_df(path)
    df = se.clean_df(df_raw)

    # Pick target
    tgt = detect_depression_target(df, user_target=target)

    # Per-feature R^2 (skip excluded)
    table = compute_feature_r2_table(df, tgt, exclude_patterns=exclude_patterns)
    if save_csv:
        os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
        table.to_csv(save_csv, index=False)
        print(f"Saved per-feature R^2 CSV: {os.path.abspath(save_csv)}")

    # Top features chart
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        top = table.head(25)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top, x="r2", y="feature", hue="type", dodge=False)
        plt.title("Top R^2 by Feature vs Depression")
        plt.xlabel("R^2 (univariate)")
        plt.ylabel("feature")
        plt.tight_layout()
        r2_plot_path = os.path.join(save_dir, "depression_feature_r2.png")
        plt.savefig(r2_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        r2_plot_path = ""

    # Prepare data and split
    X, y, num_cols, cat_cols = _prepare_data(df, tgt, exclude_patterns=exclude_patterns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    pre = _build_preprocessor(num_cols, cat_cols)
    models = _build_model_pipelines(pre, random_state=random_state)

    # Cross-validation on training set
    cv_results = _cv_evaluate(models, X_train, y_train, cv_splits=cv_folds, random_state=random_state)
    cv_plot_path = _plot_cv_results(cv_results, save_dir or ".")

    # Fit best and evaluate on test
    best_name, best_pipe, test_metrics = _fit_best_and_test(models, cv_results, X_train, y_train, X_test, y_test)
    y_pred_test = best_pipe.predict(X_test)
    pvspath, residpath = _plot_test_diagnostics(y_test.values, y_pred_test, save_dir or ".")
    fi_path = _plot_feature_importance(best_pipe, num_cols, cat_cols, save_dir or ".") or ""

    # Save the best model
    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    joblib.dump(best_pipe, model_out)

    # Build and save report
    plot_paths = [p for p in [r2_plot_path, cv_plot_path, pvspath, residpath, fi_path] if p]
    _save_report(report_path, resolved, tgt, exclude_patterns, len(X_train), len(X_test), cv_results, best_name, test_metrics, table, model_out, plot_paths)
    print("Saved report:", os.path.abspath(report_path))
    print("Saved best model:", os.path.abspath(model_out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute feature-wise R^2 vs depression and build predictive models with CV.")
    parser.add_argument("--path", type=str, default=None, help="Optional explicit CSV path.")
    parser.add_argument("--target", type=str, default=None, help="Target column name; default auto-detect contains 'depress'.")
    parser.add_argument("--save-csv", type=str, default="eda_output/depression_r2_by_feature.csv", help="Output CSV path for per-feature R^2 table.")
    parser.add_argument("--save-dir", type=str, default="eda_output", help="Directory to save charts.")
    parser.add_argument("--report-path", type=str, default="eda_output/depression_model_report.txt", help="Path to save the text report.")
    parser.add_argument("--model-out", type=str, default="eda_output/best_depression_model.joblib", help="Path to save the best model.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size fraction (default 0.2).")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds (default 5).")
    args = parser.parse_args()

    main(
        args.path,
        args.target,
        args.save_csv,
        args.save_dir,
        args.report_path,
        args.model_out,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
    )
