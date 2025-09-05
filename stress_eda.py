# Title: stress_eda.py
# 
# Author: Noah Hicks
# Date: 2025-09-04
# Note: AI was used to help with the creation and editing of this code. 
# You may assume that AI wrote all the text, and I have approved it. 
# Models used: GPT 5 and Claude 4.1.
#
# Summary: Lightweight utilities to find/load the Student Stress datasets CSV,
# perform quick cleaning, print data summaries, and generate common EDA plots.
#
# Usage (from a notebook in the same directory):
#     import stress_eda as se
#     df, path = se.load_stress_df()
#     df = se.clean_df(df)
#     se.basic_summary(df, dataset_name=path)
#     se.plot_numeric_distributions(df)
#     se.plot_categorical_top_counts(df)
#     se.plot_correlations(df)
#     se.plot_stress_targets(df)

from __future__ import annotations

import os
import re
import argparse
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configure plotting aesthetics once for consistency
sns.set(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


# -----------------------------
# File discovery and loading
# -----------------------------

def find_stress_csv(base_dir: Optional[str] = None) -> Optional[str]:
    """Find the stress dataset CSV path.

    Searches for likely stress CSVs in a sibling folder named
    'data/' (preferred) or 'Student Stress Monitoring Datasets/' relative
    to the given base_dir (or current working directory if not provided).
    Returns the first match based on a priority order.
    """
    if base_dir is None:
        base_dir = os.getcwd()

    data_dir = os.path.join(base_dir, "data")
    legacy_dir = os.path.join(base_dir, "Student Stress Monitoring Datasets")
    search_dirs: List[str] = []
    # Prefer the new cleaned repository location first
    if os.path.isdir(data_dir):
        search_dirs.append(data_dir)
    # Backward compatibility for existing folder name
    if os.path.isdir(legacy_dir):
        search_dirs.append(legacy_dir)
    # Fallback: search recursively under base_dir if neither exists
    if not search_dirs:
        search_dirs = [base_dir]

    # Priority file names
    priority_names = [
        "StressLevelDataset.csv",
        "Stress_Dataset.csv",
    ]

    found_paths: List[str] = []
    for d in search_dirs:
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(".csv") and ("stress" in f.lower()):
                    found_paths.append(os.path.join(root, f))

    # Sort by our priority list first, then by path for stability
    def sort_key(p: str) -> Tuple[int, str]:
        name = os.path.basename(p)
        try:
            prio = priority_names.index(name)
        except ValueError:
            prio = len(priority_names)
        return (prio, p)

    found_paths.sort(key=sort_key)
    return found_paths[0] if found_paths else None


def load_stress_df(path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[str]]:
    """Load the stress dataset into a DataFrame and standardize columns.

    - If `path` is None, attempt to auto-locate a stress CSV via `find_stress_csv`.
    - Returns (df, resolved_path).
    """
    resolved = path or find_stress_csv()
    if resolved is None:
        raise FileNotFoundError(
            "Could not find a stress dataset CSV. Ensure the folder 'Student Stress Monitoring Datasets/' exists and contains a stress CSV."
        )

    df = pd.read_csv(resolved)
    df = standardize_columns(df)
    return df, resolved


# -----------------------------
# Cleaning and helpers
# -----------------------------

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with snake_case, stripped, alnum+underscore column names.

    This minimizes downstream typos and makes plotting code consistent.
    """
    def to_snake(name: str) -> str:
        name = name.strip()
        name = re.sub(r"[^0-9a-zA-Z]+", "_", name)  # non-alnum -> underscore
        name = re.sub(r"_+", "_", name).strip("_")
        return name.lower()

    df2 = df.copy()
    df2.columns = [to_snake(c) for c in df2.columns]
    return df2


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: strip strings, coerce obvious numeric-like columns.

    Keep this intentionally light; modeling steps can do task-specific cleaning.
    """
    out = df.copy()

    # Strip strings
    for c in out.select_dtypes(include=["object"]).columns:
        out[c] = out[c].astype(str).str.strip()

    # Coerce numeric-like columns (that have >80% numeric-looking values)
    for c in out.columns:
        if c in out.select_dtypes(include=[np.number]).columns:
            continue
        series = out[c].dropna().astype(str)
        numeric_like = series.str.fullmatch(r"[-+]?\d*\.?\d+").mean() if len(series) else 0
        if numeric_like >= 0.8:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


# -----------------------------
# Summaries
# -----------------------------

def basic_summary(df: pd.DataFrame, dataset_name: Optional[str] = None) -> None:
    """Print head, shape, dtypes, describe, and missingness overview.

    Works in both notebooks (rich display) and scripts/terminals (plain text).
    """
    title = f"Dataset: {dataset_name}" if dataset_name else "Dataset"
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print("Shape:", df.shape)
    print("\nDtypes:\n", df.dtypes)

    # Quick overview
    print("\nHead:")
    try:
        display(df.head())  # type: ignore[name-defined]
    except Exception:
        # Fallback for non-notebook environments
        print(df.head().to_string(index=False))

    # Describe numeric and object summaries
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if num_cols:
        print("\nNumeric describe:")
        try:
            display(df[num_cols].describe().T)  # type: ignore[name-defined]
        except Exception:
            print(df[num_cols].describe().T.to_string())
    if obj_cols:
        print("\nCategorical top counts:")
        try:
            display(df[obj_cols].agg(lambda s: s.value_counts().head(5)))  # type: ignore[name-defined]
        except Exception:
            print(df[obj_cols].agg(lambda s: s.value_counts().head(5)).to_string())

    # Missingness
    missing = df.isna().mean().sort_values(ascending=False)
    print("\nMissingness (fraction):")
    try:
        display(missing.to_frame("missing_frac").query("missing_frac > 0"))  # type: ignore[name-defined]
    except Exception:
        print(missing.to_frame("missing_frac").query("missing_frac > 0").to_string())


# -----------------------------
# Plots
# -----------------------------

def _tight_layout():
    """Helper to reduce overlap in figures."""
    try:
        plt.tight_layout()
    except Exception:
        pass


def plot_numeric_distributions(df: pd.DataFrame, max_cols: int = 12, save_dir: Optional[str] = None, show: bool = True) -> None:
    """Plot histograms for numeric columns (capped for readability).

    If save_dir is provided, saves a single figure image. If show is False, does not display.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        print("No numeric columns found.")
        return

    cols = num_cols[:max_cols]
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = np.array(axes).reshape(-1)

    for ax, c in zip(axes, cols):
        sns.histplot(df[c].dropna(), kde=True, ax=ax)
        ax.set_title(f"{c} (n={df[c].notna().sum()})")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Numeric Distributions", y=1.02)
    _tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "numeric_distributions.png")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_categorical_top_counts(df: pd.DataFrame, max_cols: int = 9, top_n: int = 10, save_dir: Optional[str] = None, show: bool = True) -> None:
    """Bar plots for the most frequent categories across object/category columns.

    If save_dir is provided, saves a single figure image. If show is False, does not display.
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        print("No categorical columns found.")
        return

    cols = cat_cols[:max_cols]
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 4.5))
    axes = np.array(axes).reshape(-1)

    for ax, c in zip(axes, cols):
        vc = df[c].value_counts(dropna=False).head(top_n)
        sns.barplot(x=vc.values, y=vc.index.astype(str), ax=ax, orient="h")
        ax.set_title(c)
        ax.set_xlabel("count")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Top Categorical Levels", y=1.02)
    _tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "categorical_top_counts.png")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_correlations(df: pd.DataFrame, method: str = "spearman", max_vars: int = 25, save_dir: Optional[str] = None, show: bool = True) -> None:
    """Heatmap of numeric correlations using the chosen method (default: spearman).

    If save_dir is provided, saves the heatmap image. If show is False, does not display.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        print("Not enough numeric columns for correlation.")
        return

    cols = num_cols[:max_vars]
    corr = df[cols].corr(method=method)
    plt.figure(figsize=(min(1 + 0.4 * len(cols), 16), min(1 + 0.4 * len(cols), 16)))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True)
    plt.title(f"{method.title()} Correlation (top {len(cols)} numeric)")
    _tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"correlations_{method}.png")
        plt.gcf().savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(plt.gcf())


def plot_stress_targets(df: pd.DataFrame, save_dir: Optional[str] = None, show: bool = True) -> None:
    """Plot distributions for columns likely representing stress labels/targets.

    If save_dir is provided, saves one image per target-like column. If show is False, does not display.
    """
    # Heuristic: any column name containing 'stress' or 'anxiety' or 'depression'
    patt = re.compile(r"(stress|anxiety|depress|score|level)")
    cols = [c for c in df.columns if patt.search(c)]
    if not cols:
        print("No obvious stress-related columns found by name.")
        return

    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            plt.figure(figsize=(7, 4))
            sns.histplot(df[c].dropna(), kde=True)
            plt.title(f"Distribution: {c}")
            _tight_layout()
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                out_path = os.path.join(save_dir, f"stress_{c}.png")
                plt.gcf().savefig(out_path, bbox_inches="tight", dpi=150)
                print(f"Saved: {out_path}")
            if show:
                plt.show()
            else:
                plt.close(plt.gcf())
        else:
            plt.figure(figsize=(7, 4))
            vc = df[c].value_counts(dropna=False)
            sns.barplot(x=vc.index.astype(str), y=vc.values)
            plt.title(f"Counts: {c}")
            plt.xticks(rotation=30, ha="right")
            _tight_layout()
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                out_path = os.path.join(save_dir, f"stress_{c}.png")
                plt.gcf().savefig(out_path, bbox_inches="tight", dpi=150)
                print(f"Saved: {out_path}")
            if show:
                plt.show()
            else:
                plt.close(plt.gcf())

if __name__ == "__main__":
    # Simple CLI so running this file does useful work from the terminal.
    parser = argparse.ArgumentParser(description="Run EDA on the Student Stress dataset.")
    parser.add_argument("--path", type=str, default=None, help="Optional explicit CSV path to load.")
    parser.add_argument("--save-dir", type=str, default="eda_output", help="Directory to save plots.")
    parser.add_argument("--show", action="store_true", help="Show plots interactively in a window.")
    args = parser.parse_args()

    try:
        df_raw, resolved_path = load_stress_df(args.path)
        df = clean_df(df_raw)
        basic_summary(df, dataset_name=resolved_path)

        # Generate plots; by default save to disk and do not block on GUI windows
        plot_numeric_distributions(df, save_dir=args.save_dir, show=args.show)
        plot_categorical_top_counts(df, save_dir=args.save_dir, show=args.show)
        plot_correlations(df, method="spearman", save_dir=args.save_dir, show=args.show)
        plot_stress_targets(df, save_dir=args.save_dir, show=args.show)

        print("EDA complete. Plots saved to:", os.path.abspath(args.save_dir))
    except Exception as e:
        print("ERROR:", e)

# -----------------------------
# End of module
# -----------------------------
