# Depression Modeling and Stress EDA

Lightweight utilities to explore the "Student Stress Monitoring Datasets" and run a simple depression-target modeling pipeline with cross-validation and feature importance plots.

## Repository Structure

- `stress_eda.py` — Utilities to find/load the dataset, basic cleaning, dataset summary, and common EDA plots.
- `depression_feature_importance.py` — Modeling pipeline to rank features by univariate R^2 vs. a depression-like target and evaluate multiple models with CV.
- `requirements.txt` — Python dependencies.
- `.gitignore` — Excludes data and environment artifacts from version control.
- `data/` — Place the CSV(s) you download here (ignored by git). Alternatively, an existing folder named `Student Stress Monitoring Datasets/` will also be detected for backward compatibility.
- `eda_output/` — Auto-created folder where plots, reports, and models are saved (ignored by git).

## Setup

1. Create a virtual environment (recommended):
   - Windows PowerShell
     ```powershell
     python -m venv .venv
     .venv\\Scripts\\Activate.ps1
     ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Acquisition (Not Committed)

Raw data is not tracked in this repo. Download it yourself and place the CSV(s) into the `data/` directory.

- Kaggle Dataset: Student Stress Monitoring Datasets, Sultanul Ovi, 2025. Available at: https://www.kaggle.com/datasets/mdsultanulislamovi/student-stress-monitoring-datasets
- The dataset is licensed under Apache 2.0 and is updated annually.

This project will automatically look for CSVs in `data/` first, then in a legacy folder named `Student Stress Monitoring Datasets/` if present.

## Usage

Run quick EDA and plots:
```bash
python stress_eda.py --save-dir eda_output
```
- This prints summaries and saves plots into `eda_output/`.
- Use `--show` to display plots interactively.

Run depression modeling with CV and feature ranking:
```bash
python depression_feature_importance.py \
  --save-csv eda_output/depression_r2_by_feature.csv \
  --save-dir eda_output \
  --report-path eda_output/depression_model_report.txt \
  --model-out eda_output/best_depression_model.joblib
```
- By default, the target column is auto-detected as the first column name containing `depress`. Override with `--target <column_name>`.
- Features containing `stress` are excluded from modeling by default.

Both scripts attempt to locate the dataset automatically and will raise a helpful error if not found.

## Citation

If you use the dataset, please cite it as follows (provided on the Kaggle page):

BibTeX:
```
@article{ovi2025protecting,
  title={Protecting Student Mental Health with a Context-Aware Machine Learning Framework for Stress Monitoring},
  author={Ovi, Md Sultanul Islam and Hossain, Jamal and Rahi, Md Raihan Alam and Akter, Fatema},
  journal={arXiv preprint arXiv:2508.01105},
  year={2025}
}
```

Standard citation:

Md Sultanul Islam Ovi, Jamal Hossain, Md Raihan Alam Rahi, and Fatema Akter. "Protecting Student Mental Health with a Context-Aware Machine Learning Framework for Stress Monitoring." arXiv preprint arXiv:2508.01105 (2025).

Make sure you also mention the dataset was accessed via Kaggle:

Kaggle Dataset: Student Stress Monitoring Datasets, Sultanul Ovi, 2025. Available at: https://www.kaggle.com/datasets/mdsultanulislamovi/student-stress-monitoring-datasets

License: The dataset is licensed under Apache 2.0 and is updated annually.

## Notes

- Data files are intentionally ignored by git via `.gitignore`.
- Generated artifacts (`eda_output/`, plots, and `.joblib` models) are also ignored to keep the repo clean and reproducible.
- Set a random state or adjust CLI arguments in `depression_feature_importance.py` if you need deterministic splits or different CV settings.
