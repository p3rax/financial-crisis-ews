# Early Warning System for Financial Crises
**Author:** Andrea Perani<br>
**Course:** Data Science & Advanced Programming — HEC Lausanne (2025)<br>
**Project Type:** Final Project — Data Science • Machine Learning • Finance

---

## 1. Overview

This project implements a data-driven Early Warning System (EWS) aimed at detecting the buildup of systemic financial stress before major financial crises emerge.

The system uses weekly macro-financial indicators — including credit spreads, volatility indices, yield-curve slope, and dollar strength — to predict whether systemic stress is likely to increase within the next 4 weeks.

The entire pipeline is fully automated and reproducible: starting from raw data, all datasets, models, metrics, and figures are generated end-to-end by running a single command.

```bash
python main.py
```

Models implemented:
- Logistic Regression
- Random Forest
- XGBoost

Model interpretability is provided via SHAP values.

Probability calibration is applied to the final selected model to ensure reliable risk estimates.

---

## 2. Features

- Automatic download of macro-financial data (FRED and Yahoo Finance)
- Robust preprocessing and weekly dataset construction
- Time-series–aware cross-validation (no shuffling)
- End-to-end ML pipeline: training, evaluation, explainability
- Hold-out test evaluation with ROC and Precision–Recall curves
- Leakage-safe probability calibration for the final model (reliability analysis)
- SHAP-based model interpretation
- Optional additional visualization module
- Fully reproducible Conda environment
- Test suite implemented with `pytest`
- Clean, modular codebase under `src/`

---

## 3. Installation

1. Clone the repository:

```bash
git clone https://github.com/p3rax/financial-crisis-ews-clean
cd financial-crisis-ews-clean
```

2. Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate financial-crisis-ews
```

If the environment already exists (e.g. from a previous setup), update it instead:

```bash
conda env update -f environment.yml --prune
conda activate financial-crisis-ews
```

To verify that the environment was created and activated correctly, run:

```bash
conda env list
which python
```

The active environment should be `financial-crisis-ews`, and the Python path should point to that environment.

Requirements: A working Conda installation (Anaconda or Miniconda) is required.
Platform note: The project was tested on Linux (x86_64). The Conda environment is platform-agnostic and should work on macOS and Windows as well.

---

## 4. Running the Full Pipeline

Run the complete workflow with:

```bash
python main.py
```

This performs the following steps:
1.	Download raw financial data (if missing)
2.	Build the processed EWS dataset
3.	Train machine learning models
4.	Evaluate models on a hold-out test set
5.	Generate SHAP-based explanations
6.	Generate additional summary plots

All required folders are created automatically if they do not exist.

---

## 5. Reproducibility

This project is fully reproducible by design.

- A global random seed is defined in `main.py` and used to initialize both the Python and NumPy random number generators.
- Stochastic behavior is controlled via this globally fixed random state, ensuring deterministic execution of the pipeline.
- Time-series splits are deterministic and strictly chronological.
- Running `python main.py` multiple times yields identical results, given the same raw data snapshots.

Note: Results are reproducible conditional on the raw CSV files stored in `data/raw/`.
If data providers update historical values, re-downloading may lead to different datasets.

---

## 6. Running Tests

To run the full test suite:

```bash
python -m pytest -q
```

What the tests cover:
- Data availability: checks that the required raw CSV files exist in data/raw/ (tests are skipped if raw data has not been downloaded yet).
- Time-series integrity (no leakage): verifies that the date-based train/test split is strictly chronological (train dates precede test dates).
- Model training and prediction validity: ensures models can be trained on a window containing both classes and that predict_proba outputs well-formed probabilities in [0, 1].
- Reproducibility: confirms that training with a fixed seed yields identical predictions.
- Explainability (SHAP): verifies that SHAP computation runs on a tree-based model and returns arrays with consistent dimensions.

Expected outcome:
- After running python main.py at least once, all tests should pass.
- On a fresh clone (no generated data yet), tests that require raw/processed data are automatically skipped, and the remaining unit tests still run.

This setup keeps the suite robust both on a clean environment and after full pipeline execution, while explicitly validating reproducibility and time-series correctness.

---

## 7. Outputs

Models (generated automatically):
- `models/logistic_regression.joblib`
- `models/random_forest.joblib`
- `models/xgboost.joblib`

Tables:
- `results/tables/model_cv_results.csv`
- `results/tables/model_test_results.csv`
- `results/tables/calibration_results.csv`

Figures:
- ROC curves (test set)
- Precision–Recall curves (test set)
- Timeline of systemic stress and predicted risk
- SHAP summary (beeswarm) plot
- SHAP bar plot
- SHAP dependence plot
- Test ROC-AUC comparison bar plot
- Test Average Precision (PR-AUC) comparison bar plot
- Calibration curve (test set)

---

## 8. Methodology Summary

Frequency: Weekly
Target: STLFSI4 exceeding mean + 2 standard deviations within 4 weeks
Validation: TimeSeriesSplit (5 folds, expanding window)
Test split: Chronological (date-based)
Metrics: ROC-AUC, PR-AUC
Models: Logistic Regression, Random Forest, XGBoost
Explainability: SHAP values (sample size = 400)
Probability calibration: Sigmoid calibration applied to the final selected model using time-series cross-validation on the training set

---

## 9. Project Proposal and Technical Report

This project includes both a proposal and a final technical report, stored under the `docs/` directory:

- Project Proposal
  `docs/proposal/PROPOSAL.md`  
  Documents the motivation, scope, and methodological plan of the project as submitted to the Teaching Assistants.

- Final Technical Report (LaTeX source)
  `docs/report/REPORT.tex`  
  The report follows the official course structure (Abstract, Introduction, Methodology, Implementation, Results, Conclusion, Appendix).

The compiled PDF of the final report is submitted separately via the official course submission channel, as required.

### Compiling the LaTeX Report (Optional)

The repository includes the full LaTeX source of the report to ensure transparency and reproducibility.

To compile the report locally, a LaTeX distribution providing `pdflatex` (e.g. TeX Live) is required:

```bash
cd docs/report
pdflatex REPORT.tex
```

If LaTeX is not available locally, the report can be compiled using Overleaf by uploading the contents of the `docs/report/` directory.

This repository intentionally tracks only the LaTeX source files (not the compiled PDF), in line with best practices for version control and reproducibility.

---

## 10. Git and Data Policy

The following directories are excluded via `.gitignore`:
- `data/`
- `models/`
- `results/`
- `__pycache__/`
- `.vscode/`

All excluded files and folders are automatically regenerated by running:

```bash
python main.py
```

---

## 11. AI Tools Usage

AI tools (e.g. ChatGPT) were used as learning and support tools for:
- debugging
- code review
- documentation refinement

All design decisions, modeling choices, and final code structure were fully understood and validated by the author.
Details are documented in `AI_USAGE.md`, as required by the course rules.

---

## 12. Author

Andrea Perani  
Master in Finance — HEC Lausanne  
andrea.perani@unil.ch

---

## 13. License

This project is intended exclusively for academic evaluation within the context of the Data Science & Advanced Programming course at HEC Lausanne.