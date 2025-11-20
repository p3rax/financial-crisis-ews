# financial-crisis-ews
# Early Warning System for Financial Crises

This repository contains the code and experiments for a data-driven **Early Warning
System (EWS)** for systemic financial stress.

The goal is to use weekly macro-financial data (credit spreads, volatility indices,
FX, yield curve, financial stress index) to predict whether the **STLFSI4** stress
index will enter a “high stress” regime in the following weeks.

## Project Structure

- `src/` – Python modules for data download, preprocessing, feature engineering and
  model training.
- `data/` – Raw and processed datasets (not all files will be tracked on GitHub).
- `notebooks/` – Jupyter notebooks for exploratory analysis and visualizations.
- `models/` – Saved machine learning models and scalers.
- `results/` – Figures and metrics used in the final report.
- `tests/` – Simple tests to check that core functions work as expected.
- `docs/` – Notes and outline for the written report.
- `AI_USAGE.md` – Declaration of how AI tools were used during the project.

## Course Context

This is the individual capstone project for the course **“Introduction to Data
Science and Advanced Programming” (HEC Lausanne, 2025)**. Deliverables:

- ~10-page LaTeX report
- GitHub repository with clean and reproducible code
- ~10–15 minute presentation video

More details will be added as the project progresses.