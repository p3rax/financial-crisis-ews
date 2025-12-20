# Project Proposal  
## Early Warning System for Financial Crises

**Author:** Andrea Perani  
**Email:** andrea.perani@unil.ch  

---

## 1. Project Title and Category

**Title:** Early Warning System for Financial Crises  
**Category:** Data Science, Machine Learning, Finance  

---

## 2. Motivation and Background

Systemic financial crises such as those of 2008 and 2020 revealed how quickly financial stress can propagate across markets.  
Most traditional financial indicators are primarily reactive rather than predictive, limiting their usefulness for timely policy intervention or portfolio risk management.

This project proposes a data-driven Early Warning System (EWS) capable of detecting early co-movement patterns among key macro-financial indicators — such as credit spreads, market volatility, liquidity conditions, and the yield curve — that signal the build-up of systemic stress.  
The objective is to develop a transparent and interpretable machine learning framework that provides early alerts and probabilistic risk estimates of future systemic stress episodes.

---

## 3. Methodology and Tools

The analysis will rely on weekly macro-financial data obtained from public sources such as FRED and Yahoo Finance.  
The St. Louis Financial Stress Index (STLFSI4) will be used **exclusively as the target variable** to define systemic stress episodes.

Specifically, the model will predict whether STLFSI4 exceeds its long-term mean plus two standard deviations within a forward-looking horizon of approximately one month.  
All other indicators — including high-yield and investment-grade credit spreads (HY OAS, IG OAS), volatility measures (VIX, MOVE), dollar strength (DXY), and the yield-curve slope (T10Y2Y) — will serve strictly as predictors.  
This setup explicitly avoids any circularity between features and the target variable, as clarified and approved by the teaching assistants.

Data preprocessing and analysis will be implemented using `pandas`, `numpy`, and `scikit-learn`.  
The modeling stage will compare Logistic Regression, Random Forest, and XGBoost classifiers using time-series–aware cross-validation to respect the temporal structure of the data.  
Model interpretability will be addressed through SHAP value decomposition.

---

## 4. Challenges and Risk Mitigation

Key challenges include data alignment across sources, severe class imbalance due to the rarity of crisis events, potential model overfitting, and interpretability of results.

These risks will be mitigated through careful resampling and alignment of time series, chronological validation using time-series splits, appropriate evaluation metrics for imbalanced classification (e.g. PR-AUC), and post-hoc explainability methods.  
Probability calibration techniques will be applied to ensure that predicted risk estimates remain meaningful and stable.

---

## 5. Evaluation Criteria

The model will be considered successful if it consistently identifies stress build-ups preceding known crisis periods and outperforms a baseline logistic model in terms of ROC-AUC and PR-AUC metrics on a hold-out test set.  
Reproducibility and clarity of visualization will constitute central evaluation criteria.

---

## 6. Optional Extensions

If time permits, the analysis may be extended to include banking-sector proxies such as the XLF ETF, interbank stress measures (e.g. LIBOR–OIS), and a simple Streamlit dashboard for dynamic visualization of systemic risk levels.

---