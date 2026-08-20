# House Price Prediction (Ames Housing)

A machine learning regression project predicting house sale prices from 79 features. **Graded A.**

## Overview
End-to-end regression pipeline on the Ames Housing dataset (1,460 houses) to predict sale prices.

## Approach
- **Data cleaning:** Four-tiered missing-value strategy (None/0/median/mode) based on data context, across 19 columns
- **EDA:** Identified OverallQual and GrLivArea as strongest predictors; log-transformed the right-skewed target
- **Encoding:** Ordinal for quality-ranked features, one-hot for nominal categories
- **Modeling:** Compared Linear, Ridge, Lasso, Random Forest, LightGBM
- **Evaluation:** MAE, RMSE, R²; 5-fold cross-validation; diagnosed overfitting via train/test gap

## Key Result
Ridge/Linear Regression best (R² ≈ 0.92). Simpler linear models beat the tree models — the small dataset caused trees to overfit, showing model complexity should match data size.

## Tech Stack
Python, pandas, scikit-learn, LightGBM, Matplotlib, Seaborn
