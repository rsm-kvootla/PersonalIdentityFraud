# Identity Fraud Detection Using Machine Learning

## Project Overview

This project focuses on the end-to-end development of a machine learning model to detect identity-based fraud in synthetic application data. The dataset simulates real-world financial product applications and includes key PII fields such as SSN, DOB, address, and phone number, along with a binary fraud label. The goal was to build a robust, generalizable model capable of ranking high-risk applications using behavioral and linkage-based features, while minimizing false positives.

## Dataset

- **Source**: Synthetic dataset of 1,000,000 application records
- **Fields**: Name, SSN, DOB, address, ZIP code, phone number, application date, fraud label
- **Fraud Rate**: ~1.44%
- **Structure**: Clean, no missing values, categorical and numeric fields properly typed

## Feature Engineering

Over 100 identity linkage and temporal behavior features were generated, including:
- Frequency counts (`address_count_0_by_30`, `ssn_dob_count_14`)
- Maximum reuse indicators (`max_count_by_address_7`)
- Recency (`fulladdress_day_since`)
- Cross-field uniqueness

After filter and wrapper-based feature selection, a final set of **15 high-impact variables** was selected for modeling.

## Model Development

The following algorithms were trained and evaluated:
- Logistic Regression
- Decision Tree
- Random Forest
- LightGBM
- Neural Network (MLPClassifier)
- CatBoost

### Evaluation Metric
- **False Discovery Rate (FDR) @ 3%**: Measures percentage of true frauds within top 3% of scored applications
- **Train/Test/OOT Splits**: Models evaluated on held-out test and future (OOT) data

## Final Model

- **Type**: LightGBM Classifier
- **Hyperparameters**:
  - `n_estimators`: 150
  - `learning_rate`: 0.05
  - `max_depth`: 8
  - `num_leaves`: 31
  - `min_child_samples`: 20
- **Performance**:
  - **Train FDR@3%**: 0.635
  - **Test FDR@3%**: 0.627
  - **OOT FDR@3%**: 0.593

## Financial Impact Analysis

- **Fraud savings**: $4,000 per fraud caught
- **False positive cost**: $100 per investigated case
- **Optimal score threshold**: **Bin 2**, maximizing annual net savings
- **Estimated net benefit**: **$3.2 billion annually**

## Conclusion

This project delivers a well-tuned fraud detection model that is technically sound, financially optimized, and generalizes effectively across time. The solution is suitable for deployment in real-world fraud prevention pipelines where investigation budgets are limited and accuracy is critical.

## Files Included

- `Data explore vs1_3 applications.ipynb`: Data exploration
- `applications_clean_make_variables.ipynb`, `dedup columns.ipynb`, `feature_selection_binary_classification.ipynb`: Creation and selection of features
- `model_training.ipynb`: Training and evaluation of all model types
- `applications_models.ipynb`: Score cutoff optimization and cost modeling
- `applications data.csv` main dataset
- `vars.csv`, `vars_final.csv`: List of variables created and finalized respoectively
- `FDR_trn.xlsx`, `FDR_tst.xlsx`, `FDR_oot.xlsx`: Binned evaluation results


