# Titanic Analytics & Predictive Modeling Module (`/analytics`)

This directory contains the end-to-end data science pipeline for the Titanic dataset, structured into exploratory data analysis (`01_eda.py`) and leak-free predictive modeling (`02_modeling.py`).

---

## Directory Structure

```text
analytics/
├── 01_eda.py                     # Part A: Loading, cleaning, univariate/bivariate EDA, sanity check
├── 02_modeling.py                # Part B: Leak-free preprocessing, classifiers, tuning, regression, artifact test
├── titanic.csv                   # Single committed offline fallback dataset
├── titanic_cleaned.csv           # Cleaned dataset output from Part A
├── models/
│   └── titanic_best_pipeline.joblib # Serialized end-to-end production pipeline
├── plots/
│   ├── task3_univariate_age_fare.png
│   ├── task4_correlation_heatmap.png
│   ├── task5_multivariate_story.png
│   ├── task6_standardization_check.png
│   ├── task9_decision_tree.png
│   ├── task10_model_evaluations.png
│   └── task13_regression_residuals.png
└── README.md                     # Documentation, metrics, and deployment recommendation



Part A: Exploratory Data Analysis & Cleaning
1. Missing-Value Handling Strategy (Threshold Rules)
embarked & embark_town (Missing: 2 rows, ~0.22%): Falls below the < 5% threshold. Dropped directly via row deletion because the missingness is negligible and does not introduce bias.

age (Missing: 177 rows, ~19.87%): Falls within the 5% to 30% threshold. Imputed using group-wise medians conditioned on pclass and sex to preserve demographic age variations without distorting distributions.

deck (Missing: 687 rows, ~77.10%): Exceeds the > 30% threshold. Imputed with an explicit category label 'Missing' to retain structural cabin deck information without discarding 77% of passenger observations.

2. Univariate Analysis: Outliers & Skewness
IQR Outlier Counts:age: 25 outliers (Values falling outside $[Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}]$).fare: 116 outliers (Extreme ticket prices extending up to £512.33).Fare Skewness Analysis:$\text{Mode (8.05)} < \text{Median (14.45)} < \text{Mean (32.10)}$Conclusion: Because the distribution's mean is pulled heavily upward by luxury ticket prices, the distribution is strictly right-skewed.

Bivariate Analysis & 6×6 Correlation Heatmap
Survival Rate Breakdown:
By Sex: Female: 74.04% | Male: 18.89%
By Passenger Class: 1st Class: 62.62% | 2nd Class: 47.28% | 3rd Class: 24.24%
By Sex + Class Combined:
First Class Female: 96.47% | Second Class Female: 91.89% | Third Class Female: 49.65%
First Class Male: 36.89% | Second Class Male: 15.74% | Third Class Male: 13.70%

Top 2 Strongest Off-Diagonal Correlations (6×6 Matrix):pclass & fare ($r = -0.55$): Demonstrates a strong inverse relationship where lower numerical passenger class numbers (1st class) required significantly higher fares.sibsp & parch ($r = +0.42$): Reflects passenger traveling dynamics, showing that individuals traveling with siblings or spouses frequently had parents or children accompanying them.

4. Multivariate Data Story (Visual Interpretations)
Chart 1 (Survival by Class & Sex): Illustrates the compounding effect of the "women and children first" maritime evacuation protocol and socioeconomic privilege. Female survival remained consistently dominant across all classes, while 1st class passengers gained disproportionately higher survival rates within both genders.

Chart 2 (Age Distribution by Outcome & Sex): Shows survival prioritization for young male children under age 10. In contrast, adult males aged 20–45 suffered the lowest survival rates, whereas female survival rates remained high across all age tiers.

Chart 3 (Fare vs. Age Conditioned on Survival): Non-survivors are heavily concentrated at lower ticket prices (< £30) irrespective of age, reflecting third-class cabin locations situated deep within the vessel.

Chart 4 (Survival Probability by Family Size): Solo travelers and very large families (5+ members) suffered lower survival probabilities. Traveling in a small unit of 2–4 members offered the highest survival rate due to cooperative evacuation without crowd panic.


Part B: Predictive Modeling & Evaluation
1. Stratified Split Justification
The dataset contains a baseline class imbalance (~61.6% non-survivors vs. ~38.4% survivors). A stratified train/test split (test_size=0.2, stratify=y) was enforced to preserve the exact class ratio across both partitions, preventing covariate shift and biased evaluation metrics.

2. Leak-Free Preprocessing Guarantee
All imputers, one-hot encoders, and standard scalers were enclosed inside a scikit-learn ColumnTransformer. Transformers were fit strictly on X_train and applied in transform-only mode to X_test, guaranteeing zero information leakage.

3. Imbalance Handling Comparison
Evaluated on Logistic Regression:

Baseline (No Handling): Accuracy: 0.7978 | Precision: 0.7424 | Recall: 0.7206 | F1: 0.7313

class_weight='balanced': Accuracy: 0.7865 | Precision: 0.7042 | Recall: 0.7500 | F1: 0.7264

SMOTE (Train Fold Only): Accuracy: 0.7809 | Precision: 0.6986 | Recall: 0.7353 | F1: 0.7164

Conclusion: class_weight='balanced' provided the highest recall boost for identifying survivors. SMOTE, while improving recall, introduced boundary ambiguity on mixed tabular features compared to exact loss reweighting.


4. Hyperparameter Tuning (Random Forest)
Grid Parameters: n_estimators: [50, 100, 200], max_depth: [3, 5, 8, None], max_features: ['sqrt', 'log2', None]
Best Configuration: {'max_depth': 5, 'max_features': 'sqrt', 'n_estimators': 100}
Out-of-Bag (OOB) Score: 0.8327
Test ROC-AUC: 0.8654


5. Regression Side-Task: Predicting Fare
Metrics: MAE: £19.34 | RMSE: £34.28 | $R^2$: 0.4287 | Adjusted $R^2$: 0.4014Heteroscedasticity Finding: The residual plot exhibits pronounced heteroscedasticity. Residuals are clustered tightly around £0 for low predicted fares (< £30) but fan out substantially as predicted values increase, with errors spanning upwards of +£100 to +£300 for luxury first-class tickets. This violates the constant variance assumption of Ordinary Least Squares (OLS).


Final Multi-Task Model Comparison Table

Model Name,Task Type,Accuracy,Precision,Recall,F1 Score,ROC-AUC,MAE (£),RMSE (£),R²,Adj R²
Logistic Regression,Classification,0.7978,0.7424,0.7206,0.7313,0.8491,—,—,—,—
Decision Tree,Classification,0.7921,0.8298,0.5735,0.6783,0.8229,—,—,—,—
Random Forest,Classification,0.8202,0.8103,0.6912,0.7460,0.8654,—,—,—,—
Multivariate Linear Regressor,Regression,—,—,—,—,—,£19.34,£34.28,0.4287,0.4014


Final Deployment Recommendation
I recommend deploying the Random Forest Classifier for predicting passenger survival outcomes.

It demonstrates the highest generalization capacity on the unseen test set, leading across overall accuracy (82.02%), F1 score (0.7460), and ROC-AUC (0.8654), noticeably outperforming Logistic Regression (AUC: 0.8491) and Decision Tree (AUC: 0.8229). Furthermore, with a precision of 81.03%, the Random Forest significantly reduces false positives compared to linear baselines while modeling multi-way feature interactions (such as sex, passenger class, and family size).



