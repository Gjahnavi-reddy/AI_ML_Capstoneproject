# Module 2 — Analytics Pipeline

## 1. Overview

This module performs exploratory data analysis (EDA), data preprocessing, classification, imbalance handling, hyperparameter tuning, and regression using the Titanic dataset.

The pipeline uses a single cleaned Titanic dataset saved as `titanic.csv`. The same dataset is used for EDA and machine learning tasks.

---

## 2. Setup

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### Requirements

The module uses:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* imbalanced-learn
* joblib

---

## 3. How to Run

Run the EDA pipeline first:

```bash
python 01_eda.py
```

This performs data profiling, missing-value analysis, univariate and bivariate analysis, outlier analysis, correlation analysis, standardization checks, and generates the required charts.

Then run the modeling pipeline:

```bash
python 02_modeling.py
```

This performs stratified train-test splitting, preprocessing, classification, imbalance handling, Random Forest hyperparameter tuning, regression, evaluation, and model saving.

---

## 4. Dataset

The Titanic dataset was loaded using Seaborn and saved locally as:

```text
titanic.csv
```

The final dataset contains 889 rows and 16 columns.

The target variable for classification is:

```text
survived
```

The main features used for machine learning are:

* pclass
* sex
* age
* sibsp
* parch
* fare
* embarked

The `age_zscore` and `fare_zscore` columns were created for EDA standardization checks and were not used as machine-learning features.

---

## 5. Missing Value Analysis

The main missing-value percentages were:

| Feature     | Missing (%) |
| ----------- | ----------: |
| deck        |      77.22% |
| age         |      19.87% |
| embarked    |       0.22% |
| embark_town |       0.22% |

The `deck` feature had a very high missing-value percentage and was therefore not used as a machine-learning feature.

For the modeling pipeline:

* Numerical features use median imputation.
* Categorical features use most-frequent imputation.

---

## 6. Exploratory Data Analysis

### Age and Fare

The analysis identified outliers using the IQR method.

For `age`:

* Q1 = 22.0
* Q3 = 35.0
* Lower bound = 2.5
* Upper bound = 54.5
* Number of outliers = 65

For `fare`:

* Q1 = 7.8958
* Q3 = 31.0
* Lower bound = -26.7605
* Upper bound = 65.6563
* Number of outliers = 114

Fare was found to be right-skewed because its mean was considerably higher than its median.

### Survival by Sex

The survival rates were:

* Female: 74.04%
* Male: 18.89%

This shows a strong relationship between sex and survival.

### Survival by Passenger Class

The survival rates were:

* First class: 62.62%
* Second class: 47.28%
* Third class: 24.24%

Passengers in higher classes had higher survival rates.

### Survival by Sex and Class

The highest survival rate was for female first-class passengers:

* Female, First Class: 96.74%

The lowest was for male third-class passengers:

* Male, Third Class: 13.54%

These results indicate that both sex and passenger class were important factors associated with survival.

---

## 7. Correlation Analysis

Important correlations identified were:

* `pclass` vs `fare`: -0.5482
* `sibsp` vs `parch`: 0.4145

The negative correlation between passenger class and fare indicates that lower class numbers were generally associated with higher fares.

The positive correlation between `sibsp` and `parch` indicates that passengers travelling with siblings/spouses were also somewhat likely to travel with parents/children.

---

## 8. Standardization Check

Z-score standardization was checked for age and fare.

After standardization:

```text
Age Z-score  -> Mean: 0.0000, Std: 1.0000
Fare Z-score -> Mean: 0.0000, Std: 1.0000
```

The standardized columns were used for analysis only and were excluded from the machine-learning features.

---

## 9. Machine Learning Preprocessing

A stratified 80/20 train-test split was used to preserve the target-class distribution.

The preprocessing pipeline contains:

### Numerical features

* Median imputation
* StandardScaler

### Categorical features

* Most-frequent imputation
* OneHotEncoder with `handle_unknown='ignore'`

A `ColumnTransformer` combines the numerical and categorical preprocessing steps.

This preprocessing is fitted only on the training data to avoid data leakage.

---

## 10. Classification Models

Three classification models were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

### Results

| Model               | Accuracy | Precision | Recall |     F1 |    AUC |
| ------------------- | -------: | --------: | -----: | -----: | -----: |
| Logistic Regression |   0.8090 |    0.7833 | 0.6912 | 0.7344 | 0.8610 |
| Decision Tree       |   0.7697 |    0.6901 | 0.7206 | 0.7050 | 0.7541 |
| Random Forest       |   0.8202 |    0.7813 | 0.7353 | 0.7576 | 0.8179 |

### Interpretation

Random Forest achieved the highest accuracy and F1-score among the three models.

Logistic Regression achieved the highest AUC.

Based on the overall classification performance, Random Forest was selected for further tuning.

---

## 11. Class Imbalance Handling

Logistic Regression was compared using:

* Baseline
* `class_weight='balanced'`
* SMOTE

| Method                | Precision | Recall |     F1 |
| --------------------- | --------: | -----: | -----: |
| Baseline              |    0.7833 | 0.6912 | 0.7344 |
| Class Weight Balanced |    0.7183 | 0.7500 | 0.7338 |
| SMOTE                 |    0.7353 | 0.7353 | 0.7353 |

The baseline produced the highest precision.

Class weighting produced the highest recall.

SMOTE produced the highest F1-score in this comparison, although the improvement over the baseline was very small.

---

## 12. Random Forest Hyperparameter Tuning

GridSearchCV was used to tune the Random Forest model.

The best parameters were:

```text
max_depth = 5
max_features = sqrt
n_estimators = 200
```

Best cross-validation F1-score:

```text
0.7408
```

The tuned Random Forest achieved an out-of-bag (OOB) score of:

```text
0.8214
```

The final trained pipeline was saved as:

```text
best_titanic_pipeline.joblib
```

The saved pipeline was also reloaded and tested successfully using raw, unprocessed input data.

---

## 13. Regression Side Task

A regression model was used to predict passenger fare.

The target variable was:

```text
fare
```

The evaluation results were:

| Metric      |   Value |
| ----------- | ------: |
| MAE         | 21.0986 |
| RMSE        | 41.7021 |
| R²          |  0.3482 |
| Adjusted R² |  0.3091 |

The residual analysis was saved as:

```text
fare_residual_plot.png
```

The R² value indicates that the selected features explain part of the variation in fare, but a substantial amount remains unexplained.

---

## 14. Generated Outputs

The module produces the following major outputs:

* `titanic.csv`
* `classification_results.csv`
* `imbalance_comparison.csv`
* `best_titanic_pipeline.joblib`
* `age_boxplot.png`
* `age_histogram.png`
* `fare_boxplot.png`
* `fare_histogram.png`
* `correlation_heatmap.png`
* `chart_1_survival_by_sex.png`
* `chart_2_survival_by_class.png`
* `chart_3_survival_by_sex_class.png`
* `chart_4_age_fare_survival.png`
* `decision_tree.png`
* `roc_curves.png`
* `fare_residual_plot.png`

---

## 15. Design Decisions

The following design decisions were made:

1. A single cleaned Titanic dataset is used throughout the module to maintain consistency between EDA and modeling.
2. Stratified splitting is used to preserve the target-class distribution.
3. Preprocessing is performed using a scikit-learn pipeline to reduce data leakage risk and make the model reusable.
4. Median imputation is used for numerical missing values because it is less sensitive to outliers than mean imputation.
5. One-hot encoding is used for categorical features.
6. Logistic Regression, Decision Tree, and Random Forest are compared to evaluate different classification approaches.
7. Class weighting and SMOTE are compared to study the effect of class imbalance.
8. GridSearchCV is used to tune the Random Forest hyperparameters.
9. The complete tuned preprocessing and modeling pipeline is saved using Joblib so it can be reloaded for prediction.

---

## 16. Final Conclusion

The analysis shows that passenger sex and passenger class were strongly associated with Titanic survival.

Among the tested classifiers, Random Forest provided the strongest overall classification performance based on accuracy and F1-score. The tuned Random Forest pipeline was saved as a reusable Joblib artifact.

The regression analysis showed that the selected passenger-related features had moderate predictive ability for fare.
