import pandas as pd

print("=" * 60)
print("LOADING CLEANED TITANIC DATA")
print("=" * 60)

# Load the cleaned offline dataset created by 01_eda.py
df = pd.read_csv("titanic.csv")

print("Dataset loaded successfully.")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())
# ---------------------------------------------------------
# STRATIFIED TRAIN / TEST SPLIT
# ---------------------------------------------------------

from sklearn.model_selection import train_test_split

print("\n" + "=" * 60)
print("CLASS BALANCE")
print("=" * 60)

class_balance = df["survived"].value_counts()
class_percentage = df["survived"].value_counts(normalize=True) * 100

print("\nClass counts:")
print(class_balance)

print("\nClass percentages:")
print(class_percentage)


# ---------------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------------

target = "survived"

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[features]
y = df[target]


# ---------------------------------------------------------
# STRATIFIED SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting class distribution:")
print(y_test.value_counts(normalize=True))

print("\nStratified train/test split completed.")
# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Numeric and categorical columns
numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]


# Numeric preprocessing
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# Combine both preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


print("\n" + "=" * 60)
print("PREPROCESSING PIPELINE CREATED")
print("=" * 60)

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)
print("Preprocessing will be fitted only on training data.")
# ---------------------------------------------------------
# TRAIN THREE CLASSIFICATION MODELS
# ---------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


print("\n" + "=" * 60)
print("TRAINING CLASSIFICATION MODELS")
print("=" * 60)


# Logistic Regression
logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]
)


# Decision Tree
decision_tree_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(
            random_state=42
        ))
    ]
)


# Random Forest
random_forest_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ]
)


# Train all three models
logistic_pipeline.fit(X_train, y_train)

decision_tree_pipeline.fit(X_train, y_train)

random_forest_pipeline.fit(X_train, y_train)


print("Logistic Regression trained successfully.")
print("Decision Tree trained successfully.")
print("Random Forest trained successfully.")
# ---------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)
import matplotlib.pyplot as plt


models = {
    "Logistic Regression": logistic_pipeline,
    "Decision Tree": decision_tree_pipeline,
    "Random Forest": random_forest_pipeline
}

results = []


print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)


plt.figure(figsize=(8, 6))


for model_name, model in models.items():

    # Predictions
    y_pred = model.predict(X_test)

    # Probability for ROC/AUC
    y_probability = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_probability
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n" + "-" * 50)
    print(model_name)
    print("-" * 50)

    print("Confusion Matrix:")
    print(cm)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

    # ROC curve
    fpr, tpr, _ = roc_curve(
        y_test,
        y_probability
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc:.3f})"
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc
    })


# Random classifier reference line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()

plt.savefig("roc_curves.png")

plt.show()


# ---------------------------------------------------------
# MODEL COMPARISON TABLE
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("CLASSIFICATION MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)

results_df.to_csv(
    "classification_results.csv",
    index=False
)
# ---------------------------------------------------------
# DECISION TREE VISUALIZATION
# ---------------------------------------------------------

from sklearn.tree import plot_tree

print("\n" + "=" * 60)
print("DECISION TREE VISUALIZATION")
print("=" * 60)

# Get the fitted preprocessing step
tree_preprocessor = decision_tree_pipeline.named_steps["preprocessor"]

# Transform training data using the already-fitted preprocessor
X_train_tree = tree_preprocessor.transform(X_train)

# Get feature names after one-hot encoding
feature_names = tree_preprocessor.get_feature_names_out()

# Get the fitted Decision Tree
tree_model = decision_tree_pipeline.named_steps["model"]

plt.figure(figsize=(24, 12))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    rounded=True,
    fontsize=7
)

plt.title("Decision Tree for Titanic Survival")
plt.tight_layout()

plt.savefig("decision_tree.png", dpi=150)

plt.show()

print("Decision tree visualization saved as decision_tree.png")
# ---------------------------------------------------------
# IMBALANCE HANDLING COMPARISON
# ---------------------------------------------------------

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


print("\n" + "=" * 60)
print("IMBALANCE HANDLING COMPARISON")
print("=" * 60)


# ---------------------------------------------------------
# 1. BASELINE
# ---------------------------------------------------------

baseline_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]
)

baseline_model.fit(X_train, y_train)

baseline_pred = baseline_model.predict(X_test)

baseline_precision = precision_score(
    y_test, baseline_pred, zero_division=0
)

baseline_recall = recall_score(
    y_test, baseline_pred, zero_division=0
)

baseline_f1 = f1_score(
    y_test, baseline_pred, zero_division=0
)


# ---------------------------------------------------------
# 2. CLASS WEIGHT = BALANCED
# ---------------------------------------------------------

balanced_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ]
)

balanced_model.fit(X_train, y_train)

balanced_pred = balanced_model.predict(X_test)

balanced_precision = precision_score(
    y_test, balanced_pred, zero_division=0
)

balanced_recall = recall_score(
    y_test, balanced_pred, zero_division=0
)

balanced_f1 = f1_score(
    y_test, balanced_pred, zero_division=0
)


# ---------------------------------------------------------
# 3. SMOTE
# ---------------------------------------------------------

smote_model = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", LogisticRegression(max_iter=1000))
    ]
)

smote_model.fit(X_train, y_train)

smote_pred = smote_model.predict(X_test)

smote_precision = precision_score(
    y_test, smote_pred, zero_division=0
)

smote_recall = recall_score(
    y_test, smote_pred, zero_division=0
)

smote_f1 = f1_score(
    y_test, smote_pred, zero_division=0
)


# ---------------------------------------------------------
# COMPARISON TABLE
# ---------------------------------------------------------

imbalance_results = pd.DataFrame([
    {
        "Strategy": "Baseline",
        "Precision": baseline_precision,
        "Recall": baseline_recall,
        "F1": baseline_f1
    },
    {
        "Strategy": "class_weight=balanced",
        "Precision": balanced_precision,
        "Recall": balanced_recall,
        "F1": balanced_f1
    },
    {
        "Strategy": "SMOTE",
        "Precision": smote_precision,
        "Recall": smote_recall,
        "F1": smote_f1
    }
])

print("\nImbalance Handling Results:")

print(
    imbalance_results.to_string(index=False)
)

imbalance_results.to_csv(
    "imbalance_comparison.csv",
    index=False
)
oob_score=True
# ---------------------------------------------------------
# RANDOM FOREST HYPERPARAMETER TUNING
# ---------------------------------------------------------

from sklearn.model_selection import GridSearchCV


print("\n" + "=" * 60)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 60)


# Random Forest with OOB enabled
rf_for_tuning = RandomForestClassifier(
    random_state=42,
    oob_score=True
)


# Complete pipeline
rf_tuning_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf_for_tuning)
    ]
)


# Parameter grid
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"]
}


# Grid Search
grid_search = GridSearchCV(
    estimator=rf_tuning_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)


grid_search.fit(X_train, y_train)


print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation F1:")
print(grid_search.best_score_)


# ---------------------------------------------------------
# OOB SCORE
# ---------------------------------------------------------

best_rf_pipeline = grid_search.best_estimator_

best_rf_model = best_rf_pipeline.named_steps["model"]

print("\nOOB Score:")
print(best_rf_model.oob_score_)
# ---------------------------------------------------------
# REGRESSION SIDE-TASK: PREDICT FARE
# ---------------------------------------------------------

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


print("\n" + "=" * 60)
print("REGRESSION: PREDICTING FARE")
print("=" * 60)


# Features used to predict fare
regression_features = [
    "survived",
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "embarked"
]

X_reg = df[regression_features]
y_reg = df["fare"]


# Train/test split
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)


# Numeric and categorical columns
reg_numeric_features = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch"
]

reg_categorical_features = [
    "sex",
    "embarked"
]


# Regression preprocessing
reg_numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

reg_categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


reg_preprocessor = ColumnTransformer(
    transformers=[
        ("num", reg_numeric_transformer, reg_numeric_features),
        ("cat", reg_categorical_transformer, reg_categorical_features)
    ]
)


# Complete regression pipeline
regression_pipeline = Pipeline(
    steps=[
        ("preprocessor", reg_preprocessor),
        ("model", LinearRegression())
    ]
)


# Train
regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)


# Predict
y_reg_pred = regression_pipeline.predict(X_reg_test)


# Metrics
mae = mean_absolute_error(
    y_reg_test,
    y_reg_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        y_reg_pred
    )
)

r2 = r2_score(
    y_reg_test,
    y_reg_pred
)


# Adjusted R-squared
n = len(y_reg_test)

p = len(
    regression_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

adjusted_r2 = 1 - (
    (1 - r2) * (n - 1)
    / (n - p - 1)
)


print("\nRegression Metrics:")
print(f"MAE         : {mae:.4f}")
print(f"RMSE        : {rmse:.4f}")
print(f"R²          : {r2:.4f}")
print(f"Adjusted R² : {adjusted_r2:.4f}")


# ---------------------------------------------------------
# RESIDUAL PLOT
# ---------------------------------------------------------

residuals = y_reg_test - y_reg_pred

plt.figure(figsize=(8, 6))

plt.scatter(
    y_reg_pred,
    residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title("Fare Regression Residual Plot")

plt.tight_layout()

plt.savefig(
    "fare_residual_plot.png"
)

plt.show()


print("\nResidual plot saved as fare_residual_plot.png")
# ---------------------------------------------------------
# SAVE BEST COMPLETE PIPELINE
# ---------------------------------------------------------

import joblib

print("\n" + "=" * 60)
print("SAVING BEST COMPLETE PIPELINE")
print("=" * 60)

full_pipeline = best_rf_pipeline

joblib.dump(
    full_pipeline,
    "best_titanic_pipeline.joblib"
)

print("Complete pipeline saved as:")
print("best_titanic_pipeline.joblib")
# ---------------------------------------------------------
# RELOAD AND TEST SAVED PIPELINE
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RELOADING SAVED PIPELINE")
print("=" * 60)

loaded_pipeline = joblib.load(
    "best_titanic_pipeline.joblib"
)

# Raw input example — NO preprocessing applied manually
raw_input = X_test.iloc[[0]]

prediction = loaded_pipeline.predict(raw_input)

print("Raw input:")
print(raw_input)

print("\nPrediction:")
print(prediction)

print("\nReloaded pipeline works successfully on raw input.")
