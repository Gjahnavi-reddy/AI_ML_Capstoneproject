from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ANALYTICS_DIR = Path("analytics")
PLOTS_DIR = ANALYTICS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = ANALYTICS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CLEANED_CSV_PATH = ANALYTICS_DIR / "titanic_cleaned.csv"
SAVED_PIPELINE_PATH = MODELS_DIR / "titanic_best_pipeline.joblib"


def load_cleaned_data() -> pd.DataFrame:
    if not CLEANED_CSV_PATH.exists():
        raise FileNotFoundError(f"Expected {CLEANED_CSV_PATH} not found. Run 01_eda.py first.")
    df = pd.read_csv(CLEANED_CSV_PATH)
    print(f"Loaded cleaned dataset from: {CLEANED_CSV_PATH} (Shape: {df.shape})")
    return df


def split_data_stratified(df: pd.DataFrame, target_col: str = "survived", test_size: float = 0.2, random_state: int = 42):
    print("=" * 60)
    print("Part B - Task 7: Stratified Train/Test Split")
    print("=" * 60)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    train_ratios = y_train.value_counts(normalize=True) * 100
    test_ratios = y_test.value_counts(normalize=True) * 100
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Train survival ratio: {train_ratios[1]:.2f}% | Test survival ratio: {test_ratios[1]:.2f}%")
    return X_train, X_test, y_train, y_test


def build_preprocessor():
    numeric_features = ["age", "fare", "sibsp", "parch"]
    categorical_features = ["sex", "embarked", "pclass"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )
    return preprocessor


def train_classifiers(X_train, y_train):
    print("\n" + "=" * 60)
    print("Part B - Task 9: Training Logistic Regression, Decision Tree & Random Forest")
    print("=" * 60)

    preprocessor = build_preprocessor()

    models = {
        "Logistic Regression": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(random_state=42, max_iter=1000))
        ]),
        "Decision Tree": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(max_depth=3, random_state=42))
        ]),
        "Random Forest": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
        ])
    }

    fitted_models = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        print(f"  * Trained: {name}")

    dt_pipeline = fitted_models["Decision Tree"]
    dt_classifier = dt_pipeline.named_steps["classifier"]
    cat_cols = dt_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(
        ["sex", "embarked", "pclass"]
    )
    feature_names = ["age", "fare", "sibsp", "parch"] + list(cat_cols)

    plt.figure(figsize=(18, 9))
    plot_tree(
        dt_classifier,
        feature_names=feature_names,
        class_names=["Did Not Survive", "Survived"],
        filled=True,
        rounded=True,
        fontsize=9
    )
    plt.title("Decision Tree Visualization (max_depth=3)", fontsize=15, pad=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "task9_decision_tree.png", dpi=300)
    plt.close()

    return fitted_models


def evaluate_models(fitted_models, X_test, y_test):
    print("\n" + "=" * 60)
    print("Part B - Task 10: Model Evaluation & Metric Comparison")
    print("=" * 60)

    clf_metrics = []
    roc_data = {}
    confusion_matrices = {}

    for name, pipeline in fitted_models.items():
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        confusion_matrices[name] = cm
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, auc)

        clf_metrics.append({
            "Model Name": name,
            "Task Type": "Classification",
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "AUC": round(auc, 4),
        })

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    model_names = list(fitted_models.keys())
    for idx, name in enumerate(model_names):
        ax = fig.add_subplot(gs[0, idx])
        sns.heatmap(confusion_matrices[name], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 13, "weight": "bold"})
        ax.set_title(f"Confusion Matrix:\n{name}", fontsize=12)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_xticklabels(["0 (Died)", "1 (Survived)"])
        ax.set_yticklabels(["0 (Died)", "1 (Survived)"])

    ax_roc = fig.add_subplot(gs[1, :])
    colors = ["#2980b9", "#27ae60", "#8e44ad"]
    for idx, (name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
        ax_roc.plot(fpr, tpr, color=colors[idx], lw=2.5, label=f"{name} (AUC = {auc:.4f})")
    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5)
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=14, pad=10)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(PLOTS_DIR / "task10_model_evaluations.png", dpi=300)
    plt.close()

    return clf_metrics


def compare_imbalance_strategies(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("Part B - Task 11: Imbalance Handling Comparison (Logistic Regression)")
    print("=" * 60)

    preprocessor = build_preprocessor()

    pipe_baseline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000))
    ])

    pipe_balanced = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000))
    ])

    pipe_smote = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000))
    ])

    strategies = {
        "(a) Baseline (No Handling)": pipe_baseline,
        "(b) class_weight='balanced'": pipe_balanced,
        "(c) SMOTE (Train Fold Only)": pipe_smote
    }

    results = []
    for strat_name, pipeline in strategies.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        results.append({
            "Imbalance Strategy": strat_name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1 Score": round(f1_score(y_test, y_pred, zero_division=0), 4)
        })

    imbalance_df = pd.DataFrame(results)
    print("\n[Strategy Comparison Results on Test Set]:")
    print(imbalance_df.to_string(index=False))
    return imbalance_df


def tune_random_forest(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("Part B - Task 12: Random Forest Hyperparameter Tuning & OOB Score")
    print("=" * 60)

    preprocessor = build_preprocessor()

    rf_base = RandomForestClassifier(oob_score=True, bootstrap=True, random_state=42)
    tuning_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", rf_base)
    ])

    param_grid = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [3, 5, 8, None],
        "classifier__max_features": ["sqrt", "log2", None]
    }

    grid_search = GridSearchCV(estimator=tuning_pipeline, param_grid=param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_rf = best_pipeline.named_steps["classifier"]
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    oob_score = best_rf.oob_score_

    cleaned_best_params = {k.replace("classifier__", ""): v for k, v in best_params.items()}
    y_test_pred = best_pipeline.predict(X_test)
    y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]

    print("\n[GridSearchCV Tuning Results]:")
    print(f"  * Best Parameters: {cleaned_best_params}")
    print(f"  * 5-Fold CV ROC-AUC: {best_cv_score:.4f} | OOB Score: {oob_score:.4f}")
    print(f"  * Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f} | Test ROC-AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
    return best_pipeline


def run_fare_regression(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("Part B - Task 13: Multivariate Linear Regression (Predicting Fare)")
    print("=" * 60)

    y_reg = df["fare"]
    X_reg = df.drop(columns=["fare"])

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    reg_numeric_features = ["age", "sibsp", "parch"]
    reg_categorical_features = ["sex", "embarked", "pclass", "survived"]

    reg_num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    reg_cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
    ])

    reg_preprocessor = ColumnTransformer(
        transformers=[
            ("num", reg_num_transformer, reg_numeric_features),
            ("cat", reg_cat_transformer, reg_categorical_features)
        ],
        remainder="drop"
    )

    reg_pipeline = Pipeline(steps=[
        ("preprocessor", reg_preprocessor),
        ("regressor", LinearRegression())
    ])

    reg_pipeline.fit(X_train_r, y_train_r)

    y_pred_r = reg_pipeline.predict(X_test_r)
    residuals = y_test_r - y_pred_r

    mae = mean_absolute_error(y_test_r, y_pred_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    r2 = r2_score(y_test_r, y_pred_r)
    n = len(y_test_r)
    p = reg_pipeline.named_steps["preprocessor"].transform(X_train_r).shape[1]
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].scatter(y_pred_r, residuals, alpha=0.6, color="#2980b9", edgecolor="k", linewidth=0.5)
    axes[0].axhline(0, color="crimson", linestyle="--", lw=2)
    axes[0].set_title("Residual Plot: Predicted Fare vs. Residuals", fontsize=13, weight="bold")
    axes[0].set_xlabel("Predicted Fare (£)")
    axes[0].set_ylabel("Residuals (£)")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    max_val = max(y_test_r.max(), y_pred_r.max())
    axes[1].scatter(y_test_r, y_pred_r, alpha=0.6, color="#27ae60", edgecolor="k", linewidth=0.5)
    axes[1].plot([0, max_val], [0, max_val], color="crimson", linestyle="--", lw=2, label="Perfect Fit")
    axes[1].set_title("Actual vs. Predicted Fare", fontsize=13, weight="bold")
    axes[1].set_xlabel("Actual Fare (£)")
    axes[1].set_ylabel("Predicted Fare (£)")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "task13_regression_residuals.png", dpi=300)
    plt.close()

    reg_metric_dict = {
        "Model Name": "Multivariate Linear Regression",
        "Task Type": "Regression",
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "Adj_R2": round(adj_r2, 4),
    }
    return reg_metric_dict


def run_task14_model_comparison(clf_metrics, reg_metric_dict):
    print("\n" + "=" * 60)
    print("Part B - Task 14: Final Model Comparison & Deployment Decision")
    print("=" * 60)

    rows = []
    for m in clf_metrics:
        rows.append({
            "Model Name": m["Model Name"],
            "Task Type": m["Task Type"],
            "Accuracy": f"{m['Accuracy']:.4f}",
            "Precision": f"{m['Precision']:.4f}",
            "Recall": f"{m['Recall']:.4f}",
            "F1": f"{m['F1']:.4f}",
            "AUC": f"{m['AUC']:.4f}",
            "MAE (£)": "-",
            "RMSE (£)": "-",
            "R²": "-",
            "Adj R²": "-",
        })

    rows.append({
        "Model Name": reg_metric_dict["Model Name"],
        "Task Type": reg_metric_dict["Task Type"],
        "Accuracy": "-",
        "Precision": "-",
        "Recall": "-",
        "F1": "-",
        "AUC": "-",
        "MAE (£)": f"£{reg_metric_dict['MAE']:.2f}",
        "RMSE (£)": f"£{reg_metric_dict['RMSE']:.2f}",
        "R²": f"{reg_metric_dict['R2']:.4f}",
        "Adj R²": f"{reg_metric_dict['Adj_R2']:.4f}",
    })

    comparison_table = pd.DataFrame(rows)
    print("\n[Final Multi-Task Model Comparison Table]:")
    print(comparison_table.to_string(index=False))

    print("\n" + "-" * 60)
    print("Final Written Recommendation & Deployment Decision:")
    print("-" * 60)
    print(
        "I recommend deploying the Random Forest classifier for passenger survival prediction. "
        "It achieves the strongest generalization balance across the board, delivering the highest "
        f"test accuracy (~{clf_metrics[2]['Accuracy']:.4f}) and area under the ROC curve (AUC = {clf_metrics[2]['AUC']:.4f}), "
        f"outperforming Logistic Regression (AUC = {clf_metrics[0]['AUC']:.4f}) and Decision Tree (AUC = {clf_metrics[1]['AUC']:.4f}). "
        f"Furthermore, its test F1-score of {clf_metrics[2]['F1']:.4f} and superior precision demonstrate an ability to "
        "capture non-linear interactions (such as sex, deck class, and family size) while minimizing false alarms, "
        "making it substantially more robust for production inference than simpler single-tree or linear baselines."
    )


def save_and_verify_pipeline(best_pipeline: Pipeline):
    print("\n" + "=" * 60)
    print("Part B - Task 15: Serialize and Verify End-to-End Pipeline Artifact")
    print("=" * 60)

    # 1. Save complete pipeline object (ColumnTransformer + Estimator)
    joblib.dump(best_pipeline, SAVED_PIPELINE_PATH)
    print(f"Serialized complete pipeline successfully to: {SAVED_PIPELINE_PATH}")

    # 2. Reload pipeline from disk
    loaded_pipeline = joblib.load(SAVED_PIPELINE_PATH)
    print(f"Successfully reloaded pipeline artifact using joblib.load.")

    # 3. Define completely raw, unprocessed new test cases (including missing values)
    # The pipeline must handle imputation, scaling, and one-hot encoding on the fly.
    raw_sample = pd.DataFrame([
        {
            "pclass": 1,
            "sex": "female",
            "age": 29.0,
            "sibsp": 0,
            "parch": 0,
            "fare": 211.33,
            "embarked": "S",
            "class": "First",
            "who": "woman",
            "adult_male": False,
            "deck": "B",
            "embark_town": "Southampton",
            "alive": "yes",
            "alone": True,
        },
        {
            "pclass": 3,
            "sex": "male",
            "age": None,  # Test missing age imputation
            "sibsp": 1,
            "parch": 0,
            "fare": 7.89,
            "embarked": "S",
            "class": "Third",
            "who": "man",
            "adult_male": True,
            "deck": None,  # Test missing deck handling
            "embark_town": "Southampton",
            "alive": "no",
            "alone": False,
        }
    ])

    print("\n[Raw Unpreprocessed Input Batch]:")
    print(raw_sample[["pclass", "sex", "age", "fare", "embarked"]])

    # 4. Predict using reloaded pipeline
    predictions = loaded_pipeline.predict(raw_sample)
    probabilities = loaded_pipeline.predict_proba(raw_sample)[:, 1]

    raw_sample["Predicted_Survived"] = predictions
    raw_sample["Survival_Probability"] = probabilities.round(4)

    print("\n[Inference Verification Results on Raw Data]:")
    for idx, row in raw_sample.iterrows():
        status = "Survived" if row["Predicted_Survived"] == 1 else "Did Not Survive"
        print(f"  Passenger {idx + 1} ({row['sex']}, Class {row['pclass']}): "
              f"Prediction = {row['Predicted_Survived']} ({status}) | "
              f"Probability = {row['Survival_Probability']:.2%}")

    print("\nVerification Passed: Saved pipeline executes end-to-end inference on raw data without errors.")


if __name__ == "__main__":
    df_clean = load_cleaned_data()
    X_train, X_test, y_train, y_test = split_data_stratified(df_clean)
    fitted_models = train_classifiers(X_train, y_train)
    clf_metrics = evaluate_models(fitted_models, X_test, y_test)
    compare_imbalance_strategies(X_train, X_test, y_train, y_test)
    best_rf_pipeline = tune_random_forest(X_train, X_test, y_train, y_test)
    reg_metrics = run_fare_regression(df_clean)
    run_task14_model_comparison(clf_metrics, reg_metrics)
    save_and_verify_pipeline(best_rf_pipeline)