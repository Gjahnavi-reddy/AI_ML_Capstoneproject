import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ANALYTICS_DIR = Path("analytics")
ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = ANALYTICS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV_PATH = ANALYTICS_DIR / "titanic.csv"
CLEANED_CSV_PATH = ANALYTICS_DIR / "titanic_cleaned.csv"

def load_and_profile_titanic():
    print("=" * 60)
    print("Part A - Task 1: Loading and Profiling Titanic Dataset")
    print("=" * 60)
    if RAW_CSV_PATH.exists():
        print(f"Loading cached dataset from {RAW_CSV_PATH}...")
        df = pd.read_csv(RAW_CSV_PATH)
    else:
        print("Fetching Titanic dataset via sns.load_dataset('titanic')...")
        df = sns.load_dataset("titanic")
        df.to_csv(RAW_CSV_PATH, index=False)
        print(f"Saved offline fallback to: {RAW_CSV_PATH}")
    print(f"Dataset Shape: Rows = {df.shape[0]}, Columns = {df.shape[1]}")
    return df

def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Part A - Task 2: Missing-Value Handling")
    print("=" * 60)
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=["embarked", "embark_town"])
    median_age_grouped = df_clean.groupby(["pclass", "sex"])["age"].transform("median")
    df_clean["age"] = df_clean["age"].fillna(median_age_grouped)
    df_clean["age"] = df_clean["age"].fillna(df_clean["age"].median())
    if "deck" in df_clean.columns:
        df_clean["deck"] = df_clean["deck"].astype(object).fillna("Missing")
    df_clean.to_csv(CLEANED_CSV_PATH, index=False)
    print(f"Cleaned dataset saved: {CLEANED_CSV_PATH} (Rows: {len(df_clean)})")
    return df_clean

def detect_iqr_outliers(series: pd.Series, col_name: str):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    print(f"--- Outliers for '{col_name}' ---: Count = {len(outliers)} ({len(outliers)/len(series)*100:.2f}%)")
    return outliers

def run_univariate_analysis(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("Part A - Task 3: Univariate Analysis (Age & Fare)")
    print("=" * 60)
    detect_iqr_outliers(df["age"], "age")
    detect_iqr_outliers(df["fare"], "fare")

    fare_mean = df["fare"].mean()
    fare_median = df["fare"].median()
    fare_mode = df["fare"].mode()[0]
    print(f"Fare Stats: Mode={fare_mode:.2f} < Median={fare_median:.2f} < Mean={fare_mean:.2f} -> RIGHT-SKEWED")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(df["age"], kde=True, ax=axes[0, 0], color="skyblue", bins=30)
    axes[0, 0].set_title("Age Histogram")
    sns.boxplot(x=df["age"], ax=axes[0, 1], color="lightgreen")
    axes[0, 1].set_title("Age Boxplot")
    sns.histplot(df["fare"], kde=True, ax=axes[1, 0], color="salmon", bins=40)
    axes[1, 0].set_title("Fare Histogram")
    sns.boxplot(x=df["fare"], ax=axes[1, 1], color="orchid")
    axes[1, 1].set_title("Fare Boxplot")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "task3_univariate_age_fare.png", dpi=300)
    plt.close()

def run_bivariate_analysis(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("Part A - Task 4: Bivariate Analysis & Correlation Heatmap")
    print("=" * 60)

    corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr_matrix = df[corr_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("6x6 Numeric Feature Correlation Matrix", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "task4_correlation_heatmap.png", dpi=300)
    plt.close()

def run_multivariate_story(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("Part A - Task 5: Multivariate Visual Data Story (4 Distinct Charts)")
    print("=" * 60)

    # Ensure family size exists for analysis
    df_story = df.copy()
    df_story["family_size"] = df_story["sibsp"] + df_story["parch"] + 1

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.set_theme(style="whitegrid")

    # Chart 1: Barplot of Survival by Class and Sex
    sns.barplot(
        data=df_story, x="pclass", y="survived", hue="sex",
        palette={"female": "#e74c3c", "male": "#3498db"}, ci=None, ax=axes[0, 0]
    )
    axes[0, 0].set_title("Chart 1: Survival Rate by Passenger Class and Sex", fontsize=13, weight="bold")
    axes[0, 0].set_ylabel("Survival Rate (0 - 1)")
    axes[0, 0].set_xlabel("Passenger Class")

    # Chart 2: Violin Plot of Age Distribution by Survival and Sex
    sns.violinplot(
        data=df_story, x="sex", y="age", hue="survived",
        split=True, palette={0: "#95a5a6", 1: "#2ecc71"}, ax=axes[0, 1]
    )
    axes[0, 1].set_title("Chart 2: Age Distribution by Survival Outcome and Sex", fontsize=13, weight="bold")
    axes[0, 1].set_ylabel("Age (Years)")

    # Chart 3: Scatter Plot of Fare vs Age conditioned on Survival
    sns.scatterplot(
        data=df_story, x="age", y="fare", hue="survived",
        palette={0: "#e74c3c", 1: "#2980b9"}, alpha=0.7, ax=axes[1, 0]
    )
    axes[1, 0].set_title("Chart 3: Fare vs. Age Conditioned on Survival", fontsize=13, weight="bold")
    axes[1, 0].set_ylabel("Fare Paid (£)")
    axes[1, 0].set_xlabel("Age")
    axes[1, 0].set_ylim(-5, 300)  # Zoom in slightly to clearly view the main clusters

    # Chart 4: Point Plot of Survival by Family Size
    sns.pointplot(
        data=df_story, x="family_size", y="survived",
        color="#8e44ad", markers="o", linestyles="-", ax=axes[1, 1]
    )
    axes[1, 1].set_title("Chart 4: Survival Probability by Family Size", fontsize=13, weight="bold")
    axes[1, 1].set_ylabel("Survival Rate")
    axes[1, 1].set_xlabel("Family Size (SibSp + Parch + 1)")

    plt.tight_layout()
    story_plot_path = PLOTS_DIR / "task5_multivariate_story.png"
    plt.savefig(story_plot_path, dpi=300)
    plt.close()
    print(f"Visual data story saved to: {story_plot_path}")

if __name__ == "__main__":
    raw_df = load_and_profile_titanic()
    cleaned_df = clean_missing_values(raw_df)
    run_univariate_analysis(cleaned_df)
    run_bivariate_analysis(cleaned_df)
    run_multivariate_story(cleaned_df)