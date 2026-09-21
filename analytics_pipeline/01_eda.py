import pandas as pd
import seaborn as sns


# ---------------------------------------------------------
# LOAD TITANIC DATASET
# ---------------------------------------------------------

print("=" * 60)
print("LOADING TITANIC DATASET")
print("=" * 60)

df = sns.load_dataset("titanic")


# ---------------------------------------------------------
# REQUIRED OFFLINE FALLBACK
# ---------------------------------------------------------

df.to_csv("titanic.csv", index=False)

print("Titanic dataset loaded successfully.")
print("Offline copy saved as titanic.csv")


# ---------------------------------------------------------
# BASIC DATA PROFILE
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

df.info()


print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(df.describe())


print("\n" + "=" * 60)
print("DATASET SHAPE")
print("=" * 60)

print(df.shape)


# ---------------------------------------------------------
# MISSING VALUE PERCENTAGES
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUE PERCENTAGES")
print("=" * 60)

missing_percentage = (
    df.isnull().mean() * 100
)

missing_percentage = (
    missing_percentage[missing_percentage > 0]
    .sort_values(ascending=False)
)

print(missing_percentage)
# ---------------------------------------------------------
# MISSING VALUE HANDLING
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUE HANDLING")
print("=" * 60)

# deck: 77.22% missing
# Too much missing data for reliable imputation, so drop column.
df = df.drop(columns=["deck"])

print("Dropped 'deck' because 77.22% of its values were missing.")


# age: 19.87% missing
# Between 5% and 30%, so use median imputation.
age_median = df["age"].median()

df["age"] = df["age"].fillna(age_median)

print(
    f"Filled missing 'age' values using median: "
    f"{age_median:.2f}"
)


# embarked: 0.22% missing
# Under 5%, so drop rows with missing values.
df = df.dropna(subset=["embarked"])

print("Dropped rows with missing 'embarked' values.")


# embark_town: 0.22% missing
# Under 5%, so drop rows with missing values.
df = df.dropna(subset=["embark_town"])

print("Dropped rows with missing 'embark_town' values.")


# ---------------------------------------------------------
# VERIFY CLEANED DATA
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CLEANED DATASET")
print("=" * 60)

print("Shape after cleaning:", df.shape)

print("\nRemaining missing values:")

remaining_missing = df.isnull().sum()

print(
    remaining_missing[
        remaining_missing > 0
    ]
)
# ---------------------------------------------------------
# UNIVARIATE ANALYSIS
# AGE AND FARE
# ---------------------------------------------------------

import matplotlib.pyplot as plt


print("\n" + "=" * 60)
print("UNIVARIATE ANALYSIS")
print("=" * 60)


# ---------------------------------------------------------
# AGE HISTOGRAM
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.hist(df["age"], bins=30)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution")
plt.tight_layout()
plt.savefig("age_histogram.png")
plt.show()


# ---------------------------------------------------------
# AGE BOX PLOT
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.boxplot(df["age"].dropna())
plt.ylabel("Age")
plt.title("Age Box Plot")
plt.tight_layout()
plt.savefig("age_boxplot.png")
plt.show()


# ---------------------------------------------------------
# FARE HISTOGRAM
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.hist(df["fare"], bins=30)
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.title("Fare Distribution")
plt.tight_layout()
plt.savefig("fare_histogram.png")
plt.show()


# ---------------------------------------------------------
# FARE BOX PLOT
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.boxplot(df["fare"].dropna())
plt.ylabel("Fare")
plt.title("Fare Box Plot")
plt.tight_layout()
plt.savefig("fare_boxplot.png")
plt.show()


# ---------------------------------------------------------
# IQR OUTLIER FUNCTION
# ---------------------------------------------------------

def count_iqr_outliers(series):

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outliers = series[
        (series < lower_bound) |
        (series > upper_bound)
    ]

    return len(outliers), lower_bound, upper_bound


# ---------------------------------------------------------
# AGE OUTLIERS
# ---------------------------------------------------------

age_outliers, age_lower, age_upper = count_iqr_outliers(
    df["age"]
)

print("\nAGE IQR ANALYSIS")
print("Q1:", df["age"].quantile(0.25))
print("Q3:", df["age"].quantile(0.75))
print("Lower bound:", age_lower)
print("Upper bound:", age_upper)
print("Number of age outliers:", age_outliers)


# ---------------------------------------------------------
# FARE OUTLIERS
# ---------------------------------------------------------

fare_outliers, fare_lower, fare_upper = count_iqr_outliers(
    df["fare"]
)

print("\nFARE IQR ANALYSIS")
print("Q1:", df["fare"].quantile(0.25))
print("Q3:", df["fare"].quantile(0.75))
print("Lower bound:", fare_lower)
print("Upper bound:", fare_upper)
print("Number of fare outliers:", fare_outliers)


# ---------------------------------------------------------
# FARE MEAN, MEDIAN AND MODE
# ---------------------------------------------------------

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode()[0]

print("\nFARE STATISTICS")
print("Mean:", fare_mean)
print("Median:", fare_median)
print("Mode:", fare_mode)


# ---------------------------------------------------------
# FARE SKEWNESS INTERPRETATION
# ---------------------------------------------------------

if fare_mean > fare_median > fare_mode:

    fare_skew = "right-skewed"

elif fare_mean < fare_median < fare_mode:

    fare_skew = "left-skewed"

else:

    fare_skew = "approximately symmetric or not strictly ordered"


print("Fare distribution:", fare_skew)
# ---------------------------------------------------------
# BIVARIATE ANALYSIS
# SURVIVAL RATES
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("BIVARIATE ANALYSIS")
print("=" * 60)


# ---------------------------------------------------------
# SURVIVAL RATE BY SEX
# ---------------------------------------------------------

survival_by_sex = (
    df.groupby("sex")["survived"]
    .mean()
    .mul(100)
)

print("\nSurvival Rate by Sex:")
print(survival_by_sex)


# ---------------------------------------------------------
# SURVIVAL RATE BY PCLASS
# ---------------------------------------------------------

survival_by_pclass = (
    df.groupby("pclass")["survived"]
    .mean()
    .mul(100)
)

print("\nSurvival Rate by Passenger Class:")
print(survival_by_pclass)


# ---------------------------------------------------------
# SURVIVAL RATE BY SEX AND PCLASS
# ---------------------------------------------------------

survival_by_sex_pclass = (
    df.groupby(["sex", "pclass"])["survived"]
    .mean()
    .mul(100)
)

print("\nSurvival Rate by Sex and Passenger Class:")
print(survival_by_sex_pclass)


# ---------------------------------------------------------
# BOOLEAN MASKING EXAMPLES
# ---------------------------------------------------------

female_first_class = df[
    (df["sex"] == "female") &
    (df["pclass"] == 1)
]

male_third_class = df[
    (df["sex"] == "male") &
    (df["pclass"] == 3)
]

print("\nFemale passengers in First Class:")
print(
    female_first_class["survived"].mean() * 100
)

print("\nMale passengers in Third Class:")
print(
    male_third_class["survived"].mean() * 100
)
# ---------------------------------------------------------
# CORRELATION MATRIX
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

corr_matrix = df[correlation_columns].corr()

print(corr_matrix)


# ---------------------------------------------------------
# FIND TWO STRONGEST CORRELATIONS
# ---------------------------------------------------------

corr_pairs = []

for i in range(len(correlation_columns)):
    for j in range(i + 1, len(correlation_columns)):

        feature_1 = correlation_columns[i]
        feature_2 = correlation_columns[j]

        correlation_value = corr_matrix.loc[
            feature_1,
            feature_2
        ]

        corr_pairs.append(
            (
                feature_1,
                feature_2,
                correlation_value,
                abs(correlation_value)
            )
        )

corr_pairs.sort(
    key=lambda x: x[3],
    reverse=True
)

print("\nTwo strongest correlations:")

for pair in corr_pairs[:2]:

    print(
        f"{pair[0]} vs {pair[1]}: "
        f"{pair[2]:.4f}"
    )


# ---------------------------------------------------------
# CORRELATION HEATMAP
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True
)

plt.title("Correlation Matrix of Selected Titanic Features")
plt.tight_layout()

plt.savefig("correlation_heatmap.png")

plt.show()
# ---------------------------------------------------------
# MULTIVARIATE DATA STORY
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MULTIVARIATE DATA STORY")
print("=" * 60)


# ---------------------------------------------------------
# CHART 1: SURVIVAL BY SEX
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))

sns.barplot(
    data=df,
    x="sex",
    y="survived"
)

plt.ylabel("Survival Rate")
plt.xlabel("Sex")
plt.title("Survival Rate by Sex")
plt.tight_layout()

plt.savefig("chart_1_survival_by_sex.png")

plt.show()


# ---------------------------------------------------------
# CHART 2: SURVIVAL BY PASSENGER CLASS
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))

sns.barplot(
    data=df,
    x="pclass",
    y="survived"
)

plt.ylabel("Survival Rate")
plt.xlabel("Passenger Class")
plt.title("Survival Rate by Passenger Class")
plt.tight_layout()

plt.savefig("chart_2_survival_by_class.png")

plt.show()


# ---------------------------------------------------------
# CHART 3: SURVIVAL BY SEX AND CLASS
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.ylabel("Survival Rate")
plt.xlabel("Passenger Class")
plt.title("Survival Rate by Sex and Passenger Class")
plt.tight_layout()

plt.savefig("chart_3_survival_by_sex_class.png")

plt.show()


# ---------------------------------------------------------
# CHART 4: AGE VS FARE BY SURVIVAL
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived"
)

plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Age vs Fare by Survival")
plt.tight_layout()

plt.savefig("chart_4_age_fare_survival.png")

plt.show()


print("\nFour multivariate charts created successfully.")
# ---------------------------------------------------------
# EXPLORATORY STANDARDIZATION CHECK
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("EXPLORATORY STANDARDIZATION CHECK")
print("=" * 60)


# Keep original values for comparison
age_mean_before = df["age"].mean()
age_std_before = df["age"].std()

fare_mean_before = df["fare"].mean()
fare_std_before = df["fare"].std()


# Standardize age and fare
df["age_zscore"] = (
    (df["age"] - age_mean_before)
    / age_std_before
)

df["fare_zscore"] = (
    (df["fare"] - fare_mean_before)
    / fare_std_before
)


# ---------------------------------------------------------
# BEFORE STANDARDIZATION
# ---------------------------------------------------------

print("\nBEFORE STANDARDIZATION")

print(
    f"Age   -> Mean: {age_mean_before:.4f}, "
    f"Std: {age_std_before:.4f}"
)

print(
    f"Fare  -> Mean: {fare_mean_before:.4f}, "
    f"Std: {fare_std_before:.4f}"
)


# ---------------------------------------------------------
# AFTER STANDARDIZATION
# ---------------------------------------------------------

print("\nAFTER STANDARDIZATION")

print(
    f"Age Z-score  -> Mean: "
    f"{df['age_zscore'].mean():.4f}, "
    f"Std: {df['age_zscore'].std():.4f}"
)

print(
    f"Fare Z-score -> Mean: "
    f"{df['fare_zscore'].mean():.4f}, "
    f"Std: {df['fare_zscore'].std():.4f}"
)


# ---------------------------------------------------------
# SAVE FINAL CLEANED DATASET
# ---------------------------------------------------------

df.to_csv("titanic.csv", index=False)

print("\nUpdated cleaned dataset saved as titanic.csv")