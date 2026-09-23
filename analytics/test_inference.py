import joblib
import pandas as pd

# 1. Load the serialized end-to-end pipeline
pipeline = joblib.load("analytics/models/titanic_best_pipeline.joblib")

# 2. Provide raw, unpreprocessed data (with a missing age)
raw_passenger = pd.DataFrame([{
    "pclass": 3,
    "sex": "male",
    "age": None,          # Handled dynamically by median imputer
    "sibsp": 0,
    "parch": 0,
    "fare": 7.89,
    "embarked": "S"
}])

# 3. Predict class outcome (0 = Did not survive, 1 = Survived)
prediction = pipeline.predict(raw_passenger)

# 4. Predict survival probability
prob = pipeline.predict_proba(raw_passenger)[:, 1]

# 5. Display the result
print(f"Survival Prediction: {prediction[0]} | Survival Probability: {prob[0]:.2%}")