import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load and clean the CSVs
dt = pd.read_csv("Training.csv").loc[:, ~pd.read_csv("Training.csv").columns.str.contains('^Unnamed')]
dts = pd.read_csv("Testing.csv").loc[:, ~pd.read_csv("Testing.csv").columns.str.contains('^Unnamed')]

# Split into features and target
train_features = dt.drop(columns=["prognosis"])
train_result = dt["prognosis"]

print(train_result)
test_features = dts.drop(columns=["prognosis"])
test_result = dts["prognosis"]

# Ensure test features have the same columns as training features
test_features = test_features[train_features.columns]

# Train the model
model = RandomForestClassifier()
model.fit(train_features, train_result)

# Save the trained model to a .pkl file
if not os.path.exists("model"):
    os.makedirs("model")  # Create the directory if it doesn't exist
joblib.dump(model, "model/diseasemodel.pkl")

# Make predictions
preds = model.predict(test_features)
print("Accuracy:", accuracy_score(test_result, preds))
