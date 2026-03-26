import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import os

# -------------------
# Step 1: Generate data
# -------------------
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=200, n_features=5, n_classes=2, random_state=42)

data = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
data["target"] = y

X = data.drop("target", axis=1)
y = data["target"]

# -------------------
# Step 2: Split
# -------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------
# Step 3: Train model
# -------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

# -------------------
# Step 4: MLflow Logging (Safe)
# -------------------
try:
    mlflow.set_tracking_uri("http://localhost:5000")  # عدلي لو عندك server حقيقي
    mlflow.set_experiment("ML_Pipeline_Assignment")
    run_id = "run_123"
    with mlflow.start_run(run_name=run_id):
        mlflow.log_param("model", "RandomForest")
        mlflow.log_metric("accuracy", accuracy)
except Exception as e:
    print("⚠️ MLflow logging skipped:", e)
    run_id = "run_123"

# -------------------
# Step 5: Save files
# -------------------
with open("accuracy.txt", "w") as f:
    f.write(str(accuracy))

with open("model_info.txt", "w") as f:
    f.write(run_id)

print("Accuracy:", accuracy)
print("Run ID:", run_id)
