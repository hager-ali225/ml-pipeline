import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import os

# -------------------
# Step 1: Load or generate data
# -------------------
if os.path.exists("data.csv"):
    data = pd.read_csv("data.csv")
else:
    # Generate dummy data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    data = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    data["target"] = y

X = data.drop("target", axis=1)
y = data["target"]

# -------------------
# Step 2: Split
# -------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------
# Step 3: Train
# -------------------
model = LogisticRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

# -------------------
# Step 4: MLflow Logging
# -------------------
mlflow.set_tracking_uri("http://localhost:5000")  # عدلي URI لو عندك MLflow server
mlflow.set_experiment("ML_Pipeline_Assignment")
run_id = "run_123"

with mlflow.start_run(run_name=run_id):
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)

# -------------------
# Step 5: Save files
# -------------------
with open("accuracy.txt", "w") as f:
    f.write(str(accuracy))

with open("model_info.txt", "w") as f:
    f.write(run_id)

print("Accuracy:", accuracy)
print("Run ID:", run_id)
