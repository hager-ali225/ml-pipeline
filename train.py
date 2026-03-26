import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow

# Load real dataset
data = pd.read_csv("data.csv")  # حطي مسار الداتا عندك
X = data.drop("target", axis=1)
y = data["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

accuracy = accuracy_score(y_test, preds)

# Save accuracy
with open("accuracy.txt", "w") as f:
    f.write(str(accuracy))

# Save model info
run_id = "run_123"
with open("model_info.txt", "w") as f:
    f.write(run_id)

print("Accuracy:", accuracy)
print("Run ID:", run_id)
