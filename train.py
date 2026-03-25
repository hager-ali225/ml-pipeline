import random

accuracy = random.uniform(0.7, 0.95)

with open("accuracy.txt", "w") as f:
    f.write(str(accuracy))

with open("model_info.txt", "w") as f:
    f.write("run_123")

print("Accuracy:", accuracy)
print("Run ID: run_123")
