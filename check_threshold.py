THRESHOLD = 0.85

# Simulated accuracy (لأن مفيش MLflow حقيقي)
import random
accuracy = random.uniform(0.7, 0.95)

print("Accuracy:", accuracy)

if accuracy < THRESHOLD:
    raise Exception("Model failed ❌")
else:
    print("Model passed ✅")
