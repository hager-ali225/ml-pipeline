THRESHOLD = 0.95

with open("accuracy.txt", "r") as f:
    accuracy = float(f.read())

print("Accuracy:", accuracy)

if accuracy < THRESHOLD:
    raise Exception("Model failed ❌")
else:
    print("Model passed ✅")
