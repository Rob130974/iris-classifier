from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

import joblib
import os
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)
# Save the trained model
os.makedirs("outputs", exist_ok=True)
joblib.dump(model, "outputs/model.joblib")
# Make predictions
predictions = model.predict(X_test)

# Create and save confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, predictions)
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# Check accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"Model accuracy: {accuracy:.2f}")