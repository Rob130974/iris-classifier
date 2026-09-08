from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Create the visualisation
plt.figure(figsize=(16, 10))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Iris Decision Tree Classifier")
plt.tight_layout()

# Save the diagram
plt.savefig("outputs/decision_tree.png", dpi=200)

plt.show()
