from sklearn import datasets  # Load the built-in Iris dataset from scikit-learn
from sklearn.model_selection import train_test_split  # Helper function to split dataset into training and testing sets
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay  # Evaluation metrics to test how well our model performed
import matplotlib.pyplot as plot  # Plotting library to display charts and visual graphs
import numpy as np
from collections import Counter

def Euclidian_Distance (X1, X2):
    return np.sqrt(np.sum((X1 - X2) ** 2))

def KNN_Predict(X_train, Y_train, x_new, k =4):
    distances = [Euclidian_Distance(x_new, x_row) for x_row in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [Y_train[i] for i in k_indices]

    most_common_label = Counter(k_nearest_labels).most_common(1)
    return most_common_label[0][0]

iris = datasets.load_iris()  # Load the classic Iris flower dataset
X = iris.data  # Extract feature matrix 'X' (sepal length, sepal width, petal length, petal width)
Y = iris.target  # Extract target vector 'Y' (the species class: 0=Setosa, 1=Versicolor, 2=Virginica)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.3, random_state = None)
predictions = [KNN_Predict(X_Train, Y_Train, x_new, k=4) for x_new in X_Test]
accuracy = np.mean(np.array(predictions) == Y_Test)

print(f"Predictions: {predictions}")
print(f"Actual:      {list(Y_Test)}")
print(f"Accuracy: {accuracy * 100:.2f}%")

ConfusionMatrix = confusion_matrix(Y_Test, predictions)
print("Confusion Matrix: ")
print(ConfusionMatrix)

print("Classification Report: ")
print(classification_report(Y_Test, predictions, target_names=iris.target_names))

display = ConfusionMatrixDisplay(confusion_matrix=ConfusionMatrix, display_labels=iris.target_names)  # Create a visual heat-map representation of the confusion matrix using the class names
display.plot(cmap=plot.cm.Blues)  # Plot using a blue color gradient map

plot.title("Iris Dataset: Confusion Matrix")  # Set the title of the matplotlib window/figure
plot.show()  # Render and display the visual plot on screen

