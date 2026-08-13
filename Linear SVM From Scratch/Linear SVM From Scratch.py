import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class StandardScalerFromScratch:
    def __init__(self):
        self.Mean_ = None
        self.Scale_ = None

    def fit(self, X):
        """Calculate mean and standard deviation along each feature column."""
        X = np.asarray(X, dtype=np.float64)
        self.Mean_ = np.mean(X, axis=0)
        self.Scale_ = np.std(X, axis=0, ddof=0)

        # Prevent division by zero if standard deviation is 0
        self.Scale_ = np.where(self.Scale_ == 0, 1.0, self.Scale_)
        return self

    def transform(self, X):
        """Apply z-score normalization to X using stored mean and standard deviation."""
        if self.Mean_ is None or self.Scale_ is None:
            raise RuntimeError("This StandardScalerFromScratch instance is not fitted yet. Call 'fit' first.")

        X = np.asarray(X, dtype=np.float64)
        return (X - self.Mean_) / self.Scale_

    def fit_transform(self, X):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)

class LinearSVMFromScratch:
    def __init__(self, LearningRate=0.001, LambaParameter=0.01, NumberOfIterations=1000):
        self.LearningRate = LearningRate
        self.LambaParameter = LambaParameter
        self.NumberOfIterations = NumberOfIterations
        self.Weights = None
        self.Bias = None

    def fit(self, X, Y):
        NumberOfSamples, NumberOfFeatures = X.shape

        Y_ = np.where(Y <= 0, -1, 1)

        self.Weights = np.zeros(NumberOfFeatures)
        self.Bias = 0

        for _ in range (self.NumberOfIterations):
            for Index, XInIndex in enumerate(X):
                condition = Y_[Index] * (np.dot(XInIndex, self.Weights) - self.Bias) >= 1

                if condition:
                    self.Weights -= self.LearningRate * (2 * self.LambaParameter * self.Weights)
                else:
                    self.Weights -= self.LearningRate * (2 * self.LambaParameter * self.Weights - XInIndex * Y_[Index])
                    self.Bias -= self.LearningRate * Y_[Index]

    def predict(self, X):
        approx = np.dot(X, self.Weights) - self.Bias
        return np.sign(approx)

CancerDataSet = load_breast_cancer()

X = CancerDataSet.data
Y = CancerDataSet.target

X_Train, X_Test, Y_Train, Y_Test = train_test_split (
    X, Y, test_size=0.2, random_state=None
)

scaler = StandardScalerFromScratch()

X_Train_Scaled = scaler.fit_transform(X_Train)
X_Test_Scaled = scaler.transform(X_Test)

model = LinearSVMFromScratch(LearningRate=0.001, LambaParameter=0.01, NumberOfIterations=1000)
model.fit(X_Train_Scaled, Y_Train)

RawPredictions = model.predict(X_Test_Scaled)
FinalPredictions = np.where(RawPredictions <= 0, 0, 1)
Accuracy = np.sum(Y_Test == FinalPredictions) / len(Y_Test)

print(f"Custom SVM Performance on Tumor Data:")
print(f"Trained Weights Shape: {model.Weights.shape}")
print(f"Final Model Bias:      {model.Bias:.4f}")
print(f"Test Classification Accuracy: {Accuracy * 100:.2f}%")
print("Classification Report: ")
print(classification_report(Y_Test, FinalPredictions, target_names=CancerDataSet.target_names))
ConfusionMatrix = confusion_matrix(Y_Test, FinalPredictions)
disp = ConfusionMatrixDisplay(confusion_matrix=ConfusionMatrix, display_labels=CancerDataSet.target_names)
disp.plot(cmap=plt.cm.Blues)  #
plt.title("Cancer Dataset: Confusion Matrix")
plt.show()
