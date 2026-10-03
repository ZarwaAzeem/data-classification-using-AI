import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)


def load_data():
    """Step 1: Load the Iris dataset into a pandas DataFrame."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return iris, df


def explore_data(df):
    """Step 2: Understand the dataset."""
    print("=== Dataset Overview ===")
    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nClass counts:")
    print(df["species"].value_counts())
    print("\nSummary statistics:")
    print(df.describe())


def main():
    iris, df = load_data()
    explore_data(df)

    # Step 3: Split data into training (80%) and testing (20%) sets
    X = df[iris.feature_names]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("\n=== Train/Test Split ===")
    print("Training samples:", len(X_train))
    print("Testing samples: ", len(X_test))

    # Step 4: Train a K-Nearest Neighbors classifier
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # Step 5: Evaluate the model on unseen test data
    y_pred = model.predict(X_test)
    print("\n=== Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Save a confusion matrix image
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=iris.target_names
    )
    plt.title("Confusion Matrix (KNN, k=5)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Saved confusion matrix to confusion_matrix.png")

    # Step 6: Predict a new, unseen flower
    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=iris.feature_names)
    prediction = model.predict(sample)[0]
    print("\n=== New Prediction ===")
    print("Sample measurements:", sample.values.tolist()[0])
    print("Predicted species:", iris.target_names[prediction])


if __name__ == "__main__":
    main()
