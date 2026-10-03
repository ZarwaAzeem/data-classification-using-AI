# Data Classification Using AI 🌸

A basic supervised learning project that classifies iris flowers into three species using the K-Nearest Neighbors (KNN) algorithm.

## Goal

Build a basic classification model using a small dataset.

## What It Does

1. **Loads** the Iris dataset (150 flowers, 4 measurements each, 3 species) into a pandas DataFrame
2. **Explores** the data: shape, first rows, class counts, and summary statistics
3. **Splits** the data into 80% training and 20% testing sets
4. **Trains** a K-Nearest Neighbors classifier (k = 5)
5. **Evaluates** the model with accuracy, a classification report, and a confusion matrix
6. **Predicts** the species of a new, unseen flower

## Skills Demonstrated

- Data handling
- Supervised learning basics
- Model training and evaluation

## Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib

Install everything with:

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python classifier.py
```

The script prints the results to the terminal and saves a `confusion_matrix.png` image in the same folder.

## Sample Output

```
=== Train/Test Split ===
Training samples: 120
Testing samples:  30

=== Evaluation ===
Accuracy: 100.00%

              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00        10
   virginica       1.00      1.00      1.00        10

=== New Prediction ===
Sample measurements: [5.1, 3.5, 1.4, 0.2]
Predicted species: setosa
```

Iris is a small, clean dataset, so very high accuracy is expected. Results can vary slightly with a different split or `random_state`.

## How It Works

KNN classifies a new flower by finding the 5 training flowers with the most similar measurements and choosing the species that appears most often among them. The test set is kept separate during training, so the accuracy reflects performance on data the model has never seen.

## Project Structure

```
data-classification/
├── classifier.py
├── requirements.txt
└── README.md
```

## Future Improvements

- Compare KNN with a Decision Tree or Logistic Regression
- Try different values of K and plot accuracy against K
- Use cross-validation for a more reliable accuracy estimate
- Try a larger or real-world dataset

## License

This project is open source and available for learning purposes.
