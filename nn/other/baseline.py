from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import pandas as pd
from scipy.io import arff
import numpy as np


def load(file: str):
    contents = arff.loadarff(file)
    df = pd.DataFrame(contents[0])
    return df


def process(df: pd.DataFrame):
    attr = [col for col in df.columns if col.startswith("attr")]
    classes = ['Beach', 'Sunset', 'FallFoliage', 'Field', 'Mountain', 'Urban']

    data = df[attr].to_numpy(dtype=np.float32)
    targets = df[classes].apply(lambda s: s.map(int)).to_numpy(dtype=np.float32)

    return data, targets


if __name__ == "__main__":
    # Load and process data
    df = load("../scene.arff")
    X, y = process(df)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression for each label
    n_classes = y.shape[1]
    predictions = []
    print("Baseline Logistic Regression Results:")
    for i in range(n_classes):
        # Train logistic regression for each class
        model = LogisticRegression(max_iter=500, solver='liblinear')
        model.fit(X_train, y_train[:, i])

        # Predict probabilities for ROC AUC and binary predictions for F1 score
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred_binary = (y_pred_prob > 0.5).astype(int)

        # Append predictions for overall metrics
        predictions.append(y_pred_prob)

        # Print per-class metrics
        print(f"\nClass: {['Beach', 'Sunset', 'FallFoliage', 'Field', 'Mountain', 'Urban'][i]}")
        print(classification_report(y_test[:, i], y_pred_binary))
        print(f"ROC AUC: {roc_auc_score(y_test[:, i], y_pred_prob):.3f}")

    # Calculate overall metrics
    y_pred = np.stack(predictions, axis=1)
    y_pred_binary = (y_pred > 0.5).astype(int)

    print("\nOverall Metrics:")
    overall_f1 = f1_score(y_test, y_pred_binary, average="macro")
    overall_roc_auc = roc_auc_score(y_test, y_pred, average="macro")
    print(f"Macro F1 Score: {overall_f1:.3f}")
    print(f"Macro ROC AUC Score: {overall_roc_auc:.3f}")
