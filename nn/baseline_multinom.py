from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from scipy.io import arff
import pandas as pd
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

    # Create and train logistic regression model for multi-label classification
    base_model = LogisticRegression(max_iter=500, solver='liblinear')  # Logistic Regression
    multi_label_model = MultiOutputClassifier(base_model)  # Wrap for multi-label
    multi_label_model.fit(X_train, y_train)

    # Predict probabilities and binary outcomes
    y_pred_prob = multi_label_model.predict_proba(X_test)  # Probabilities for ROC AUC
    y_pred_binary = multi_label_model.predict(X_test)      # Binary predictions for F1 score

    # Reshape probabilities into a 2D array
    y_pred_prob = np.stack([prob[:, 1] for prob in y_pred_prob], axis=1)

    # Evaluate each class
    print("Multi-label Logistic Regression Results:")
    class_names = ['Beach', 'Sunset', 'FallFoliage', 'Field', 'Mountain', 'Urban']
    for i, class_name in enumerate(class_names):
        print(f"\nClass: {class_name}")
        print(classification_report(y_test[:, i], y_pred_binary[:, i]))
        print(f"ROC AUC: {roc_auc_score(y_test[:, i], y_pred_prob[:, i]):.3f}")

    # Calculate overall metrics
    overall_f1 = f1_score(y_test, y_pred_binary, average="macro")
    overall_roc_auc = roc_auc_score(y_test, y_pred_prob, average="macro")
    print("\nOverall Metrics:")
    print(f"Macro F1 Score: {overall_f1:.3f}")
    print(f"Macro ROC AUC Score: {overall_roc_auc:.3f}")
