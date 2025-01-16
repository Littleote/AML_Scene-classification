import numpy as np
import pandas as pd
from sklearn.metrics import hamming_loss
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dataset import load_npy


def svm(X, y, **args):
    model = Pipeline(
        steps=[
            ("scaling", StandardScaler()),
            ("SVM", MultiOutputClassifier(SVC(**args, max_iter=10_000))),
        ]
    )

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    y_hat = np.zeros(y.shape, dtype=bool)
    for train, test in kfold.split(X, y):
        model.fit(X[train], y[train])
        y_hat[test] = np.round(model.predict(X[test])).astype(bool)
    return y_hat


def alpha_score(Yx, Px, alpha, beta, gamma):
    Mx = np.sum(Yx & ~Px, axis=1)
    Fx = np.sum(~Yx & Px, axis=1)
    YPx = np.sum(Yx | Px, axis=1)
    score = np.power(1 - (beta * Mx + gamma * Fx) / YPx, alpha)
    return score


def measures(Yx, Px, names, **alphas):
    score = alpha_score(Yx, Px, **alphas)
    accuracy = np.mean(score)
    precision = np.sum(score[:, np.newaxis] * Px, axis=0) / np.maximum(
        1, np.sum(Px, axis=0)
    )
    recall = np.sum(score[:, np.newaxis] * Yx, axis=0) / np.sum(Yx, axis=0)
    results = (
        {"accuracy -": accuracy}
        | {f"precision {name}": p for p, name in zip(precision, names)}
        | {f"recall {name}": r for r, name in zip(recall, names)}
        | {"Hamming-Loss -": hamming_loss(Yx, Px)}
    )
    return results


if __name__ == "__main__":
    X, _, y, labels = load_npy("scene_train.npy")
    y = y.astype(bool)
    n = X.shape[1]
    params = dict(alpha=1, beta=1, gamma=0.25)
    X_root4 = X.reshape(-1, 2, 49)
    X_root4[:, 1] = np.sqrt(np.sqrt(X_root4[:, 1]))
    X_root4 = X_root4.reshape(X.shape)
    results = []
    configs = [
        {"class_weight": "balanced", "kernel": "rbf", "C": 1, "gamma": 1 / n},
        {"class_weight": "balanced", "kernel": "rbf", "C": 10, "gamma": 1 / n},
        {"class_weight": "balanced", "kernel": "rbf", "C": 100, "gamma": 1 / n},
        {
            "class_weight": "balanced",
            "kernel": "poly",
            "C": 0.1,
            "gamma": 1 / n,
            "coef0": 1,
            "degree": 5,
        },
        {
            "class_weight": "balanced",
            "kernel": "poly",
            "C": 1,
            "gamma": 1 / n,
            "coef0": 1,
            "degree": 3,
        },
        {
            "class_weight": "balanced",
            "kernel": "poly",
            "C": 0.01,
            "gamma": 1 / np.sqrt(n),
            "coef0": 1,
            "degree": 2,
        },
    ]
    for config in configs:
        y_hat = svm(X_root4, y, **config)
        results.append(config | measures(y, y_hat, labels, **params))
    df = pd.DataFrame(results)
    df.to_csv("multi_svm.csv", index=False)
