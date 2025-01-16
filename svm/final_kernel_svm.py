import numpy as np
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC

from dataset import load_npy
from kernel_multi_svm import measures


def svm(X, y, X_test, y_test, **args):
    model = Pipeline(
        steps=[
            ("scaling", StandardScaler()),
            ("SVM", MultiOutputClassifier(SVC(**args, max_iter=10_000))),
        ]
    )

    y_hat = np.zeros(y_test.shape, dtype=bool)
    model.fit(X, y)
    y_hat = np.round(model.predict(X_test)).astype(bool)
    return y_hat


if __name__ == "__main__":
    X, _, y, labels = load_npy("scene_train.npy")
    X_test, _, y_test, _ = load_npy("scene_test.npy")
    y = y.astype(bool)
    y_test = y_test.astype(bool)
    n = X.shape[1]
    params = dict(alpha=1, beta=1, gamma=0.25)
    X_root4 = X.reshape(-1, 2, 49)
    X_root4[:, 1] = np.sqrt(np.sqrt(X_root4[:, 1]))
    X = X_root4.reshape(X.shape)
    X_root4 = X_test.reshape(-1, 2, 49)
    X_root4[:, 1] = np.sqrt(np.sqrt(X_root4[:, 1]))
    X_test = X_root4.reshape(X_test.shape)
    results = []
    config = {"class_weight": "balanced", "kernel": "rbf", "C": 1, "gamma": 1 / n}
    y_hat = svm(X, y, X_test, y_test, **config)
    out = measures(y_test, y_hat, labels, **params)
    with open("final_svm.json", mode="w") as file:
        json.dump(out, file)
