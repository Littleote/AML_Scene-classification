import numpy as np
import json
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dataset import load_npy
from kernel_multi_svm import measures


def svm(X, y, **args):
    model = Pipeline(
        steps=[
            ("scaling", StandardScaler()),
            ("SVM", SVC(**args, max_iter=10_000)),
        ]
    )

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    y_hat = np.zeros(y.shape, dtype=bool)
    for train, test in kfold.split(X, y):
        model.fit(X[train], y[train])
        y_hat[test] = np.round(model.predict(X[test])).astype(bool)
    return y_hat


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
        {
            "class_weight": "balanced",
            "kernel": "poly",
            "C": 0.1,
            "gamma": 1 / n,
            "coef0": 1,
            "degree": 5,
        },
    ]
    y_hat = np.zeros(y.shape, dtype=bool)
    for idx, conf in enumerate([0, 0, 1, 1, 0, 1]):
        y_hat[:, idx] = svm(X_root4, y[:, idx], **configs[conf])
    out = measures(y, y_hat, labels, **params)
    with open("mixed_svm.json", mode="w") as file:
        json.dump(out, file)
