from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np

from dataset import load_npy


def grid_search(X, y, *, verbose: bool = False):
    Cs = [10**i for i in range(-2, 2 + 1)]
    penalties = ["l1", "l2"]
    losses = ["hinge", "squared_hinge"]

    kfold = KFold(n_splits=10, shuffle=True)
    results = []
    for C in Cs:
        for penalty in penalties:
            for loss in losses:
                if penalty == "l1" and loss == "hinge":
                    continue
                params = dict(penalty=penalty, loss=loss, C=C, class_weight="balanced")
                model = Pipeline(
                    steps=[
                        ("scaling", StandardScaler()),
                        ("SVM", LinearSVC(**params, max_iter=10_000)),
                    ]
                )
                y_hat = np.zeros(y.shape)
                for train, test in kfold.split(X, y):
                    model.fit(X[train], y[train])
                    y_hat[test] = model.predict(X[test])
                stats = {
                    "true_positive": np.count_nonzero((y == 1) & (y_hat == 1)),
                    "false_positive": np.count_nonzero((y == 0) & (y_hat == 1)),
                    "true_negative": np.count_nonzero((y == 0) & (y_hat == 0)),
                    "false_negative": np.count_nonzero((y == 1) & (y_hat == 0)),
                }
                if verbose:
                    print("Params:", params)
                    print("\t=>", stats)
                results.append(params | stats)
    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    X, _, y, labels = load_npy("scene_train.npy")
    y = y[:, labels == "Urban"].flatten()
    df = grid_search(X, y, verbose=True)
    df.to_csv("linear_svm.csv", index=False)
