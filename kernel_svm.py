import itertools
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np

from dataset import load_npy


def run(X, y, kfold, args):
    model = Pipeline(
        steps=[
            ("scaling", StandardScaler()),
            ("SVM", SVC(**args, max_iter=10_000)),
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
    F1 = {
        "positive_F1": 2
        * stats["true_positive"]
        / (
            2 * stats["true_positive"]
            + stats["false_positive"]
            + stats["false_negative"]
        ),
        "negaitve_F1": 2
        * stats["true_negative"]
        / (
            2 * stats["true_negative"]
            + stats["false_positive"]
            + stats["false_negative"]
        ),
    }
    F1["mean_F1"] = (F1["positive_F1"] + F1["negaitve_F1"]) / 2
    return stats | F1


def grid_search(X, y, *, verbose: bool = False):
    blank = dict(
        degree=None,
        gamma=None,
        coef0=None,
    )
    n = X.shape[1]

    Cs = [10**i for i in range(-2, 2 + 1)]
    degrees = [2, 3, 4, 5]
    gammas = [np.power(n, i / 2) for i in range(-4, 0 + 1)]
    coef0 = [0, 1]
    kernels = {
        "linear": {},
        "poly": {"degree": degrees, "gamma": gammas, "coef0": coef0},
        "rbf": {"gamma": gammas},
    }

    kfold = KFold(n_splits=10, shuffle=True)
    results = []
    for C in Cs:
        for kernel, options in kernels.items():
            common = dict(C=C, class_weight="balanced", kernel=kernel)
            for values in itertools.product(*options.values()):
                specific = {name: value for name, value in zip(options.keys(), values)}
                params = common | specific
                stats = run(X, y, kfold, params)
                if verbose:
                    print("Params:", params)
                    print("\t=>", stats)
                results.append(blank | params | stats)
    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    X, _, y, labels = load_npy("scene_train.npy")
    y = y[:, labels == "Urban"].flatten()
    df = grid_search(X, y, verbose=True)
    df.to_csv("kernel_svm.csv", index=False)
