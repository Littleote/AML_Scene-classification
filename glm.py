from sklearn.model_selection import KFold
from statsmodels import api as sm

import pandas as pd
import numpy as np

from dataset import load_npy


def glm(X, y, target: str, *, verbose: bool = False):
    X = sm.add_constant(X)
    epsilon = 0.01
    z = epsilon + (1 - 2 * epsilon) * y

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    params = dict(target=target)
    y_hat = np.zeros(y.shape)
    for train, test in kfold.split(X, y):
        model = sm.GLM(z[train], X[train], family=sm.families.Binomial()).fit()
        y_hat[test] = np.round(model.predict(X[test])).astype(int)
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
    if verbose:
        print("Params:", params)
        print("\t=>", stats)
        print("\t=>", F1)
    result = params | stats | F1
    return result


if __name__ == "__main__":
    X, _, y, labels = load_npy("scene_train.npy")
    results = []
    for label in labels:
        y_label = y[:, labels == label].flatten()
        results.append(glm(X, y_label, label, verbose=True))
    df = pd.DataFrame(results)
    df.to_csv("glm.csv", index=False)
