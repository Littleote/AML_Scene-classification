import numpy as np
import pacmap
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from statsmodels import api as sm

from dataset import load_npy


def f1_score(stats):
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
    return F1


def glm(X, y, *, verbose: bool = False, **info):
    X = sm.add_constant(X)
    epsilon = 0.01
    z = epsilon + (1 - 2 * epsilon) * y

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
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
    F1 = f1_score(stats)
    if verbose:
        print("Params:", info)
        print("\t=>", stats)
        print("\t=>", F1)
    result = info | stats | F1
    return result


def rf(X, y, *, verbose: bool = False, **info):
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    y_hat = np.zeros(y.shape)
    for train, test in kfold.split(X, y):
        model = RandomForestClassifier(random_state=42).fit(X[train], y[train])
        y_hat[test] = np.round(model.predict(X[test])).astype(int)
    stats = {
        "true_positive": np.count_nonzero((y == 1) & (y_hat == 1)),
        "false_positive": np.count_nonzero((y == 0) & (y_hat == 1)),
        "true_negative": np.count_nonzero((y == 0) & (y_hat == 0)),
        "false_negative": np.count_nonzero((y == 1) & (y_hat == 0)),
    }
    F1 = f1_score(stats)
    if verbose:
        print("Params:", info)
        print("\t=>", stats)
        print("\t=>", F1)
    result = info | stats | F1
    return result


if __name__ == "__main__":
    X, _, y, labels = load_npy("scene_train.npy")
    X_root4 = X.reshape(-1, 2, 49)
    X_root4[:, 1] = np.sqrt(np.sqrt(X_root4[:, 1]))
    X_root4 = X_root4.reshape(X.shape)
    mapper2D = pacmap.PaCMAP(
        n_components=50, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=42
    )
    X_lowdim = mapper2D.fit_transform(X)
    results = []
    for label in labels:
        y_label = y[:, labels == label].flatten()
        results.append(
            glm(
                X,
                y_label,
                target=label,
                transformation="None",
                model="GLM",
                verbose=True,
            )
        )
        results.append(
            glm(
                X_root4,
                y_label,
                target=label,
                transformation="Root 4",
                model="GLM",
                verbose=True,
            )
        )
        results.append(
            glm(
                X_lowdim,
                y_label,
                target=label,
                transformation="PaCMAP",
                model="GLM",
                verbose=True,
            )
        )
        results.append(
            rf(
                X,
                y_label,
                target=label,
                transformation="None",
                model="Random Forest",
                verbose=True,
            )
        )
    df = pd.DataFrame(results)
    df.to_csv("basline.csv", index=False)
