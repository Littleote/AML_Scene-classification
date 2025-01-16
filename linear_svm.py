import numpy as np
import pacmap
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from dataset import load_npy


def grid_search(X, y, *, verbose: bool = False, **info):
    Cs = [10**i for i in range(-2, 2 + 1)]
    penalties = ["l1", "l2"]
    losses = ["hinge", "squared_hinge"]

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
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
                    print("Params:", info | params)
                    print("\t=>", stats)
                    print("\t=>", F1)
                results.append(info | params | stats | F1)
    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    X, _, y, labels = load_npy("scene_train.npy")
    X_root4 = X.reshape(-1, 2, 49)
    X_root4[:, 1] = np.sqrt(np.sqrt(X_root4[:, 1]))
    X_root4 = X_root4.reshape(X.shape)
    mapper2D = pacmap.PaCMAP(
        n_components=50, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=42
    )
    X_lowdim = mapper2D.fit_transform(X)
    y = y[:, labels == "Urban"].flatten()
    df_1 = grid_search(X, y, verbose=True, transformation="none")
    df_2 = grid_search(X_root4, y, verbose=True, transformation="root 4")
    df_3 = grid_search(X_lowdim, y, verbose=True, transformation="PaCMAP")
    df = pd.concat([df_1, df_2, df_3])
    df.to_csv("linear_svm.csv", index=False)
