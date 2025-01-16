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


def glm(X, y, *, verbose: bool = False):
    X = sm.add_constant(X)
    epsilon = 0.01
    z = epsilon + (1 - 2 * epsilon) * y

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    y_hat = np.zeros(y.shape)
    for train, test in kfold.split(X, y):
        model = sm.GLM(z[train], X[train], family=sm.families.Binomial()).fit()
        y_hat[test] = np.round(model.predict(X[test])).astype(bool)
    return y_hat


def rf(X, y, *, verbose: bool = False, **info):
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    y_hat = np.zeros(y.shape)
    for train, test in kfold.split(X, y):
        model = RandomForestClassifier(random_state=42).fit(X[train], y[train])
        y_hat[test] = np.round(model.predict(X[test])).astype(bool)
    return y_hat


def alpha_score(Yx, Px, alpha, beta, gamma):
    Mx = np.sum(Yx & ~Px, axis=1)
    Fx = np.sum(~Yx & Px, axis=1)
    YPx = np.sum(Yx | Px, axis=1)
    score = np.power(1 - (beta * Mx + gamma * Fx) / YPx, alpha)
    return score


def measures(Yx, Px, score, names):
    accuracy = np.mean(score)
    precision = np.sum(score[:, np.newaxis] * Px, axis=0) / np.sum(Px, axis=0)
    recall = np.sum(score[:, np.newaxis] * Yx, axis=0) / np.sum(Yx, axis=0)
    results = (
        {"accuracy -": accuracy}
        | {f"precision {name}": p for p, name in zip(precision, names)}
        | {f"recall {name}": r for r, name in zip(recall, names)}
    )
    return results


if __name__ == "__main__":
    X, _, y, labels = load_npy("scene_train.npy")
    y = y.astype(bool)
    X_root4 = X.reshape(-1, 2, 49)
    X_root4[:, 1] = np.sqrt(np.sqrt(X_root4[:, 1]))
    X_root4 = X_root4.reshape(X.shape)
    mapper2D = pacmap.PaCMAP(
        n_components=50, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=42
    )
    X_lowdim = mapper2D.fit_transform(X)
    y_glm = np.zeros(y.shape, dtype=bool)
    y_root4 = np.zeros(y.shape, dtype=bool)
    y_lowdim = np.zeros(y.shape, dtype=bool)
    y_rf = np.zeros(y.shape, dtype=bool)
    results = []
    for idx in range(len(labels)):
        y_label = y[:, idx]
        y_glm[:, idx] = glm(X, y_label)
        y_root4[:, idx] = glm(X_root4, y_label)
        y_lowdim[:, idx] = glm(X_lowdim, y_label)
        y_rf[:, idx] = rf(X, y_label)
    results = [
        measures(y, y_glm, alpha_score(y, y_root4, 1, 1, 0.25), labels),
        measures(y, y_root4, alpha_score(y, y_root4, 1, 1, 0.25), labels),
        measures(y, y_lowdim, alpha_score(y, y_root4, 1, 1, 0.25), labels),
        measures(y, y_rf, alpha_score(y, y_root4, 1, 1, 0.25), labels),
    ]
    df = pd.DataFrame(results)
    df.to_csv("baseline-multilabel.csv", index=False)
