import numpy as np
import pacmap
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss
from sklearn.metrics import label_ranking_average_precision_score as lrap
from sklearn.model_selection import KFold
from statsmodels import api as sm

from dataset import load_npy


def glm(X, y, *, verbose: bool = False):
    X = sm.add_constant(X)
    epsilon = 0.01
    z = epsilon + (1 - 2 * epsilon) * y

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    p_hat = np.zeros(y.shape, dtype=float)
    y_hat = np.zeros(y.shape, dtype=bool)
    for train, test in kfold.split(X, y):
        model = sm.GLM(z[train], X[train], family=sm.families.Binomial()).fit()
        p_hat[test] = model.predict(X[test])
        y_hat[test] = np.round(p_hat[test]).astype(bool)
    return y_hat, p_hat


def rf(X, y, *, verbose: bool = False, **info):
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    p_hat = np.zeros(y.shape, dtype=float)
    y_hat = np.zeros(y.shape, dtype=bool)
    for train, test in kfold.split(X, y):
        model = RandomForestClassifier(random_state=42).fit(X[train], y[train])
        p_hat[test] = model.predict_proba(X[test])[:, 1]
        y_hat[test] = model.predict(X[test]).astype(bool)
    return y_hat, p_hat


def alpha_score(Yx, Px, alpha, beta, gamma):
    Mx = np.sum(Yx & ~Px, axis=1)
    Fx = np.sum(~Yx & Px, axis=1)
    YPx = np.sum(Yx | Px, axis=1)
    score = np.power(1 - (beta * Mx + gamma * Fx) / YPx, alpha)
    return score


def measures(Yx, Px, prob, names, **alphas):
    score = alpha_score(Yx, Px, **alphas)
    accuracy = np.mean(score)
    precision = np.sum(score[:, np.newaxis] * Px, axis=0) / np.sum(Px, axis=0)
    recall = np.sum(score[:, np.newaxis] * Yx, axis=0) / np.sum(Yx, axis=0)
    results = (
        {"accuracy -": accuracy}
        | {f"precision {name}": p for p, name in zip(precision, names)}
        | {f"recall {name}": r for r, name in zip(recall, names)}
        | {"Hamming-Loss -": hamming_loss(Yx, Px)}
        | {"LRAP -": lrap(Yx, prob)}
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
    p_glm = np.zeros(y.shape, dtype=float)
    p_root4 = np.zeros(y.shape, dtype=float)
    p_lowdim = np.zeros(y.shape, dtype=float)
    p_rf = np.zeros(y.shape, dtype=float)
    results = []
    for idx in range(len(labels)):
        y_label = y[:, idx]
        y_glm[:, idx], p_glm[:, idx] = glm(X, y_label)
        y_root4[:, idx], p_root4[:, idx] = glm(X_root4, y_label)
        y_lowdim[:, idx], p_lowdim[:, idx] = glm(X_lowdim, y_label)
        y_rf[:, idx], p_rf[:, idx] = rf(X, y_label)
    params = dict(alpha=1, beta=1, gamma=0.25)
    results = [
        measures(y, y_glm, p_glm, labels, **params),
        measures(y, y_root4, p_root4, labels, **params),
        measures(y, y_lowdim, p_lowdim, labels, **params),
        measures(y, y_rf, p_rf, labels, **params),
    ]
    df = pd.DataFrame(results)
    df.to_csv("baseline-multilabel.csv", index=False)
