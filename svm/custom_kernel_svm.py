from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np

from dataset import load_npy


def mu_var(X):
    aux = X.reshape(-1, 2, 49)
    return aux[:, 0].reshape(-1, 147), aux[:, 1].reshape(-1, 147)


def kld_kernel(mu_0, var_0, mu_1, var_1, *, gamma, scale):
    """
    Kernel implementing as distance the Kullback-Leibler divergence
    """
    sigma_0 = scale * np.sqrt(var_0) + 1e-12
    sigma_1 = scale * np.sqrt(var_1) + 1e-12
    dim_0 = mu_0.shape[0]
    dim_1 = mu_1.shape[0]
    n = mu_0.shape[1]
    # If it occupies less than 1 Gb do a direct calculation
    if dim_0 * dim_1 * n * 8 < 1e9:
        Mu_0 = np.repeat(mu_0[:, np.newaxis], dim_1, 1)
        Mu_1 = np.repeat(mu_1[np.newaxis], dim_0, 0)
        Sigma_0 = np.repeat(sigma_0[:, np.newaxis], dim_1, 1)
        Sigma_1 = np.repeat(sigma_1[np.newaxis], dim_0, 0)
        trace = np.sum(Sigma_0 / Sigma_1 + Sigma_1 / Sigma_0, -1)
        inner = np.sum(np.square(Mu_1 - Mu_0) * (1 / Sigma_1 + 1 / Sigma_0), -1)
    # If it occupies more than 1 Gb do element wise
    else:
        mu_1 = mu_1.transpose()
        sigma_1 = sigma_1.transpose()
        trace = np.zeros((dim_0, dim_1))
        inner = np.zeros((dim_0, dim_1))
        for i in range(n):
            trace += (
                sigma_0[:, [i]] / sigma_1[[i], :] + sigma_1[[i], :] / sigma_0[:, [i]]
            )
            inner += np.square(mu_1[[i], :] - mu_0[:, [i]]) * (
                1 / sigma_1[[i], :] + 1 / sigma_0[:, [i]]
            )
    divergence = 1 / 2 * (trace + inner) - n
    return np.exp(-gamma * divergence)


def run(X, y, kfold, kernel_args, svm_args):
    model = Pipeline(
        steps=[
            ("SVM", SVC(**svm_args, max_iter=10_000)),
        ]
    )
    y_hat = np.zeros(y.shape)
    mu, var = mu_var(X)
    K = kld_kernel(mu, var, mu, var, **kernel_args)

    for train, test in kfold.split(X, y):
        model.fit(K[train][:, train], y[train])
        y_hat[test] = model.predict(K[test][:, train])
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
    n = X.shape[1]

    Cs = [10**i for i in range(-1, 3 + 1)]
    scales = [10**i for i in range(0, 5)]
    gammas = [np.power(n, i / 2) for i in range(-4, 0 + 1)]

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    results = []
    for C in Cs:
        for scale in scales:
            for gamma in gammas:
                svm = {"C": C, "kernel": "precomputed", "class_weight": "balanced"}
                kernel = {"scale": scale, "gamma": gamma}
                stats = run(X, y, kfold, kernel_args=kernel, svm_args=svm)
                if verbose:
                    print("Params:", kernel | svm)
                    print("\t=>", stats)
                results.append(kernel | svm | stats)
    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    X, _, y, labels = load_npy("scene_train.npy")
    y = y[:, labels == "Urban"].flatten()
    df = grid_search(X, y, verbose=True)
    df.to_csv("custom_kernel_svm.csv", index=False)
