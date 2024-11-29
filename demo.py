from scipy.io import arff
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay


def load(file: str):
    contents = arff.loadarff(file)
    df = pd.DataFrame(contents[0])
    return df


def process(df: pd.DataFrame):
    attr = [col for col in df.columns if col.startswith("attr")]
    kind = [col for col in df.columns if not col.startswith("attr")]
    data = df[attr].to_numpy()
    img = data.reshape((-1, 6, 7, 7))
    class_ = df[kind].apply(lambda s: s.map(int)).to_numpy()

    # displayed = np.random.randint(0, len(df))
    displayed = 1239
    fig, axs = plt.subplots(2, 3)
    row_name = ["mean {}", "{} sd"]
    for row in [0, 1]:
        col_name = ["L", "u", "v"]
        for col in [0, 1, 2]:
            axs[row, col].imshow(img[displayed, row + 2 * col])
            axs[row, col].set_title(row_name[row].format(col_name[col]))
    fig.suptitle(f"Image nº{displayed}")
    plt.show()
    return (data, img), (kind, class_)


if __name__ == "__main__":
    df = load("scene.arff")
    (X, _), (names, y) = process(df)
    model = MultiOutputClassifier(LogisticRegression(penalty="l2", C=1))
    model.fit(X, y)
    pred = model.predict_proba(X)
    fig, axs = plt.subplots(2, 3)
    for row in [0, 1]:
        for col in [0, 1, 2]:
            idx = row + 2 * col
            roc_curve = RocCurveDisplay.from_predictions(
                y[:, idx], pred[idx][:, 1], ax=axs[row, col], name=names[idx]
            )
    plt.show()
