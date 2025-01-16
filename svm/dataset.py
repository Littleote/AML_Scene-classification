from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as split


def load_arff(file: str):
    contents = arff.loadarff(file)
    df = pd.DataFrame(contents[0])
    return df


def process(df: pd.DataFrame, verbose: bool = True):
    attr = [col for col in df.columns if col.startswith("attr")]
    names = [col for col in df.columns if not col.startswith("attr")]
    data = df[attr].to_numpy()
    image = data.reshape((-1, 6, 7, 7))
    kind = df[names].apply(lambda s: s.map(int)).to_numpy()
    for name, positive in zip(names, kind.sum(axis=0)):
        print(
            f"\t{name}: {positive}/{kind.shape[0]} = {round(100 * positive / kind.shape[0], 2)}%"
        )
    return data, image, kind, names


def load_npy(file: str):
    with open(file, mode="rb") as handler:
        data = np.load(handler)
        image = np.load(handler)
        kind = np.load(handler)
        names = np.load(handler)
    return data, image, kind, names


def save_npy(file: str, data, image, kind, names):
    with open(file, mode="wb") as handler:
        np.save(handler, data)
        np.save(handler, image)
        np.save(handler, kind)
        np.save(handler, names)


if __name__ == "__main__":
    df = load_arff("scene.arff")
    df_train, df_test = split(df, train_size=0.8, shuffle=True)
    print("Train:")
    save_npy("scene_train.npy", *process(df_train))
    print("Test:")
    save_npy("scene_test.npy", *process(df_test))
