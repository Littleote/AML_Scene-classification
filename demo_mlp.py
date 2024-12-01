import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report, roc_auc_score
from scipy.io import arff
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def load(file: str):
    contents = arff.loadarff(file)
    df = pd.DataFrame(contents[0])
    return df


def process(df: pd.DataFrame):
    attr = [col for col in df.columns if col.startswith("attr")]
    classes = ['Beach', 'Sunset', 'FallFoliage', 'Field', 'Mountain', 'Urban']

    data = df[attr].to_numpy(dtype=np.float32)
    targets = df[classes].apply(lambda s: s.map(int)).to_numpy(dtype=np.float32)

    return data, targets


class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiLabelClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


BATCH_SIZE = 16
LR = 0.001 # Learning rate
EPOCHS = 120

TRAIN_TEST_SPLIT = 0.8
BINARY_CRITERION = 0.45

if __name__ == "__main__":

    df = load("scene.arff")
    X, y = process(df)
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    # Split data
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(TRAIN_TEST_SPLIT * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Load data
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # TODO: try dim reduction

    # Initialize the model, loss, and optimizer
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    model = MultiLabelClassifier(input_dim, output_dim)
    criterion = nn.BCELoss()  # Bin cross entropy
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training

    train_losses = []
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        print(f">> Epoch {epoch + 1}/{EPOCHS} | Loss: {epoch_loss / len(train_loader):.3f}")

    # Eval
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            y_true.append(batch_y.numpy())
            y_pred.append(outputs.numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    # Threshold predictions at 0.5 -> is it the specific type or not? Allows for multiple at once
    y_pred_binary = (y_pred > BINARY_CRITERION).astype(int)

    # Per class stats
    print("Classification Report:")
    for i, class_name in enumerate(['Beach', 'Sunset', 'FallFoliage', 'Field', 'Mountain', 'Urban']):
        print(f"\nClass: {class_name}")
        print(classification_report(y_true[:, i], y_pred_binary[:, i]))

    overall_roc_auc = roc_auc_score(y_true, y_pred, average='macro')
    print(f"\nOverall ROC AUC Score: {overall_roc_auc:.2f}")

    # Plotting training progression / loss
    plt.figure()
    plt.plot(train_losses, label="TR Loss")
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
