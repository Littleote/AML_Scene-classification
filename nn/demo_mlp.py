import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report, roc_auc_score
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score


WEIGHTED_LOSS = False

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



def compute_class_weights(labels):
    """Compute class weights based on the inverse frequency of each class."""
    class_counts = labels.sum(axis=0)  # Sum over rows to count occurrences per class
    total_samples = labels.shape[0]
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.tensor(class_weights, dtype=torch.float32).to(device)


# class MultiLabelClassifier(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(MultiLabelClassifier, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, output_dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.model(x)

# class MultiLabelClassifier(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(MultiLabelClassifier, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, output_dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.model(x)
#


class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiLabelClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512, device=device),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(512, 256, device=device),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(256, 128, device=device),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128, 64, device=device),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(64, output_dim, device=device),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def plot_metrics(train_losses, val_accuracies, val_f1_scores, gradient_norms):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(16, 10))

    # Training Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid()
    plt.legend()

    # Validation Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", color='green')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy")
    plt.grid()
    plt.legend()

    # Validation F1 Score
    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_f1_scores, label="Validation F1 Score", color='red')
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("Validation F1 Score")
    plt.grid()
    plt.legend()

    # Gradient Norms
    plt.subplot(2, 2, 4)
    plt.plot(epochs, gradient_norms, label="Gradient Norms", color='purple')
    plt.xlabel("Epochs")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norms")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

BATCH_SIZE = 16
LR = 0.001 # Learning rate
EPOCHS = 300

TRAIN_TEST_SPLIT = 0.8
BINARY_CRITERION = 0.5

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    df = load("../scene.arff")
    X, y = process(df)
    X_tensor = torch.tensor(X).to(device)
    y_tensor = torch.tensor(y).to(device)

    # Split data
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(TRAIN_TEST_SPLIT * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    if WEIGHTED_LOSS:
        class_weights = compute_class_weights(y[:train_size])
        print(f"Class weights: {class_weights.cpu().numpy()}")
    else:
        class_weights = None

    # Load data
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the model, loss, and optimizer

    if WEIGHTED_LOSS and class_weights is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)  # Weighted loss
    else:
        # criterion = nn.BCEWithLogitsLoss()  # Default loss
        criterion = nn.BCELoss()  # Binary cross entropy
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    model = MultiLabelClassifier(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training metrics
    train_losses = []
    val_accuracies = []
    val_f1_scores = []
    gradient_norms = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        gradient_norm = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Compute gradient norm
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            gradient_norm += total_norm ** 0.5

            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        gradient_norms.append(gradient_norm / len(train_loader))

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()

                predictions = (outputs > BINARY_CRITERION).float()
                all_preds.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                correct += (predictions == labels).all(dim=1).sum().item()
                total += labels.size(0)

        val_accuracy = 100 * correct / total if total > 0 else 0
        val_accuracies.append(val_accuracy)

        if all_preds and all_labels:
            all_preds = np.vstack(all_preds)
            all_labels = np.vstack(all_labels)
            val_f1 = f1_score(all_labels, all_preds, average='micro')
        else:
            val_f1 = 0
        val_f1_scores.append(val_f1)

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {train_losses[-1]:.3f} | Val Acc: {val_accuracies[-1]:.2f}% | F1: {val_f1:.3f} | Grad Norm: {gradient_norms[-1]:.3f}")

    # Plot metrics
    plot_metrics(train_losses, val_accuracies, val_f1_scores, gradient_norms)

    # Final evaluation
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            y_true.append(batch_y.numpy())
            y_pred.append(outputs.numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    y_pred_binary = (y_pred > BINARY_CRITERION).astype(int)

    print("Classification Report:")
    for i, class_name in enumerate(['Beach', 'Sunset', 'FallFoliage', 'Field', 'Mountain', 'Urban']):
        print(f"\nClass: {class_name}")
        print(classification_report(y_true[:, i], y_pred_binary[:, i]))

    overall_roc_auc = roc_auc_score(y_true, y_pred, average='macro')
    print(f"\nOverall ROC AUC Score: {overall_roc_auc:.2f}")
