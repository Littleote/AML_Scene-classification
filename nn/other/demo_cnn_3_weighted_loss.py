import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 32
LR = 0.001  # Learning rate
EPOCHS = 250
BINARY_CRITERION = 0.5


# Step 1: Load the ARFF file
def load(file: str):
    contents = arff.loadarff(file)
    df = pd.DataFrame(contents[0])
    return df


def process(df: pd.DataFrame):
    # Identify the columns for the means
    attr = [col for col in df.columns if col.startswith("attr")]

    # Split attributes into groups of 49
    means_indices = np.arange(0, len(attr)).reshape(-1, 49)  # Shape (6, 49)

    # Select only the first, third, and fifth groups of 49 attributes
    selected_indices = np.concatenate([means_indices[0], means_indices[2], means_indices[4]])
    selected_attr = [attr[i] for i in selected_indices]

    # Extract the selected features and targets
    data = df[selected_attr].to_numpy(dtype=np.float32)
    classes = ['Beach', 'Sunset', 'FallFoliage', 'Field', 'Mountain', 'Urban']
    targets = df[classes].apply(lambda s: s.map(int)).to_numpy(dtype=np.float32)

    return data, targets


class UrbanSceneCNN(nn.Module):
    def __init__(self):
        super(UrbanSceneCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)

        # Adjusted input size for fc1 based on the final feature map size
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 6)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))

        x = x.contiguous().view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_model(model, train_loader, test_loader, num_epochs=EPOCHS, learning_rate=LR):
    # Compute class counts from the training data
    all_labels = torch.cat([labels for _, labels in train_loader.dataset])
    class_counts = all_labels.sum(dim=0).float()  # Sum along columns to get counts for each class
    total_samples = len(train_loader.dataset)
    class_weights = total_samples / class_counts

    # Define weighted loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Track metrics
    train_losses = []
    val_accuracies = []
    val_f1_scores = []
    gradient_norms = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        gradient_norm = 0
        for inputs, labels in train_loader:
            inputs = inputs.view(-1, 7, 7, 3).permute(0, 3, 1, 2)  # Reshape to (N, C, H, W)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # Compute gradient norm
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            gradient_norm += total_norm ** 0.5

            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))  # Average loss for the epoch
        gradient_norms.append(gradient_norm / len(train_loader))

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.view(-1, 7, 7, 3).permute(0, 3, 1, 2)  # Reshape to (N, C, H, W)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                predictions = (torch.sigmoid(outputs) > BINARY_CRITERION).float()
                all_preds.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                correct += (predictions == labels).all(dim=1).sum().item()
                total += labels.size(0)

        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)

        # Compute F1 score
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        val_f1 = f1_score(all_labels, all_preds, average='micro')  # Micro-average F1 score
        val_f1_scores.append(val_f1)

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {train_losses[-1]:.3f} | Val Acc: {val_accuracies[-1]:.2f}% | F1: {val_f1:.3f} | Grad Norm: {gradient_norms[-1]:.3f}")

    # Plot metrics
    plot_metrics(train_losses, val_accuracies, val_f1_scores, gradient_norms)


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


# Step 5: Main execution
if __name__ == "__main__":
    # Load and process the dataset
    df = load("../scene.arff")
    X, y = process(df)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    # Split the dataset into training and testing sets
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(TRAIN_TEST_SPLIT * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize and train the model
    model = UrbanSceneCNN()
    train_model(model, train_loader, test_loader)