import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, ToTensor
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import Dataset

# Configuration
TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 32
LR = 0.001  # Learning rate
EPOCHS = 160
BINARY_CRITERION = 0.5

# Configurable Options
BALANCED_SAMPLING = True
AUGMENT_ROTATIONS = False
AUGMENT_HFLIP = True
WEIGHTED_LOSS = True


# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")

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


class TransformDataset(Dataset):
    """Wraps a dataset to apply transforms on the data."""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        data = data.view(3, 7, 7) #.numpy()  # Reshape for transform
        data = np.moveaxis(self.transform(data).numpy(), 0, -1)
        return data, target


def compute_class_weights(labels):
    """Compute class weights based on the inverse frequency of each class."""
    class_counts = labels.sum(axis=0)  # Sum over rows to count occurrences per class
    total_samples = labels.shape[0]
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.tensor(class_weights, dtype=torch.float32)


def create_sampler(labels):
    """Create a WeightedRandomSampler to balance the dataset."""
    class_counts = labels.sum(axis=0)  # Count the number of samples per class
    total_samples = labels.shape[0]

    # Calculate weights for each sample
    weights_per_class = 1.0 / class_counts
    weights = np.dot(labels, weights_per_class)

    return WeightedRandomSampler(weights, num_samples=total_samples, replacement=True)


def get_transforms():
    """Create a data augmentation pipeline."""
    transforms = []
    if AUGMENT_ROTATIONS:
        transforms.append(RandomRotation(degrees=15))  # Rotate images by ±15 degrees
    if AUGMENT_HFLIP:
        transforms.append(RandomHorizontalFlip(p=0.5))  # Flip horizontally with 50% probability
    transforms.append(ToTensor())  # Convert to tensor

    return Compose(transforms)


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


def train_model(model, train_loader, test_loader, class_weights=None, num_epochs=EPOCHS, learning_rate=LR):
    # Loss and optimizer
    if WEIGHTED_LOSS and class_weights is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)  # Weighted loss
    else:
        criterion = nn.BCEWithLogitsLoss()  # Default loss

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Metrics tracking
    train_losses = []
    val_accuracies = []
    val_f1_scores = []
    grad_norms = []  # Gradient norms for each epoch

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        grad_norm_epoch = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.view(-1, 7, 7, 3).permute(0, 3, 1, 2)  # Reshape to (N, C, H, W)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Track gradient norms
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm_epoch += total_norm ** 0.5

            optimizer.step()
            running_loss += loss.item()

        grad_norm_epoch /= len(train_loader)  # Average gradient norm
        grad_norms.append(grad_norm_epoch)

        train_losses.append(running_loss / len(train_loader))  # Average loss for the epoch

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
                predictions = (torch.sigmoid(outputs) > BINARY_CRITERION).float()
                all_preds.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                correct += (predictions == labels).all(dim=1).sum().item()
                total += labels.size(0)

        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        val_f1 = f1_score(all_labels, all_preds, average='micro')  # Micro-average F1 score
        val_f1_scores.append(val_f1)

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {train_losses[-1]:.3f} | Grad Norm: {grad_norm_epoch:.3f} | "
              f"Val Acc: {val_accuracies[-1]:.2f}% | F1: {val_f1:.3f}")

    plot_metrics(train_losses, val_accuracies, val_f1_scores, grad_norms)
    return train_losses, val_accuracies, val_f1_scores

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



if __name__ == "__main__":
    # Load and process data
    df = load("../scene.arff")
    data, targets = process(df)

    # Split into training and testing datasets
    num_samples = len(data)
    train_size = int(TRAIN_TEST_SPLIT * num_samples)
    test_size = num_samples - train_size

    dataset = TensorDataset(torch.tensor(data), torch.tensor(targets))
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Data augmentation and transforms
    if AUGMENT_ROTATIONS or AUGMENT_HFLIP:
        transform = get_transforms()
    else:
        transform = ToTensor()  # Default to basic ToTensor

    if transform:
        # Wrap train_dataset with transformations
        train_dataset = TransformDataset(train_dataset, transform)

    # Weighted sampling
    if BALANCED_SAMPLING:
        sampler = create_sampler(targets[:train_size])  # Use only training set targets
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Compute class weights for the loss function if enabled
    if WEIGHTED_LOSS:
        class_weights = compute_class_weights(targets[:train_size])
        print(f"Class weights: {class_weights.numpy()}")
    else:
        class_weights = None

    # Initialize the model
    model = UrbanSceneCNN()

    # Train the model and visualize results
    train_losses, val_accuracies, val_f1_scores = train_model(
        model, train_loader, test_loader, class_weights=class_weights, num_epochs=EPOCHS, learning_rate=LR
    )
