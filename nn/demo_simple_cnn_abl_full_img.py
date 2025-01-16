import os
import random

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from sklearn.metrics import classification_report, roc_auc_score
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, classification_report, label_ranking_average_precision_score

np.random.seed(0)
torch.manual_seed(0)

########################################################################################################################

DATA_PATH = "./luv_transform_ablation/miml-image-data"

# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HYPERPARAMETERS
INPUT_SIZE = 49 # 7x7

BATCH_SIZE = 32
LR = 0.0001 # Learning rate
EPOCHS = 100

TRAIN_TEST_SPLIT = 0.8
BINARY_CRITERION = 0.5
########################################################################################################################
# DATA LOADING

def load_img(image_path, target_size=(128, 128)):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    image = image[...,::-1]

    # Visualize for correctness
    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    feature_vector = np.asarray(image) / 255.0

    # Reshape to (channels, height, width)
    feature_vector = feature_vector.transpose(2, 0, 1)  # Transpose to (C, H, W)
    # feature_vector = feature_vector.reshape(3, feature_vector.shape[0], feature_vector.shape[1])

    return np.array(feature_vector)

def load(base_dir: str):

    import scipy as sc

    # File containing target labels
    mat_file_path = os.path.join(base_dir, 'processed', 'miml data.mat')
    mat_file = sc.io.loadmat(mat_file_path)

    # Array of target labels
    target_array = mat_file['targets'].T
    target_list = [[j if j == 1 else 0 for j in row] for row in target_array]

    # Class list
    class_list = [j[0][0] for j in mat_file['class_name']]

    # Create DataFrame for images and labels
    file_list = [str(a) + '.jpg' for a in range(1, 2001, 1)]
    df_train = pd.DataFrame({'image': file_list, 'labels': target_list})

    return df_train, class_list

def process_dataset(image_folder, df_train):
    all_features = []

    for idx, row in df_train.iterrows():
        image_path = os.path.join(DATA_PATH, image_folder, row['image'])
        try:
            features = load_img(image_path)
            all_features.append(features)
        except FileNotFoundError as e:
            print(e)

    # Convert to tensor
    feature_tensor = torch.tensor(all_features, dtype=torch.float32)
    target_tensor = torch.tensor(df_train['labels'].tolist(), dtype=torch.float32)

    return feature_tensor, target_tensor

def compute_class_weights(labels):
    """Compute class weights based on the inverse frequency of each class."""
    class_counts = labels.sum(axis=0)  # Sum over rows to count occurrences per class
    total_samples = labels.shape[0]
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.tensor(class_weights, dtype=torch.float32).to(device)



from skmultilearn.model_selection import iterative_train_test_split

# TODO: img loading 49*2*3 | 49*3 or 49*6 reshaped
class SceneDataset(Dataset):
    def __init__(self, device, train, test_size=0.2, transform=None):
        #data loading
        df, class_list = load(DATA_PATH)
        X, y = process_dataset("original", df)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)
        # X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.2)

        if train:
            self.x = torch.tensor(X_train).to(device)
            self.y = torch.tensor(y_train).to(device)
        else:
            self.x = torch.tensor(X_test).to(device)
            self.y = torch.tensor(y_test).to(device)

        self.n_samples = self.x.shape[0]

        self.transform = transform

        self.class_weights = compute_class_weights(y)

        self.class_list = class_list

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            return self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples


class HorizontalFlip:
    def __init__(self, probability=0.4):
        self.probability = probability

    def __call__(self, sample):
        x,y = sample
        if random.randrange(0,1) >= 0.5:
            x = torch.flip(x, [0, 1])
        return x, y


def get_dataloaders(flipping=False):
    if flipping:
        train_dataset = SceneDataset(device=device, transform=HorizontalFlip, train=True)
        test_dataset = SceneDataset(device=device, transform=HorizontalFlip, train=False)
    else:
        train_dataset = SceneDataset(device=device, train=True)
        test_dataset = SceneDataset(device=device, train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



    return train_loader, test_loader, train_dataset.class_weights, train_dataset.class_list


########################################################################################################################
# MODEL SPECIFICATION

class UrbanSceneCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UrbanSceneCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1,  device=device)
        # self.bn1 = nn.BatchNorm2d(32,  device=device)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1,  device=device)
        # self.bn2 = nn.BatchNorm2d(64,  device=device)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1,  device=device)
        # self.bn3 = nn.BatchNorm2d(128,  device=device)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)

        # self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 16*16, 64,  device=device)
        # self.fc1 = nn.Linear(128 * 16*16, 64,  device=device)

        # self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(64, output_dim,  device=device)
        self.output_sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        # x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        # x = self.pool(torch.relu(self.bn3(self.conv3(x))))

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten
        # x = x.contiguous().view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        return self.output_sigmoid(x)

# # AlexNet
# class UrbanSceneCNN(nn.Module):
#     def __init__(self):
#         super(UrbanSceneCNN, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # Conv1
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # Pool1 -> Output: 64 x 3 x 3
#
#             nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),  # Conv2
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=1, padding=0),  # Pool2 -> Output: 192 x 2 x 2
#
#             nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1),  # Conv3
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # Conv4
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # Conv5
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=1, padding=0)  # Pool3 -> Output: 256 x 1 x 1
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(256 * 1 * 1, 4096),  # Fully connected 1
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(4096, 4096),  # Fully connected 2
#             nn.ReLU(),
#             nn.Linear(4096, 6)  # Fully connected 3 -> Output layer
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)  # Flatten feature map
#         x = self.classifier(x)
#         return x


class Criterion(nn.Module):
    def __init__(self, type="BCELoss", weights=None):
        super(Criterion, self).__init__()
        if type == "BCELoss":
            self.loss = nn.BCELoss()
        else:
            self.loss = torch.nn.MultiLabelSoftMarginLoss() # weights

    def forward(self, pred, target):
        return ((pred - target)**2).mean()

########################################################################################################################
# MODEL TRAINER


class ModelTrainer:
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader

        self.train_losses = []
        self.gradient_norms = []

    # TODO: add crosstraining
    def train_model(self):

        model.train()
        epoch_loss = 0
        gradient_norm = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y) # predicted | actual

            # Backprop
            loss.backward()

            # Compute gradient norm
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            gradient_norm += total_norm ** 0.5

            # GD
            optimizer.step()
            epoch_loss += loss.item()

        self.train_losses.append(epoch_loss / len(train_loader))
        self.gradient_norms.append(gradient_norm / len(train_loader))


########################################################################################################################
# MODEL EVALUATION
class ModelEvaluator:
    def __init__(self, model, test_loader, binary_threshold=0.5):
        self.model = model
        self.test_loader = test_loader
        # self.criterion = criterion
        self.binary_threshold = binary_threshold
        self.val_f1_scores = []

        self.val_hamming_losses = []
        self.val_subset_accuracies = []
        self.lrap_scores = []

        self.results = dict()

    def evaluate_model(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                # Forward pass
                outputs = self.model(inputs)
                # val_loss += self.criterion(outputs, labels).item()

                # Convert outputs to binary predictions based on threshold
                predictions = (outputs > self.binary_threshold).float()
                all_preds.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        if all_preds and all_labels:
            # Combine all predictions and labels
            all_preds = np.vstack(all_preds)
            all_labels = np.vstack(all_labels)

            # Calculate metrics
            val_f1_micro = f1_score(all_labels, all_preds, average='macro')
            val_hamming_loss = hamming_loss(all_labels, all_preds)
            val_subset_accuracy = accuracy_score(all_labels, all_preds)
            lrap = label_ranking_average_precision_score(all_labels, all_preds)

            # Save metrics
            self.val_f1_scores.append(val_f1_micro)
            self.val_hamming_losses.append(val_hamming_loss)
            self.val_subset_accuracies.append(val_subset_accuracy)
            self.lrap_scores.append(lrap)


# Util plotting fnc
def plot_metrics(train_losses, val_f1_scores, val_hamming_losses, val_subset_accuracies, lrap_scores, gradient_norms):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(16, 10))

    # Training Loss
    plt.subplot(3, 3, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid()
    plt.legend()

    # Validation Accuracy
    plt.subplot(3, 3, 2)
    plt.plot(epochs, val_subset_accuracies, label="Validation Subset Accuracies", color='green')
    plt.xlabel("Epochs")
    plt.ylabel("Subset Accuracy")
    plt.title("Subset Accuracies")
    plt.grid()
    plt.legend()

    # Validation F1 Score
    plt.subplot(3, 3, 3)
    plt.plot(epochs, val_f1_scores, label="Validation Micro F1 Score", color='red')
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("Validation Micro F1 Score")
    plt.grid()
    plt.legend()

    # Validation Hamming Losses
    plt.subplot(3, 3, 4)
    plt.plot(epochs, val_hamming_losses, label="Hamming loss", color='brown')
    plt.xlabel("Epochs")
    plt.ylabel("Hamming loss")
    plt.title("Validation Hamming Losses")
    plt.grid()
    plt.legend()

    # Validation Lrap Score
    plt.subplot(3, 3, 5)
    plt.plot(epochs, lrap_scores, label="Validation lrap Score", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Lrap Score")
    plt.title("Validation Lrap Score")
    plt.grid()
    plt.legend()

    # Gradient Norms
    plt.subplot(3, 3, 6)
    plt.plot(epochs, gradient_norms, label="Gradient Norms", color='purple')
    plt.xlabel("Epochs")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norms")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Basic plain datasets (no transforms)
    train_loader, test_loader, class_weights, class_list = get_dataloaders()

    # TODO: test this
    # Random horizontal flipping applied datasets (no transforms)
    # train_loader_flip, test_loader_flip = get_dataloaders(flipping=True)

    examples = iter(train_loader)
    samples, labels = next(examples)

    # TODO: correct
    input_dim = samples[0].shape[0]
    output_dim = labels[0].shape[0]


    model = UrbanSceneCNN(input_dim, output_dim)
    criterion = Criterion(type="SoftMarginMultilabel", weights=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model_trainer = ModelTrainer(model, train_loader)
    model_evaluator = ModelEvaluator(model, test_loader)

    # TRAIN LOOP
    for epoch in range(EPOCHS):
        model_trainer.train_model()

        model_evaluator.evaluate_model()
        # model_evaluator.print_results(epoch)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {model_trainer.train_losses[-1]:.3f} "
              f"| Micro F1: {model_evaluator.val_f1_scores[-1]:.3f} |"
              f"| Hamming losses: {model_evaluator.val_hamming_losses[-1]:.3f} |"
              f"| Subset Accuracies: {model_evaluator.val_subset_accuracies[-1]:.3f} |"
              f"| Lrap scores: {model_evaluator.lrap_scores[-1]:.3f} |"
              f" Grad Norm: {model_trainer.gradient_norms[-1]:.3f}")



    # Plot metrics
    plot_metrics(model_trainer.train_losses, model_evaluator.val_f1_scores, model_evaluator.val_hamming_losses,
                 model_evaluator.val_subset_accuracies, model_evaluator.lrap_scores, model_trainer.gradient_norms)


    # Final evaluation
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            y_true.append(batch_y.cpu().numpy())
            y_pred.append(outputs.cpu().numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    y_pred_binary = (y_pred > BINARY_CRITERION).astype(int)

    print("Classification Report:")
    for i, class_name in enumerate(class_list):
        print(f"\nClass: {class_name}")
        print(classification_report(y_true[:, i], y_pred_binary[:, i]))

    overall_roc_auc = roc_auc_score(y_true, y_pred, average='macro')
    print(f"\nOverall ROC AUC Score: {overall_roc_auc:.2f}")



########################################################################################################################
