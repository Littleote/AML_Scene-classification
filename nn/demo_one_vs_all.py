import random

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
DATA_PATH = "../scene.arff"

# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HYPERPARAMETERS
INPUT_SIZE = 49 # 7x7

BATCH_SIZE = 16
LR = 0.001 # Learning rate
EPOCHS = 40

TRAIN_TEST_SPLIT = 0.8
BINARY_CRITERION = 0.5
########################################################################################################################
# DATA LOADING

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




# TODO: img loading 49*2*3 | 49*3 or 49*6 reshaped
class SceneDatasetOvA(Dataset):
    """
    Dataset class for 1-vs-all setup.
    Each sample is replicated once per class for binary classification.
    """
    def __init__(self, device, train=True, test_size=0.2):
        # Load and preprocess data
        df = load(DATA_PATH)
        X, y = process(df)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)

        if train:
            self.x, self.y = X_train, y_train
        else:
            self.x, self.y = X_test, y_test

        # Create 1-vs-all dataset
        # self.samples = []
        xx = []
        yy = []
        for i in range(self.y.shape[1]):  # Iterate over classes
            for xi, yi in zip(self.x, self.y):
                # Binary label for this class
                binary_label = np.zeros(self.y.shape[1], dtype=np.float32)
                binary_label[i] = yi[i]
                if yi[i]:
                    xx.append(xi)
                    yy.append(binary_label)
                    # self.samples.append([xi, binary_label])

        self.x = torch.tensor(xx, dtype=torch.float32).to(device)
        self.y = torch.tensor(yy, dtype=torch.float32).to(device)

        # self.samples = [[torch.tensor(x, dtype=torch.float32).to(device),
        #                  torch.tensor(y, dtype=torch.float32).to(device)] for x, y in self.samples]

        self.n_samples = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


def get_dataloaders_ova():
    """
    Create DataLoaders for the 1-vs-all setup.
    """
    train_dataset = SceneDatasetOvA(device=device, train=True)
    test_dataset = SceneDatasetOvA(device=device, train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader

########################################################################################################################
# MODEL SPECIFICATION

class NeuralNet(nn.Module):

    # def __init__(self, input_size, output_size):
    #     super(NeuralNet, self).__init__()
    #     self.model = nn.Sequential(
    #         nn.Linear(input_size, 128, device=device),
    #         nn.ReLU(),
    #         # nn.Dropout(0.3),
    #         nn.Linear(128, 64, device=device),
    #         nn.ReLU(),
    #         nn.Linear(64, output_size, device=device),
    #         nn.Sigmoid()
    #     )
    #
    def __init__(self, input_dim, output_dim):
        super(NeuralNet, self).__init__()
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

# Model Trainer for 1-vs-all setup
class OvATrainer:
    """
    Handles training for multiple binary classifiers (1 per class).
    """
    def __init__(self, models, train_loader, criterion, optimizers):
        self.models = models
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizers = optimizers
        self.train_losses = [[] for _ in models]

    def train_one_epoch(self):
        for class_idx, model in enumerate(self.models):
            model.train()
            epoch_loss = 0

            for inputs, labels in self.train_loader:
                # Use only the binary label for the current class
                labels = labels[:, class_idx].unsqueeze(1)

                self.optimizers[class_idx].zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizers[class_idx].step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.train_loader)
            self.train_losses[class_idx].append(avg_loss)
########################################################################################################################
# MODEL EVALUATION

def compute_alpha_score(predictions, labels, alpha=2.0, beta=1.0, gamma=0.25):
    diff_labels_sub_preds = labels - predictions
    diff_labels_sub_preds[diff_labels_sub_preds < 1] = 0

    diff_preds_sub_labels = predictions - labels
    diff_preds_sub_labels[diff_preds_sub_labels < 1] = 0

    # missed = diff_labels_sub_preds.sum()
    # false_positives = diff_preds_sub_labels.sum()

    missed = np.asarray([x.sum() for x in diff_labels_sub_preds[0]])
    false_positives = np.asarray([x.sum() for x in diff_preds_sub_labels[0]])

    union = (labels + predictions)
    union[union > 1] = 1

    union = np.asarray([x.sum() for x in union[0]])

    alpha_score = np.sum(1 - ((beta * missed + gamma * false_positives) / union)**alpha) / labels.size


    # Recall

    return alpha_score


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


# Model Evaluator for 1-vs-all setup
class OvAEvaluator:
    """
    Handles evaluation for multiple binary classifiers (1 per class).
    """
    def __init__(self, models, test_loader, binary_threshold=0.5):
        self.models = models
        self.test_loader = test_loader
        self.binary_threshold = binary_threshold

        # Metrics storage
        self.val_f1_scores = [[] for _ in models]
        self.alpha_scores = []

    def evaluate(self):
        all_preds = []
        all_labels = []


        for class_idx, model in enumerate(self.models):
            model.eval()
            preds = []
            labels = []

            with torch.no_grad():
                for inputs, batch_labels in self.test_loader:
                    outputs = model(inputs)
                    predictions = (outputs > self.binary_threshold).float()
                    preds.append(predictions.cpu().numpy())
                    labels.append(batch_labels[:, class_idx].unsqueeze(1).cpu().numpy())

            preds = np.vstack(preds)
            labels = np.vstack(labels)
            all_preds.append(preds)
            all_labels.append(labels)

            # Compute metrics for this class
            f1 = f1_score(labels, preds, average='binary')
            self.val_f1_scores[class_idx].append(f1)

        alpha = compute_alpha_score(np.asarray(all_preds).T, np.asarray(all_labels).T, 1.0, 1.0, 0.25)
        self.alpha_scores.append(alpha)
        return np.hstack(all_preds), np.hstack(all_labels)



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


# Main script
if __name__ == "__main__":
    # Load 1-vs-all DataLoaders
    train_loader, test_loader = get_dataloaders_ova()

    examples = iter(train_loader)
    samples, labels = next(examples)

    # TODO: correct
    input_dim = samples[0].shape[0]
    output_dim = labels[0].shape[0]

    # Create a separate model, optimizer, and criterion for each class
    n_classes = 6  # Number of classes
    models = [NeuralNet(input_dim, 1).to(device) for _ in range(n_classes)]
    optimizers = [torch.optim.Adam(model.parameters(), lr=LR) for model in models]
    criterion = nn.BCELoss()

    # Trainers and Evaluators
    trainer = OvATrainer(models, train_loader, criterion, optimizers)
    evaluator = OvAEvaluator(models, test_loader)

    # Train and evaluate
    for epoch in range(EPOCHS):
        trainer.train_one_epoch()
        preds, labels = evaluator.evaluate()

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        for class_idx in range(n_classes):
            print(f"  Class {class_idx} - F1 Score: {evaluator.val_f1_scores[class_idx][-1]:.4f}")
        print(f"Alpha score: {evaluator.alpha_scores[-1]}")

    # Final Evaluation
    preds, labels = evaluator.evaluate()
    preds_binary = (preds > BINARY_CRITERION).astype(int)
    overall_f1 = f1_score(labels, preds_binary, average='macro')
    print(f"\nOverall Micro F1 Score: {overall_f1:.4f}")
    print(f"Overall Alpha score avg: {compute_alpha_score(preds_binary, labels)}")
    print(f"Multi-label based alpha accuracy: {np.asarray(evaluator.alpha_scores).sum() / labels.size}")

    epochs = range(1, len(evaluator.alpha_scores) + 1)

    plt.figure(figsize=(16, 10))

    # Training Loss
    plt.subplot(1, 1, 1)
    plt.plot(epochs, evaluator.alpha_scores, label="Training Loss", color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

# if __name__ == "__main__":
#     # Basic plain datasets (no transforms)
#     train_loader, test_loader, class_weights = get_dataloaders()
#
#     # TODO: test this
#     # Random horizontal flipping applied datasets (no transforms)
#     # train_loader_flip, test_loader_flip = get_dataloaders(flipping=True)
#
#     examples = iter(train_loader)
#     samples, labels = next(examples)
#
#     # TODO: correct
#     input_dim = samples[0].shape[0]
#     output_dim = labels[0].shape[0]
#
#
#     model = NeuralNet(input_dim, output_dim)
#     criterion = Criterion(type="SoftMarginMultilabel", weights=class_weights)
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#
#     model_trainer = ModelTrainer(model, train_loader)
#     model_evaluator = ModelEvaluator(model, test_loader)
#
#     # TRAIN LOOP
#     for epoch in range(EPOCHS):
#         model_trainer.train_model()
#
#         model_evaluator.evaluate_model()
#         # model_evaluator.print_results(epoch)
#         print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {model_trainer.train_losses[-1]:.3f} "
#               f"| Micro F1: {model_evaluator.val_f1_scores[-1]:.3f} |"
#               f"| Hamming losses: {model_evaluator.val_hamming_losses[-1]:.3f} |"
#               f"| Subset Accuracies: {model_evaluator.val_subset_accuracies[-1]:.3f} |"
#               f"| Lrap scores: {model_evaluator.lrap_scores[-1]:.3f} |"
#               f" Grad Norm: {model_trainer.gradient_norms[-1]:.3f}")



    # # Plot metrics
    # plot_metrics(model_trainer.train_losses, model_evaluator.val_f1_scores, model_evaluator.val_hamming_losses,
    #              model_evaluator.val_subset_accuracies, model_evaluator.lrap_scores, model_trainer.gradient_norms)
    #
    #
    # # Final evaluation
    # y_true, y_pred = [], []
    # model.eval()
    # with torch.no_grad():
    #     for batch_X, batch_y in test_loader:
    #         outputs = model(batch_X)
    #         y_true.append(batch_y.cpu().numpy())
    #         y_pred.append(outputs.cpu().numpy())
    #
    # y_true = np.vstack(y_true)
    # y_pred = np.vstack(y_pred)
    # y_pred_binary = (y_pred > BINARY_CRITERION).astype(int)
    #
    # print("Classification Report:")
    # for i, class_name in enumerate(['Beach', 'Sunset', 'FallFoliage', 'Field', 'Mountain', 'Urban']):
    #     print(f"\nClass: {class_name}")
    #     print(classification_report(y_true[:, i], y_pred_binary[:, i]))
    #
    # overall_roc_auc = roc_auc_score(y_true, y_pred, average='macro')
    # print(f"\nOverall ROC AUC Score: {overall_roc_auc:.2f}")
    #
    #

########################################################################################################################
########################################################################################################################
########################################################################################################################
# WEIGHTED_LOSS = False
#
# def load(file: str):
#     contents = arff.loadarff(file)
#     df = pd.DataFrame(contents[0])
#     return df
#
#
# def process(df: pd.DataFrame):
#     attr = [col for col in df.columns if col.startswith("attr")]
#     classes = ['Beach', 'Sunset', 'FallFoliage', 'Field', 'Mountain', 'Urban']
#
#     data = df[attr].to_numpy(dtype=np.float32)
#     targets = df[classes].apply(lambda s: s.map(int)).to_numpy(dtype=np.float32)
#
#     return data, targets
#
#
#
# def compute_class_weights(labels):
#     """Compute class weights based on the inverse frequency of each class."""
#     class_counts = labels.sum(axis=0)  # Sum over rows to count occurrences per class
#     total_samples = labels.shape[0]
#     class_weights = total_samples / (len(class_counts) * class_counts)
#     return torch.tensor(class_weights, dtype=torch.float32).to(device)
#
#
# # class MultiLabelClassifier(nn.Module):
# #     def __init__(self, input_dim, output_dim):
# #         super(MultiLabelClassifier, self).__init__()
# #         self.model = nn.Sequential(
# #             nn.Linear(input_dim, 128),
# #             nn.ReLU(),
# #             nn.Dropout(0.3),
# #             nn.Linear(128, 64),
# #             nn.ReLU(),
# #             nn.Dropout(0.3),
# #             nn.Linear(64, output_dim),
# #             nn.Sigmoid()
# #         )
# #
# #     def forward(self, x):
# #         return self.model(x)
#
# # class MultiLabelClassifier(nn.Module):
# #     def __init__(self, input_dim, output_dim):
# #         super(MultiLabelClassifier, self).__init__()
# #         self.model = nn.Sequential(
# #             nn.Linear(input_dim, 256),
# #             nn.ReLU(),
# #             nn.Dropout(0.3),
# #             nn.Linear(256, 128),
# #             nn.ReLU(),
# #             nn.Dropout(0.3),
# #             nn.Linear(128, 64),
# #             nn.ReLU(),
# #             nn.Dropout(0.3),
# #             nn.Linear(64, output_dim),
# #             nn.Sigmoid()
# #         )
# #
# #     def forward(self, x):
# #         return self.model(x)
# #
#
#
# class MultiLabelClassifier(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(MultiLabelClassifier, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, 512, device=device),
#             nn.ReLU(),
#             # nn.Dropout(0.3),
#             nn.Linear(512, 256, device=device),
#             nn.ReLU(),
#             # nn.Dropout(0.3),
#             nn.Linear(256, 128, device=device),
#             nn.ReLU(),
#             # nn.Dropout(0.3),
#             nn.Linear(128, 64, device=device),
#             nn.ReLU(),
#             # nn.Dropout(0.3),
#             nn.Linear(64, output_dim, device=device),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.model(x)
#
#
#
# BATCH_SIZE = 16
# LR = 0.001 # Learning rate
# EPOCHS = 300
#
# TRAIN_TEST_SPLIT = 0.8
# BINARY_CRITERION = 0.5
#
# if __name__ == "__main__":
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#
#     df = load("../scene.arff")
#     X, y = process(df)
#     X_tensor = torch.tensor(X).to(device)
#     y_tensor = torch.tensor(y).to(device)
#
#     # Split data
#     dataset = TensorDataset(X_tensor, y_tensor)
#     train_size = int(TRAIN_TEST_SPLIT * len(dataset))
#     test_size = len(dataset) - train_size
#     train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
#
#
#
#     # Load data
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
#
#     if WEIGHTED_LOSS:
#         class_weights = compute_class_weights(y[:train_size])
#         print(f"Class weights: {class_weights.cpu().numpy()}")
#     else:
#         class_weights = None
#     # Initialize the model, loss, and optimizer
#
#     if WEIGHTED_LOSS and class_weights is not None:
#         criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)  # Weighted loss
#     else:
#         # criterion = nn.BCEWithLogitsLoss()  # Default loss
#         criterion = nn.BCELoss()  # Binary cross entropy
#     input_dim = X.shape[1]
#     output_dim = y.shape[1]
#     model = MultiLabelClassifier(input_dim, output_dim)
#     optimizer = optim.Adam(model.parameters(), lr=LR)
#
#     # Training metrics
#     train_losses = []
#     val_accuracies = []
#     val_f1_scores = []
#     gradient_norms = []
#
#     for epoch in range(EPOCHS):
#         model.train()
#         epoch_loss = 0
#         gradient_norm = 0
#
#         for batch_X, batch_y in train_loader:
#             optimizer.zero_grad()
#             outputs = model(batch_X)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#
#             # Compute gradient norm
#             total_norm = 0
#             for p in model.parameters():
#                 if p.grad is not None:
#                     total_norm += p.grad.data.norm(2).item() ** 2
#             gradient_norm += total_norm ** 0.5
#
#             optimizer.step()
#             epoch_loss += loss.item()
#
#         train_losses.append(epoch_loss / len(train_loader))
#         gradient_norms.append(gradient_norm / len(train_loader))
#
#         # Validation phase
#         model.eval()
#         val_loss = 0.0
#         all_preds = []
#         all_labels = []
#         correct, total = 0, 0
#
#         with torch.no_grad():
#             for inputs, labels in test_loader:
#                 outputs = model(inputs)
#                 val_loss += criterion(outputs, labels).item()
#
#                 predictions = (outputs > BINARY_CRITERION).float()
#                 all_preds.append(predictions.cpu().numpy())
#                 all_labels.append(labels.cpu().numpy())
#
#                 correct += (predictions == labels).all(dim=1).sum().item()
#                 total += labels.size(0)
#
#         val_accuracy = 100 * correct / total if total > 0 else 0
#         val_accuracies.append(val_accuracy)
#
#         if all_preds and all_labels:
#             all_preds = np.vstack(all_preds)
#             all_labels = np.vstack(all_labels)
#             val_f1 = f1_score(all_labels, all_preds, average='micro')
#         else:
#             val_f1 = 0
#         val_f1_scores.append(val_f1)
#
#         print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {train_losses[-1]:.3f} | Val Acc: {val_accuracies[-1]:.2f}% | F1: {val_f1:.3f} | Grad Norm: {gradient_norms[-1]:.3f}")
#
#     # Plot metrics
#     plot_metrics(train_losses, val_accuracies, val_f1_scores, gradient_norms)
#
#     # Final evaluation
#     y_true, y_pred = [], []
#     model.eval()
#     with torch.no_grad():
#         for batch_X, batch_y in test_loader:
#             outputs = model(batch_X)
#             y_true.append(batch_y.numpy())
#             y_pred.append(outputs.numpy())
#
#     y_true = np.vstack(y_true)
#     y_pred = np.vstack(y_pred)
#     y_pred_binary = (y_pred > BINARY_CRITERION).astype(int)
#
#     print("Classification Report:")
#     for i, class_name in enumerate(['Beach', 'Sunset', 'FallFoliage', 'Field', 'Mountain', 'Urban']):
#         print(f"\nClass: {class_name}")
#         print(classification_report(y_true[:, i], y_pred_binary[:, i]))
#
#     overall_roc_auc = roc_auc_score(y_true, y_pred, average='macro')
#     print(f"\nOverall ROC AUC Score: {overall_roc_auc:.2f}")
