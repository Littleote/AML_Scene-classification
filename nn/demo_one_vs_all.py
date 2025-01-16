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
EPOCHS = 100

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


class SceneDatasetOvA(Dataset):
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
    train_dataset = SceneDatasetOvA(device=device, train=True)
    test_dataset = SceneDatasetOvA(device=device, train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader

########################################################################################################################
# MODEL SPECIFICATION

class NeuralNet(nn.Module):

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

########################################################################################################################
# MODEL TRAINER
class OvATrainer:
    """
    Handles training for multiple binary classifiers (1 per class).
    """
    def __init__(self, models, train_loader, criterion, optimizers):
        self.models = models
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizers = optimizers
        self.train_losses = []

    def train_one_epoch(self):
        train_losses_all = []
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
            train_losses_all.append(avg_loss)
        self.train_losses.append(np.sum(np.asarray(train_losses_all)) / len(self.models))
########################################################################################################################
# MODEL EVALUATION

def compute_alpha_score(predictions, labels, alpha=2.0, beta=1.0, gamma=0.25):
    diff_labels_sub_preds = labels - predictions
    diff_labels_sub_preds[diff_labels_sub_preds < 1] = 0

    diff_preds_sub_labels = predictions - labels
    diff_preds_sub_labels[diff_preds_sub_labels < 1] = 0

    missed = np.asarray([x.sum() for x in diff_labels_sub_preds[0]])
    false_positives = np.asarray([x.sum() for x in diff_preds_sub_labels[0]])

    union = (labels + predictions)
    union[union > 1] = 1

    union = np.asarray([x.sum() for x in union[0]])

    # Compute alpha score for all
    alpha_scores = 1 - ((beta * missed + gamma * false_positives) / union)**alpha
    # Accuracy on the whole dataset
    accuracy_on_dataset = np.sum(alpha_scores) * (1 / labels.shape[1])

    # Per class precision/recall
    alpha_scores = np.asarray([alpha_scores])
    results = {"recall_C": [], "precision_C": []}
    for C in range(labels.shape[2]):
        per_label_alpha_scores = alpha_scores[labels[:, :, C] > 0]
        recal_c = np.sum(per_label_alpha_scores) * (1 / per_label_alpha_scores.size)

        results["recall_C"].append(recal_c)

        per_pred_alpha_scores = alpha_scores[predictions[:, :, C] > 0]
        precision_c = np.sum(per_pred_alpha_scores) * ((1 / per_pred_alpha_scores.size) if per_pred_alpha_scores.size > 0 else 0.0001)

        results["precision_C"].append(precision_c)


    return accuracy_on_dataset, results


# Model Evaluator for 1-vs-all setup
class OvAEvaluator:
    """
    Handles evaluation for multiple binary classifiers (1 per class).
    """
    def __init__(self, models, test_loader, binary_threshold=0.5):
        self.models = models
        self.test_loader = test_loader
        self.binary_threshold = binary_threshold
        self.criterion = criterion

        # Metrics storage
        self.val_f1_scores = [[] for _ in models]
        self.overall_f1_scores = []
        self.alpha_scores = []
        self.per_class = {"recall_C": [], "precision_C": []}

        self.val_hamming_losses = []
        self.lrap_scores = []
        self.val_losses = []

    def evaluate(self):
        all_preds = []
        all_labels = []

        val_losses_all = []
        for class_idx, model in enumerate(self.models):
            model.eval()
            val_loss = 0.0
            preds = []
            labels = []

            with torch.no_grad():
                for inputs, batch_labels in self.test_loader:
                    outputs = model(inputs)
                    predictions = (outputs > self.binary_threshold).float()
                    preds.append(predictions.cpu().numpy())
                    labels.append(batch_labels[:, class_idx].unsqueeze(1).cpu().numpy())
                    val_loss += self.criterion(outputs, batch_labels[:, class_idx].unsqueeze(1)).item()

            avg_loss = val_loss / len(self.test_loader)
            val_losses_all.append(avg_loss)

            preds = np.vstack(preds)
            labels = np.vstack(labels)
            all_preds.append(preds)
            all_labels.append(labels)

            # Compute metrics for this class
            f1 = f1_score(labels, preds, average='binary')
            self.val_f1_scores[class_idx].append(f1)

        self.val_losses.append(np.sum(np.asarray(val_losses_all)) / len(self.models))

        alpha, per_class = compute_alpha_score(np.asarray(all_preds).T, np.asarray(all_labels).T, 1.0, 1.0, 0.25)
        self.alpha_scores.append(alpha)

        all_preds_temp = (np.asarray(all_preds).T)[0]
        all_labels_temp = (np.asarray(all_labels).T)[0]

        lrap = label_ranking_average_precision_score(all_labels_temp, all_preds_temp)
        self.lrap_scores.append(lrap)

        val_hamming_loss = hamming_loss(all_labels_temp, all_preds_temp)
        self.val_hamming_losses.append(val_hamming_loss)

        overall_f1 = f1_score(all_labels_temp, all_preds_temp, average='micro')
        self.overall_f1_scores.append(overall_f1)

        for key, val in per_class.items():
            print(f"Alpha based {key}: {val}")
            self.per_class[key].append(val)
        return np.hstack(all_preds), np.hstack(all_labels)




def plot_metrics(train_losses, validation_losses, val_f1_scores, val_hamming_losses, lrap_scores, alpha_acc, alpha_per_c):
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

    # # Validation Loss
    # plt.subplot(3, 3, 2)
    # plt.plot(epochs, validation_losses, label="Validation Loss", color='green')
    # plt.xlabel("Epochs")
    # plt.ylabel("Validation Loss")
    # plt.title("Validation Loss")
    # plt.grid()
    # plt.legend()

    plt.subplot(3, 3, 2)
    plt.plot(epochs, alpha_acc, label="Alpha score accuracy", color='purple')
    plt.xlabel("Epochs")
    plt.ylabel("Alpha Accuracy")
    plt.title("Alpha score accuracy")
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

    # # Gradient Norms
    # plt.subplot(3, 3, 6)
    # plt.plot(epochs, gradient_norms, label="Gradient Norms", color='purple')
    # plt.xlabel("Epochs")
    # plt.ylabel("Gradient Norm")
    # plt.title("Gradient Norms")
    # plt.grid()
    # plt.legend()

    plt.tight_layout()
    plt.show()

    ## PLOTTING PER CLASS PRECISION AND RECALL
    classes = ['Beach', 'Sunset', 'FallFoliage', 'Field', 'Mountain', 'Urban']
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(16, 10))

    i = 0
    for key, value in alpha_per_c.items():
        value = np.asarray(value)
        i += 1
        for c in range(value.shape[1]):
            plt.subplot(6, 2, (i-1) * value.shape[1] +  c+1)
            plt.plot(epochs, (value.T)[c], label=f"{key} {classes[c]}", color='purple' if key.startswith("precision") else 'green')
            plt.xlabel("Epochs")
            plt.ylabel(f"{key}")
            plt.title(f"{key} {classes[c]}")
            plt.grid()
            plt.legend()

    plt.tight_layout()
    plt.show()


class EarlyStopper:
    def __init__(self, patience=1, max_delta=0):
        self.patience = patience
        self.max_delta = max_delta
        self.counter = 0
        self.max_accuracy = float('-inf')
        self.snapshot_results = None

    def early_stop(self, current_acc, snapshot):
        if current_acc > self.max_accuracy:
            self.max_accuracy = current_acc
            self.snapshot_results = snapshot
            self.counter = 0
        elif current_acc < (self.max_accuracy - self.max_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# Main script
if __name__ == "__main__":
    # Load 1-vs-all DataLoaders
    train_loader, test_loader = get_dataloaders_ova()

    examples = iter(train_loader)
    samples, labels = next(examples)

    input_dim = samples[0].shape[0]
    output_dim = labels[0].shape[0]

    # a separate model, optimizer, and criterion for each class
    n_classes = 6  # Number of classes
    models = [NeuralNet(input_dim, 1).to(device) for _ in range(n_classes)]
    optimizers = [torch.optim.Adam(model.parameters(), lr=LR) for model in models]
    criterion = nn.BCELoss()

    # trainers and evaluators
    trainer = OvATrainer(models, train_loader, criterion, optimizers)
    evaluator = OvAEvaluator(models, test_loader)

    early_stopper = EarlyStopper(patience=5, max_delta=10)

    # MAIN LOOP
    for epoch in range(EPOCHS):
        trainer.train_one_epoch()
        preds, labels = evaluator.evaluate()

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        for class_idx in range(n_classes):
            print(f"  Class {class_idx} - F1 Score: {evaluator.val_f1_scores[class_idx][-1]:.4f}")

        print(f"Epoch {epoch + 1}/{EPOCHS} | Training Loss: {trainer.train_losses[-1]:.3f} "
              f"| Validation Loss: {evaluator.val_losses[-1]:.3f} "
              f"| Hamming losses: {evaluator.val_hamming_losses[-1]:.3f} "
              f"| Lrap scores: {evaluator.lrap_scores[-1]:.3f} "
              f"| Overall macro avg F1 scores: {evaluator.overall_f1_scores[-1]:.3f} "
              f"Alpha score accuracy: {evaluator.alpha_scores[-1]:.3f}")

        snapshot_results = {"Hamming losses": evaluator.val_hamming_losses[-1],
                            "LRAP": evaluator.lrap_scores[-1],
                            "F1 micro": evaluator.overall_f1_scores[-1],
                            "per_class alpha based metrics: ": {"recall": evaluator.per_class["recall_C"][-1],
                                                                "precision": evaluator.per_class["precision_C"][-1]}}

        if early_stopper.early_stop(evaluator.alpha_scores[-1], snapshot_results):
            break

    print(f"Best alpha accuracy: {early_stopper.max_accuracy}")
    print(f"SNAPSHOT RESULTS: {early_stopper.snapshot_results}")

    plot_metrics(trainer.train_losses, evaluator.val_losses, evaluator.overall_f1_scores, evaluator.val_hamming_losses,
                 evaluator.lrap_scores, evaluator.alpha_scores, evaluator.per_class)

    # Final Evaluation
    preds, labels = evaluator.evaluate()
    preds_binary = (preds > BINARY_CRITERION).astype(int)
    overall_f1 = f1_score(labels, preds_binary, average='micro')
    print(f"\nOverall Micro F1 Score: {overall_f1:.4f}")
    print(f"Overall Alpha score: {compute_alpha_score(np.asarray([preds_binary]), np.asarray([labels]))}")


