import argparse
import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from ecg_arrhythmia.data.ecg_dataset import LABEL_TO_INDEX, ECGDataset
from ecg_arrhythmia.models.cnn_baseline_v1 import CNNBaselineV1
from ecg_arrhythmia.models.cnn_baseline_v2 import CNNBaselineV2

logger = logging.getLogger(__name__)

NUM_CLASSES = len(LABEL_TO_INDEX)

# Allows us to convert index to label
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}

# ---------------------------------------------------------------------
#                            Define Types
# ---------------------------------------------------------------------


# Define return types
class ClassMetrics(TypedDict):
    precision: float
    recall: float
    f1: float
    total_class_count: int


class EvaluationMetrics(TypedDict):
    loss: float
    accuracy: float
    macro_f1: float
    per_class: dict[str, ClassMetrics]
    confusion_matrix: list[list[int]]


# ---------------------------------------------------------------------
#                        Define Logger Helpers
# ---------------------------------------------------------------------


def log_per_class_metrics(evaluation_metrics: EvaluationMetrics) -> None:

    logger.info("per-class validation metrics:")

    # Grab each label and its corresponding per_class directiory and
    # log its contents.
    for label, class_metrics in evaluation_metrics["per_class"].items():
        logger.info(
            "%s | precision: %.4f | recall: %.4f | f1: %.4f | total_class_count: %s",
            label,
            class_metrics["precision"],
            class_metrics["recall"],
            class_metrics["f1"],
            class_metrics["total_class_count"],
        )


def log_confusion_matrix(evaluation_metrics: EvaluationMetrics) -> None:

    labels = [INDEX_TO_LABEL[index] for index in range(NUM_CLASSES)]

    logger.info("confusion matrix: rows=true labels, columns=predicted labels")
    logger.info("labels: %s", labels)

    for label, row in zip(labels, evaluation_metrics["confusion_matrix"], strict=True):
        logger.info("%s: %s", label, row)


# ---------------------------------------------------------------------
#                     Define Helpers For Evaluation
# ---------------------------------------------------------------------


def safe_divide(numerator: float, denominator: float) -> float:

    if denominator == 0:
        return 0.0

    return numerator / denominator


def calculate_metrics_from_confusion_matrix(
    confusion: np.ndarray,
) -> tuple[float, float, dict[str, ClassMetrics]]:
    per_class = {}
    f1_scores = []

    total_predictions = confusion.sum()

    # np.trace(confusion) sums the diagonal (i.e., the correct prediction)
    # So this is the correct predictions over the total predictions
    accuracy = safe_divide(float(np.trace(confusion)), float(total_predictions))

    for class_index in range(NUM_CLASSES):
        label = INDEX_TO_LABEL[class_index]

        # True positives: samples where the true class and predicted
        # class are both class_index. These sit on the diagonal of the
        # confusion matrix.
        tp = confusion[class_index, class_index]

        # False negatives: samples that truly belong to class_index
        # but were predicted as another class. The row contains all
        # real samples of this class, so subtract the correct ones.
        fn = confusion[class_index, :].sum() - tp

        # False positives: samples predicted as class_index but
        # whose true label was another class. The column contains
        # all predictions made as this class, so subtract the correct ones.
        fp = confusion[:, class_index].sum() - tp

        # Per-class precision =
        # correct predictions for this class / all predictions made as this class.
        precision = safe_divide(float(tp), float(tp + fp))

        # Per-class recall =
        # correct predictions for this class / all samples that truly belong to
        # this class.
        recall = safe_divide(float(tp), float(tp + fn))

        # Combines precision and recall
        f1 = safe_divide(2 * precision * recall, precision + recall)

        # Build the per class record
        per_class[label] = {
            "precision": np.round(precision, 4),
            "recall": np.round(recall, 4),
            "f1": np.round(f1, 4),
            "total_class_count": int(confusion[class_index, :].sum()),
        }

        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores))

    return accuracy, macro_f1, per_class


# ---------------------------------------------------------------------
#                             Evaluation
# ---------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    split_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EvaluationMetrics:

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_true_tensors: list[torch.Tensor] = []
    all_predicted_tensors: list[torch.Tensor] = []

    with torch.no_grad():
        for X_batch, y_batch in split_loader:
            # Move batches to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Calculate raw logits
            logits = model(X_batch)

            # Calculate loss and predictions
            loss = criterion(logits, y_batch)
            predictions = logits.argmax(dim=1)

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Note this for future reference:

            # Avoid batch-by-batch CPU/Python list overhead.
            # Keep track of tensors, then concatenate at the end.
            # You cannot convert GPU tensors to numpy arrays,
            # that is why we put them on the CPU first
            all_true_tensors.append(y_batch.cpu())
            all_predicted_tensors.append(predictions.cpu())

    # Single cat operation is significantly faster
    all_true_labels = torch.cat(all_true_tensors).numpy()
    all_predicted_labels = torch.cat(all_predicted_tensors).numpy()

    # Create the confusion matrix. Rows are true labels
    # columns are predicted labels. returns a
    # (num_classes, num_classes) numpy array.
    # i.e., row 0 (N) is the true label,
    # then row 0, column 0 (N) is the predictions for N,
    # row 0, column 1 (S) is the predictions for S when the true label was N
    # etc.
    confusion = confusion_matrix(
        all_true_labels,
        all_predicted_labels,
        labels=list(range(NUM_CLASSES)),
    )

    accuracy, macro_f1, per_class = calculate_metrics_from_confusion_matrix(confusion)

    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


# ---------------------------------------------------------------------
#                     Helpers For Training
# ---------------------------------------------------------------------


def compute_class_weights(
    dataset: ECGDataset,
    device: torch.device,
) -> torch.Tensor:
    labels = torch.tensor(dataset.y_indices, dtype=torch.long)

    # Count how many samples belong to each class.
    # Example: tensor([9000, 500, 450, 50])
    class_counts = torch.bincount(labels, minlength=NUM_CLASSES).float()
    total_samples = class_counts.sum()

    # Give rare classes larger weights and common classes smaller weights.
    # clamp_min(1.0) prevents division by zero if a class has no samples.
    class_weights = total_samples / (NUM_CLASSES * class_counts.clamp_min(1.0))

    # Example for future reference:
    # class_counts = tensor([9000, 500, 450, 50]), total_samples = 10000
    # denominator = 4 * tensor([9000, 500, 450, 50])
    #             = tensor([36000, 2000, 1800, 200])
    # class_weights = 10000 / denominator
    #               = tensor([0.2778, 5.0000, 5.5556, 50.0000])
    # This means rare classes receive larger loss weights, while common
    # classes receive smaller loss weights.

    return class_weights.to(device)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> float:

    # Put the model in train mode
    model.train()
    # This is the loss of all batches
    # across this epoch
    total_loss = 0.0
    total_samples = 0

    # For each batch in train loader
    for X_batch, y_batch in train_loader:
        # Put the x and y datapoints in
        # the current batch on device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # clear previous gradients
        optimiser.zero_grad()

        # Pass the X_batch through the model
        logits = model(X_batch)

        # Calculate the loss. Returns the average loss
        # for that batch
        loss = criterion(logits, y_batch)

        # Calculate the gradients through back prop
        loss.backward()

        # Update mode weights using those gradients
        optimiser.step()

        # Number of samples being processed
        batch_size = X_batch.size(0)

        # This gives the total loss for this batch.
        # Since loss = total_loss / batch_size
        # loss *  X_batch_size(0) = total_loss
        total_loss += loss.item() * X_batch.size(0)

        # Update total_samples
        total_samples += batch_size

    # Total loss
    return total_loss / total_samples


# Helps select model
def build_model(
    model_name: str,
    num_classes: int,
    dropout: float,
) -> nn.Module:
    if model_name == "cnn_baseline_v1":
        return CNNBaselineV1(num_classes=num_classes)

    if model_name == "cnn_baseline_v2":
        return CNNBaselineV2(
            num_classes=num_classes,
            dropout=dropout,
        )

    raise ValueError(f"Unknown model name: {model_name}")


# ---------------------------------------------------------------------
#                             CLI Parser
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    # Define parser
    parser = argparse.ArgumentParser("Execute training for CNN baseline v1.")

    # Add arguments
    parser.add_argument("--train-set-dir", type=Path, default=Path("data/splits/train"))

    parser.add_argument("--val-set-dir", type=Path, default=Path("data/splits/val"))

    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Choose learning rate"
    )

    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument(
        "--epochs", type=int, default=25, help="Number of epochs to train the model"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        choices=["cnn_baseline_v1", "cnn_baseline_v2"],
        default="cnn_baseline_v2",  # Defaults to better model
        help="Which CNN model architecture to train.",
    )

    parser.add_argument(
        "--dropout", type=float, default=0.3, help="Choose dropout probability"
    )

    parser.add_argument(
        "--model-output-path",
        type=Path,
        default=None,
        help="Where to save the best model checkpoint.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
#                        Main Training Scipt
# ---------------------------------------------------------------------


def main() -> None:

    logging.basicConfig(level="INFO")

    logger.info("Building parser...")

    args = parse_args()

    logger.info("Parser built")

    # Create train and val sets
    train_set = ECGDataset(args.train_set_dir)
    val_set = ECGDataset(args.val_set_dir)

    # Create dataloaders
    # (May add workers later)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Use GPU is possible, else use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Log specific GPU hardware if cuda is active
    if device.type == "cuda":
        logger.info("Using GPU hardware: %s", torch.cuda.get_device_name(0))

    # Define model. Model and data must be on the same device.
    # moving the model to device means putting its weights on device
    model = build_model(
        model_name=args.model_name, num_classes=NUM_CLASSES, dropout=args.dropout
    ).to(device)

    # Class weights to ensure minority classes are taken into
    # consideration
    class_weights = compute_class_weights(train_set, device)

    # Criterion used to calculate the loss, ensuring the
    # class weights are used.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    logger.info("class weights: %s", class_weights)

    # model.parameters() are all the learnable parameters in our model.
    # We give this to the optimiser so it knows what to update.
    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    logger.info(
        "device: %s | model: %s | criterion: %s | optimiser: %s",
        device,
        model,
        criterion,
        optimiser,
    )

    # starting macro_f1
    best_macro_f1 = 0

    # Define where we want the final learned model weights
    # to be stored and make the directory.
    
    if args.model_output_path is None:
        model_output_path = Path(f"artifacts/models/{args.model_name}.pt")
    else:
        model_output_path = args.model_output_path

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    # Note parent.mkdir creates the directory one level above the
    # final file name.

    logger.info("path %s created", model_output_path)

    # Run num_epochs epochs
    logger.info("Intialising training for %s epochs...", args.epochs)
    for epoch in range(1, args.epochs + 1):
        # Update model weights and return train_loss
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimiser=optimiser,
            device=device,
        )

        val_metrics = evaluate(
            model=model, split_loader=val_loader, criterion=criterion, device=device
        )

        macro_f1 = val_metrics["macro_f1"]

        logger.info(
            """
            epoch: %s| train loss: %.4f | val macro f1: %.4f |
            val acc: %.4f | val loss: %.4f
            """,
            epoch,
            train_loss,
            macro_f1,
            val_metrics["accuracy"],
            val_metrics["loss"],
        )

        # If this epoch gives the best macro F1 so far,
        # save the model weights.
        if macro_f1 > best_macro_f1:
            # Update the best_val_loss
            best_macro_f1 = macro_f1
            # Save the models current weights.
            torch.save(model.state_dict(), model_output_path)

            # only log metrics when a new highest macro_f1 has been found
            log_per_class_metrics(val_metrics)
            log_confusion_matrix(val_metrics)

    logger.info("best_macro_f1: %s", best_macro_f1)
    logger.info("saved best model weights to: %s", model_output_path)


if __name__ == "__main__":
    main()
