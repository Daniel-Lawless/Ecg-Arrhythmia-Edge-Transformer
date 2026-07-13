import argparse
import logging
from pathlib import Path
from typing import TypedDict
import random

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from ecg_arrhythmia.data.ecg_sequence_dataset import (
    LABEL_TO_INDEX,
    ECGSequenceDataset,
)
from ecg_arrhythmia.models.sequence_transformer import ECGSequenceTransformer

logger = logging.getLogger(__name__)

NUM_CLASSES = len(LABEL_TO_INDEX)

# Allows us to convert index to label.
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}
CLASS_LABELS = [INDEX_TO_LABEL[index] for index in range(NUM_CLASSES)]
CLASS_INDICES = list(range(NUM_CLASSES))


# ---------------------------------------------------------------------
#                            Define Types
# ---------------------------------------------------------------------


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

    # Log the per-class values returned by sklearn's classification_report.
    for label, class_metrics in evaluation_metrics["per_class"].items():
        logger.info(
            "%s | precision: %.4f | recall: %.4f | f1: %.4f | "
            "total_class_count: %s",
            label,
            class_metrics["precision"],
            class_metrics["recall"],
            class_metrics["f1"],
            class_metrics["total_class_count"],
        )


def log_confusion_matrix(evaluation_metrics: EvaluationMetrics) -> None:
    logger.info("confusion matrix: rows=true labels, columns=predicted labels")
    logger.info("labels: %s", CLASS_LABELS)

    for label, row in zip(
        CLASS_LABELS,
        evaluation_metrics["confusion_matrix"],
        strict=True,
    ):
        logger.info("%s: %s", label, row)


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
        for X_batch, rr_batch, y_batch in split_loader:
            # Put the ECG sequences, RR sequences, and labels on device.
            X_batch = X_batch.to(device)
            rr_batch = rr_batch.to(device)
            y_batch = y_batch.to(device)

            # The sequence model always uses both ECG and RR inputs.
            logits = model(X_batch, rr_batch)

            # Calculate loss and predictions.
            loss = criterion(logits, y_batch)
            predictions = logits.argmax(dim=1)

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Keep tensors during the loop and concatenate once at the end.
            all_true_tensors.append(y_batch.cpu())
            all_predicted_tensors.append(predictions.cpu())

    if total_samples == 0:
        raise ValueError("Cannot evaluate an empty dataset")

    all_true_labels = torch.cat(all_true_tensors).numpy()
    all_predicted_labels = torch.cat(all_predicted_tensors).numpy()

    # This time unlike in the cnn_training.py file, we can get all the 
    # metrics calculated using sklearns built in classification_report
    report = classification_report(
        all_true_labels,
        all_predicted_labels,
        labels=CLASS_INDICES,
        target_names=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )

    # Tells Pylance that report is definitley a dictionary at this point, 
    assert isinstance(report, dict)

    # Then we can populate the per class metrics using this report.
    per_class: dict[str, ClassMetrics] = {
        label: {
            "precision": round(float(report[label]["precision"]), 4),
            "recall": round(float(report[label]["recall"]), 4),
            "f1": round(float(report[label]["f1-score"]), 4),
            "total_class_count": int(report[label]["support"]),
        }
        for label in CLASS_LABELS
    }

    # Calculate the confusion matrix
    confusion = confusion_matrix(
        all_true_labels,
        all_predicted_labels,
        labels=CLASS_INDICES,
    )

    return {
        "loss": total_loss / total_samples,
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


# ---------------------------------------------------------------------
#                     Helpers For Training
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_class_weights(
    dataset: ECGSequenceDataset,
    device: torch.device,
    weighting_method: str = "inverse",
    max_class_weight: float = 10.0,
) -> torch.Tensor:
    labels = torch.tensor(dataset.y_indices, dtype=torch.long)

    # Count how often each label appears
    class_counts = torch.bincount(
        labels,
        minlength=NUM_CLASSES,
    ).float()

    # Sum up each bin to get all samples
    total_samples = class_counts.sum()

    # The more rare a class is the larger its weight
    inverse_weights = total_samples / (
        NUM_CLASSES * class_counts.clamp_min(1.0)
    )

    # Keep it as is
    if weighting_method == "inverse":
        class_weights = inverse_weights

    # Makes them less strict
    elif weighting_method == "sqrt_inverse":
        class_weights = torch.sqrt(inverse_weights)

    # Limit the weights to a certain max_weight
    elif weighting_method == "capped_inverse":
        class_weights = inverse_weights.clamp(max=max_class_weight)

    else:
        raise ValueError(
            f"Unknown class weighting method: {weighting_method}"
        )

    logger.info("class counts: %s", class_counts)
    logger.info("class weights: %s", class_weights)

    return class_weights.to(device)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    # Put the model in training mode.
    model.train()

    total_loss = 0.0
    total_samples = 0

    for X_batch, rr_batch, y_batch in train_loader:
        # Put the current batch on device.
        X_batch = X_batch.to(device)
        rr_batch = rr_batch.to(device)
        y_batch = y_batch.to(device)

        # Clear gradients left over from the previous batch.
        optimiser.zero_grad()

        # The model receives a sequence of ECG windows and a matching
        # sequence of RR feature vectors.
        logits = model(X_batch, rr_batch)

        # Calculate the average loss for this batch.
        loss = criterion(logits, y_batch)

        # Calculate gradients through backpropagation and update the weights.
        loss.backward()
        optimiser.step()

        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Cannot train on an empty dataset")

    return total_loss / total_samples


# ---------------------------------------------------------------------
#                             CLI Parser
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CNN + RR Transformer ECG sequence model."
    )

    # Add our command line argument options
    parser.add_argument(
        "--train-set-dir",
        type=Path,
        default=Path("data/splits_sequences_matched/train"),
    )

    parser.add_argument(
        "--val-set-dir",
        type=Path,
        default=Path("data/splits_sequences_matched/val"),
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate used by the Adam optimiser.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of epochs to train the model.",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability used by the model.",
    )

    parser.add_argument(
        "--model-output-path",
        type=Path,
        default=Path("artifacts/models/ecg_sequence_transformer.pt"),
        help="Where to save the best model checkpoint.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of epochs where macro f1 doesn't improve before training is stopped"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of Transformer encoder layers.",
    )

    parser.add_argument(
        "--class-weighting",
        choices=[
            "inverse",
            "sqrt_inverse",
            "capped_inverse",
        ],
        default="inverse",
    )

    parser.add_argument(
        "--max-class-weight",
        type=float,
        default=10.0,
        help="Maximum class weight when using capped_inverse.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
#                         Main Training Script
# ---------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level="INFO")

    logger.info("Building parser...")
    args = parse_args()
    logger.info("Parser built")

    set_seed(args.seed)

    # Create train and validation sequence datasets.
    train_set = ECGSequenceDataset(args.train_set_dir)
    val_set = ECGSequenceDataset(args.val_set_dir)

    # Create DataLoaders. The training loader shuffles sequence rows,
    # but the order of beats inside each individual sequence is unchanged.
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

    # Use a GPU if CUDA is available, otherwise use the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        logger.info("Using GPU hardware: %s", torch.cuda.get_device_name(0))

    # Build the sequence model and move its learnable parameters to device.
    model = ECGSequenceTransformer(
        num_classes=NUM_CLASSES,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # Use target-label frequencies from the training sequence dataset to
    # give minority classes larger contributions to the loss.
    class_weights = compute_class_weights(
        dataset=train_set,
        device=device,
        weighting_method=args.class_weighting,
        max_class_weight=args.max_class_weight,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # model.parameters() gives Adam all learnable parameters from the CNN,
    # RR encoder, input projection, positional embeddings, Transformer,
    # and classifier.
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    logger.info("class weights: %s", class_weights)
    logger.info(
        "device: %s | model: %s | criterion: %s | optimiser: %s",
        device,
        model,
        criterion,
        optimiser,
    )

    model_output_path = args.model_output_path
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("model output path: %s", model_output_path)

    # Use negative infinity so the first completed epoch is always 
    # the initial best checkpoint.
    best_macro_f1 = float("-inf")

    epochs_without_improvement = 0
    patience = args.patience

    logger.info("Initialising training for %s epochs...", args.epochs)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimiser=optimiser,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            split_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        macro_f1 = val_metrics["macro_f1"]

        logger.info(
            "epoch: %s | train loss: %.4f | val macro f1: %.4f | "
            "val acc: %.4f | val loss: %.4f",
            epoch,
            train_loss,
            macro_f1,
            val_metrics["accuracy"],
            val_metrics["loss"],
        )

        # Save the weights from the epoch with the highest validation macro F1.
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                model_output_path,
            )

            logger.info("New best validation macro F1")
            log_per_class_metrics(val_metrics)
            log_confusion_matrix(val_metrics)

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info(
                "early stopping after %d epochs without improvement",
                patience,
            )
            break

    logger.info("best_macro_f1: %.4f", best_macro_f1)
    logger.info("saved best model weights to: %s", model_output_path)


if __name__ == "__main__":
    main()
