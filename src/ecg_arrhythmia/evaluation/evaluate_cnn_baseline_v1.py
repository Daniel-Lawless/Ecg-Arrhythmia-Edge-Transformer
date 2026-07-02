import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ecg_arrhythmia.data.ecg_dataset import ECGDataset
from ecg_arrhythmia.models.cnn_baseline import CNNBaseline
from ecg_arrhythmia.training.cnn_training import (
    INDEX_TO_LABEL,
    NUM_CLASSES,
    EvaluationMetrics,
    compute_class_weights,
    evaluate,
    log_confusion_matrix,
    log_per_class_metrics,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
#                    Helpers For Evaluate Logic
# ---------------------------------------------------------------------


def load_model(checkpoint_path: Path, device: torch.device) -> CNNBaseline:
    # Make empty model skelton to load weights into
    model = CNNBaseline(num_classes=NUM_CLASSES).to(device)

    # Load the weights learned from training
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Load weights into model
    model.load_state_dict(state_dict)

    return model


def build_confusion_matrix_rows(
    confusion_matrix: list[list[int]],
) -> list[dict[str, object]]:
    labels = [INDEX_TO_LABEL[index] for index in range(NUM_CLASSES)]

    return [
        {
            "true_label": true_label,
            "predictions": dict(zip(labels, row, strict=True)),
            "total": sum(row),
        }
        for true_label, row in zip(labels, confusion_matrix, strict=True)
    ]


def format_metrics_for_json(metrics: EvaluationMetrics) -> dict[str, object]:
    labels = [INDEX_TO_LABEL[index] for index in range(NUM_CLASSES)]

    return {
        "loss": np.round(metrics["loss"], 4),
        "accuracy": np.round(metrics["accuracy"], 4),
        "macro_f1": np.round(metrics["macro_f1"], 4),
        "per_class": metrics["per_class"],
        "confusion_matrix": {
            "description": "Rows are true labels; columns are predicted labels.",
            "labels": labels,
            "rows": build_confusion_matrix_rows(metrics["confusion_matrix"]),
        },
    }


def save_metrics(metrics: EvaluationMetrics, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf8") as file:
        json.dump(format_metrics_for_json(metrics), file, indent=4)


# ---------------------------------------------------------------------
#                             CLI Parser
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    # Creat parser object
    parser = argparse.ArgumentParser(description="Evaluate CNN baseline v1.")

    # Add command line arguments
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("artifacts/models/cnn_baseline_v1.pt"),
    )
    parser.add_argument(
        "--test-split-dir",
        type=Path,
        default=Path("data/splits/test"),
    )
    parser.add_argument(
        "--train-split-dir",
        type=Path,
        default=Path("data/splits/train"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/results/cnn_baseline_v1_test_metrics.json"),
    )
    parser.add_argument("--batch-size", type=int, default=64)

    return parser.parse_args()


# ---------------------------------------------------------------------
#                       Main Evaluate Logic
# ---------------------------------------------------------------------


def main() -> None:
    # Set logging level
    logging.basicConfig(level=logging.INFO)

    # Extract the command line arguments
    args = parse_args()

    # Extract device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        logger.info("device: CUDA | %s", torch.cuda.get_device_name(0))
    else:
        logger.info("device: %s", device)

    # Create and populate datasets objects.
    train_set = ECGDataset(args.train_split_dir)
    test_set = ECGDataset(args.test_split_dir)

    # Create test loader
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Load the model
    model = load_model(args.checkpoint_path, device)

    # Important note for future reference:
    # Use the same weighted loss setup as training, based on train class counts.
    class_weights = compute_class_weights(train_set, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    logger.info("class weights: %s", class_weights)

    # Evaluate model on test set.
    metrics = evaluate(
        model=model,
        split_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    logger.info(
        "test loss: %.4f | test acc: %.4f | test macro f1: %.4f",
        metrics["loss"],
        metrics["accuracy"],
        metrics["macro_f1"],
    )
    log_per_class_metrics(metrics)
    log_confusion_matrix(metrics)

    # Save metrics
    save_metrics(metrics, args.output_path)
    logger.info("saved test metrics to: %s", args.output_path)


if __name__ == "__main__":
    main()
