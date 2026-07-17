import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from ecg_arrhythmia.data.ecg_dataset import ECGDataset
from ecg_arrhythmia.evaluation.evaluate_cnn import load_model, save_metrics
from ecg_arrhythmia.training.cnn_training import (
    EvaluationMetrics,
    compute_class_weights,
    evaluate,
    log_confusion_matrix,
    log_per_class_metrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
#                     Load Saved Beat Indices
# ---------------------------------------------------------------------


def load_test_global_indices(split_indices_path: Path) -> np.ndarray:
    """
    Load the original full-dataset indices used to create the CNN test split.
    """

    # Make sure the saved split index file exists.
    if not split_indices_path.exists():
        raise FileNotFoundError(
            f"No beat-level split index file found at {split_indices_path}"
        )

    # Load the .npz file containing the train, validation, and test indices.
    with np.load(split_indices_path) as split_indices:
        # We specifically need the indices used for the CNN test split.
        if "test_indices" not in split_indices:
            raise KeyError(f"{split_indices_path} does not contain test_indices")

        # These are the original row positions of the beats in the full dataset.
        return split_indices["test_indices"].astype(np.int64)


def load_sequence_target_indices(sequence_test_dir: Path) -> np.ndarray:
    """
    Load the original full-dataset index of the final target beat in
    every Transformer test sequence.
    """

    target_indices_path = sequence_test_dir / "target_indices.npy"

    if not target_indices_path.exists():
        raise FileNotFoundError(
            f"No sequence target indices found at {target_indices_path}"
        )

    # Each target index points back to the original beat that appears
    # at the end of a Transformer sequence.
    return np.load(target_indices_path).astype(np.int64)


# ---------------------------------------------------------------------
#                 Match Transformer Targets To CNN Rows
# ---------------------------------------------------------------------


def map_targets_to_cnn_test_rows(
    test_global_indices: np.ndarray,
    target_global_indices: np.ndarray,
) -> np.ndarray:
    """
    Convert original full-dataset target indices into local row positions
    inside the saved CNN test split.
    """

    # Each original beat should only appear once in the CNN test split.
    if len(np.unique(test_global_indices)) != len(test_global_indices):
        raise ValueError("CNN test_indices contains duplicate values")

    # Each Transformer sequence should also point to a unique target beat.
    if len(np.unique(target_global_indices)) != len(target_global_indices):
        raise ValueError("Sequence target_indices contains duplicate values")

    # Build a lookup from:
    # original full-dataset beat index -> local row in the CNN test split.
    #
    # Example:
    # test_global_indices = [10, 15, 22]
    #
    # global_to_local becomes:
    # {
    #     10: 0,
    #     15: 1,
    #     22: 2,
    # }
    global_to_local = {
        int(global_index): local_index
        for local_index, global_index in enumerate(test_global_indices)
    }

    # Check that every Transformer target beat is actually present
    # inside the original CNN test split.
    missing_targets = [
        int(global_index)
        for global_index in target_global_indices
        if int(global_index) not in global_to_local
    ]

    if missing_targets:
        raise ValueError(
            f"{len(missing_targets)} Transformer targets are missing from "
            f"the CNN test split. First missing indices: {missing_targets[:10]}"
        )

    # Convert the Transformer target indices into local CNN test-set rows.
    #
    # Keeping the same order as target_global_indices means the CNN samples
    # will line up exactly with the Transformer target samples.
    return np.array(
        [global_to_local[int(global_index)] for global_index in target_global_indices],
        dtype=np.int64,
    )


def validate_target_alignment(
    cnn_test_set: ECGDataset,
    cnn_local_indices: np.ndarray,
    sequence_test_dir: Path,
) -> None:
    """
    Confirm that the selected CNN labels exactly match the Transformer
    target labels in the same order.
    """

    sequence_labels_path = sequence_test_dir / "y.npy"

    if not sequence_labels_path.exists():
        raise FileNotFoundError(
            f"No Transformer target labels found at {sequence_labels_path}"
        )

    # Load the labels used by the Transformer test set.
    sequence_labels = np.load(sequence_labels_path).astype(str)

    # Select the corresponding labels from the CNN test dataset.
    cnn_labels = cnn_test_set.y[cnn_local_indices].astype(str)

    # Both models should be evaluated on the same number of target beats.
    if len(sequence_labels) != len(cnn_labels):
        raise ValueError(
            f"Target count mismatch: Transformer={len(sequence_labels)}, "
            f"CNN={len(cnn_labels)}"
        )

    # The labels must match exactly and in the same order.
    if not np.array_equal(sequence_labels, cnn_labels):
        # Return the positions where the two label arrays disagree.
        mismatch_positions = np.flatnonzero(sequence_labels != cnn_labels)

        raise ValueError(
            "CNN labels do not align with Transformer target labels. "
            f"Found {len(mismatch_positions)} mismatches. "
            f"First positions: {mismatch_positions[:10].tolist()}"
        )


# ---------------------------------------------------------------------
#                             CLI Parser
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an existing CNN checkpoint only on the exact beats "
            "used as targets by the matched Transformer test sequences."
        )
    )

    # Select which CNN architecture should be evaluated.
    parser.add_argument(
        "--model-name",
        choices=[
            "cnn_baseline_v1",
            "cnn_baseline_v2",
            "cnn_baseline_v2_rr",
        ],
        default="cnn_baseline_v2_rr",
    )

    # The original CNN training split is only used to recreate
    # the class weights used by the weighted loss.
    parser.add_argument(
        "--train-split-dir",
        type=Path,
        default=Path("data/splits/train"),
    )

    # The complete original CNN test split.
    parser.add_argument(
        "--test-split-dir",
        type=Path,
        default=Path("data/splits/test"),
    )

    # Contains the original full-dataset row indices used for the CNN splits.
    parser.add_argument(
        "--split-indices-path",
        type=Path,
        default=Path("data/splits/split_indices.npz"),
    )

    # Contains the exact target beats used by the matched Transformer test set.
    parser.add_argument(
        "--sequence-test-dir",
        type=Path,
        default=Path("data/splits_sequences_matched/test"),
    )

    parser.add_argument("--batch-size", type=int, default=64)

    # Optional custom path to the trained CNN weights.
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
    )

    # Optional custom path for the saved target-matched metrics.
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
    )

    # The RR model needs both ECG beat windows and RR features.
    parser.add_argument(
        "--use-rr-features",
        action="store_true",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
#                       Main Evaluate Logic
# ---------------------------------------------------------------------


def main() -> None:
    # Show INFO logs while the script is running.
    logging.basicConfig(level=logging.INFO)

    # Load command line arguments.
    args = parse_args()

    # The RR model must receive RR features.
    if args.model_name == "cnn_baseline_v2_rr" and not args.use_rr_features:
        raise ValueError("cnn_baseline_v2_rr requires --use-rr-features")

    # The non-RR CNN models do not accept RR features.
    if args.model_name != "cnn_baseline_v2_rr" and args.use_rr_features:
        raise ValueError("--use-rr-features can only be used with cnn_baseline_v2_rr")

    # Use the GPU when CUDA is available, otherwise use the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        logger.info("device: CUDA | %s", torch.cuda.get_device_name(0))
    else:
        logger.info("device: %s", device)

    # Load the original CNN train and test datasets.
    #
    # train_set is only needed to recreate the class weights.
    # cnn_test_set contains every beat in the original CNN test split.
    train_set = ECGDataset(args.train_split_dir)
    cnn_test_set = ECGDataset(args.test_split_dir)

    # Load the original global row indices used by the CNN test split.
    test_global_indices = load_test_global_indices(args.split_indices_path)

    # Load the original global row indices of the Transformer target beats.
    target_global_indices = load_sequence_target_indices(args.sequence_test_dir)

    # Convert the Transformer target indices into local rows
    # inside the saved CNN test dataset.
    cnn_local_indices = map_targets_to_cnn_test_rows(
        test_global_indices=test_global_indices,
        target_global_indices=target_global_indices,
    )

    # Confirm that both models will be evaluated against the exact same
    # target labels in the exact same order.
    validate_target_alignment(
        cnn_test_set=cnn_test_set,
        cnn_local_indices=cnn_local_indices,
        sequence_test_dir=args.sequence_test_dir,
    )

    # Create a view of the CNN test set containing only the beats
    # that were valid targets for the Transformer.
    #
    # This removes the first four possible beats from each record because
    # those beats do not have enough previous context for a length-5 sequence.
    target_test_set = Subset(
        cnn_test_set, # Full test datset
        # Contains the row numbers that correspond to the Transformer target beats.
        cnn_local_indices.tolist(),
    )

    # Load the matched CNN samples in batches.
    test_loader = DataLoader(
        target_test_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Use the provided checkpoint path, or build the default path
    # from the selected model name.
    checkpoint_path = (
        args.checkpoint_path
        if args.checkpoint_path is not None
        else Path("artifacts/models") / f"{args.model_name}.pt"
    )

    # Build the selected CNN architecture and load its saved weights.
    model = load_model(
        checkpoint_path=checkpoint_path,
        model_name=args.model_name,
        device=device,
    )

    # Recreate the class weights from the original CNN training split.
    #
    # These weights affect the reported test loss, but do not change
    # the predictions or classification metrics during evaluation.
    class_weights = compute_class_weights(
        train_set,
        device,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Log useful information so we can check that the matching worked.
    logger.info("checkpoint path: %s", checkpoint_path)
    logger.info("original CNN test samples: %d", len(cnn_test_set))
    logger.info("matched target samples: %d", len(target_test_set))
    logger.info(
        "excluded non-target samples: %d",
        len(cnn_test_set) - len(target_test_set),
    )
    logger.info("class weights: %s", class_weights)

    # Evaluate the existing CNN checkpoint on only the matched target beats.
    metrics: EvaluationMetrics = evaluate(
        model=model,
        split_loader=test_loader,
        criterion=criterion,
        device=device,
        use_rr_features=args.use_rr_features,
    )

    logger.info(
        "target-matched test loss: %.4f | test acc: %.4f | test macro f1: %.4f",
        metrics["loss"],
        metrics["accuracy"],
        metrics["macro_f1"],
    )

    # Log detailed class metrics and the confusion matrix.
    log_per_class_metrics(metrics)
    log_confusion_matrix(metrics)

    # Use the provided output path, or create a default filename
    # based on the chosen CNN model.
    output_path = (
        args.output_path
        if args.output_path is not None
        else Path("artifacts/results")
        / f"{args.model_name}_sequence_targets_test_metrics.json"
    )

    # Save the target-matched metrics as JSON.
    save_metrics(metrics, output_path)

    logger.info("saved target-matched metrics to: %s", output_path)


if __name__ == "__main__":
    main()