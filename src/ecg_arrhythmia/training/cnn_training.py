import logging
from pathlib import Path
from typing import TypedDict

import torch
from torch import nn
from torch.utils.data import DataLoader

from ecg_arrhythmia.data.ecg_dataset import LABEL_TO_INDEX, ECGDataset
from ecg_arrhythmia.models.cnn_baseline import CNNBaseline

logger = logging.getLogger(__name__)

NUM_CLASSES = len(LABEL_TO_INDEX)

# Allows us to convert index to label
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}


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


def evaluate(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device
) -> EvaluationMetrics:

    # Put model in evaluation mode.
    # dropout is disabled so all activations are used
    # Batch norm uses the running mean and variance
    # learned during training
    model.eval()
    # Total loss for val data
    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    # Initalise prediction classification as a torch tensor
    # with num_classes zeros. Each position stores the count
    # for one class so true_positives[2] means true positives
    # count for V
    true_positives = torch.zeros(NUM_CLASSES, device=device)
    false_negatives = torch.zeros(NUM_CLASSES, device=device)
    false_positives = torch.zeros(NUM_CLASSES, device=device)

    # Keep a running sum of how often a given class occurs in the dataset
    total_class_counts = torch.zeros(NUM_CLASSES, device=device)

    # no weights are updated or gradients calculated.
    # Back computation graph is not created.
    with torch.no_grad():
        # For each batch in val loader
        for X_batch, y_batch in val_loader:
            # put them on device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Pass x data through our model
            # Output: (batch_size, num_classes)
            logits = model(X_batch)

            # Calculate loss
            loss = criterion(logits, y_batch)

            # It goes through each row in logits and gives the index
            # of the largest value. i.e., for
            # # [-0.3, 1.1, 0.8, 2.7] it would return 3 and this
            # corresponds to class F. output shape (batch_size,),
            # a tensor of length batch_size
            predictions = logits.argmax(dim=1)

            # num of samples for this batch
            batch_size = X_batch.size(0)

            # Calculate total loss for this batch
            total_loss += loss.item() * batch_size

            # Update total samples
            total_samples += batch_size

            # predictions == y_batch will give a tensor of booleans.
            # we sum the True values, and extract the value from the
            # tensor using item()
            correct_predictions += (predictions == y_batch).sum().item()

            # Update the TP, FN, and FP for this batch
            for class_index in range(NUM_CLASSES):
                # Boolean mask for samples the model predicted as this class
                predicted_class = predictions == class_index

                # Boolean mask for samples whose true label is this class
                true_class = y_batch == class_index

                # TP: predicted this class, and the true label was this class.
                true_positives[class_index] += (predicted_class & true_class).sum()

                # FN: did not predict this class, but the true label was this class.
                false_negatives[class_index] += (~predicted_class & true_class).sum()

                # FP: predicted this class, but the true label was not this class.
                false_positives[class_index] += (predicted_class & ~true_class).sum()

                # update class counts for this batch
                total_class_counts[class_index] += true_class.sum()

    # Per-class precision =
    # correct predictions for this class / all predictions made as this class.
    precision = true_positives / (true_positives + false_positives + 1e-8)

    # Per-class recall =
    # correct predictions for this class / all samples that truly belong to this class.
    recall = true_positives / (true_positives + false_negatives + 1e-8)

    # Combines precision and recall
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)

    # Per class metrics
    per_class = {}

    # For each class index
    for class_index in range(NUM_CLASSES):
        # Get its label
        label = INDEX_TO_LABEL[class_index]

        # return the models precision, recall, and f1 on that class.
        # Also return how often that class appeared in the dataset.
        per_class[label] = {
            "precision": precision[class_index].item(),
            "recall": recall[class_index].item(),
            "f1": f1_scores[class_index].item(),
            "total_class_count": int(total_class_counts[class_index].item()),
        }

    # Return loss, accuracy, macro_f1, and metrics per class.
    return {
        "loss": total_loss / total_samples,
        "accuracy": correct_predictions / total_samples,
        # takes the mean of the num_classes f1_values and
        # extracts it from the tensor
        "macro_f1": f1_scores.mean().item(),
        "per_class": per_class,
    }


def main(num_epochs: int = 20) -> None:

    # Create train and val sets
    train_set = ECGDataset(Path("data/splits/train"))
    val_set = ECGDataset(Path("data/splits/val"))

    # Create dataloaders
    # (May add workers later)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=32,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=32,
        shuffle=False,
    )

    # Use GPU is possible, else use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model. Model and data must be on the same device.
    # moving the model to device means putting its weights on device
    model = CNNBaseline(num_classes=4).to(device)

    # Class weights to ensure minority classes are taken into
    # consideration
    class_weights = compute_class_weights(train_set, device)

    # Criterion used to calculate the loss, ensuring the
    # class weights are used.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    logger.info("class weights: %s", class_weights)

    # model.parameters() are all the learnable parameters in our model.
    # We give this to the optimiser so it knows what to update.
    optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)

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
    model_output_path = Path("artifacts/models/cnn_baseline.pt")
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    # Note parent.mkdir creates the directory one level above the
    # final file name.

    logger.info("path %s created", model_output_path)

    # Run num_epochs epochs
    logger.info("Intialising training for %s epochs...", num_epochs)
    for epoch in range(1, num_epochs + 1):
        # Update model weights and return train_loss
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimiser=optimiser,
            device=device,
        )

        # Calculate val_loss
        val_metrics = evaluate(
            model=model, val_loader=val_loader, criterion=criterion, device=device
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

    logger.info("best_macro_f1: %s", best_macro_f1)
    logger.info("saved best model weights to: %s", model_output_path)


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    main(num_epochs=25)
