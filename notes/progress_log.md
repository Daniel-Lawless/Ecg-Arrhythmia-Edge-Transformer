# Progress Log

## Milestone 1 — MIT-BIH data loading and beat extraction

Implemented:

- Added support for loading ECG records from the MIT-BIH Arrhythmia Database using WFDB.
- Loaded both ECG signal channels and expert beat annotations for each record.
- Added signal-channel selection, preferring `MLII` when available and falling back to the first available channel otherwise.
- Implemented beat extraction around annotation locations.
- Each beat is represented as a fixed 240-sample ECG window:
  - 90 samples before the annotation
  - 150 samples after the annotation
- Filtered out annotations that do not represent heartbeat classes.

Key lesson:

- ECG data needs careful preprocessing before it can be used by a neural network.
- Beat-level classification requires each annotation to be converted into a fixed-size input window.
- Keeping the window size fixed makes the later PyTorch dataset and CNN models much simpler.
- Some beats near the start or end of a record cannot produce a complete window and must be skipped.

## Milestone 2 — AAMI label mapping and dataset building

Implemented:

- Added AAMI-style label mapping to group raw MIT-BIH beat annotations into higher-level arrhythmia classes.
- Mapped raw annotations into:
  - `N`
  - `S`
  - `V`
  - `F`
  - `Q`
- Excluded the `Q` class during dataset build due to low support and ambiguity.
- Built the processed dataset using 4 classes:
  - `N`
  - `S`
  - `V`
  - `F`
- Saved processed arrays to disk:
  - `X.npy`
  - `y.npy`
  - `patient_ids.npy`
  - `record_segments.json`
- Stored record-level metadata including:
  - record ID
  - patient ID
  - selected ECG lead
  - start index
  - end index
  - number of extracted beats

Key lesson:

- Raw ECG annotation labels are too fragmented and imbalanced to use directly.
- AAMI grouping makes the classification problem more manageable while still preserving clinically meaningful arrhythmia categories.
- Saving `patient_ids.npy` is important because the dataset must later be split by patient, not by individual beat.
- Storing record metadata makes the processed dataset easier to inspect and debug.

## Milestone 3 — Patient-level train/validation/test splitting

Implemented:

- Created patient-level train, validation, and test splits.
- Ensured that no patient appears in more than one split.
- Added split validation to prevent patient leakage.
- Used Monte Carlo sampling to search for a split that approximately preserves:
  - target train/validation/test ratios
  - class distribution across splits
- Saved each split separately with its own:
  - `X.npy`
  - `y.npy`
  - `patient_ids.npy`
  - metadata

Key lesson:

- Patient-level splitting is essential for ECG classification.
- Random beat-level splitting would leak patient-specific ECG morphology into multiple splits and produce skewed, overly optimistic results.
- The split cannot be perfectly balanced because patients contain different numbers and types of beats.
- Monte Carlo split search gives a practical way to find a reasonable patient-safe split.

## Milestone 4 — Testing and continuous integration

Implemented:

- Added unit tests for beat extraction.
- Added unit tests for AAMI label mapping.
- Added unit tests for ECG record loading and signal-channel selection.
- Added unit tests for dataset building.
- Added unit tests for patient-level splitting.
- Added integration tests using real MIT-BIH records.
- Added GitHub Actions CI.
- Configured CI to run automated checks on pushes and pull requests.

Key lesson:

- Tests are especially useful in data pipelines because small preprocessing bugs can silently corrupt the whole dataset.
- Unit tests make individual behaviours easier to verify without depending on the full MIT-BIH database.
- Integration tests confirm that the real WFDB/MIT-BIH pipeline works end-to-end.
- CI makes the repository safer to refactor because tests are run automatically.

## Milestone 5 — PyTorch dataset and CNN Baseline V1

Implemented:

- Created a PyTorch `ECGDataset`.
- Loaded processed split data from disk.
- Converted string labels into integer class indices:
  - `N -> 0`
  - `S -> 1`
  - `V -> 2`
  - `F -> 3`
- Returned ECG windows in the shape expected by `Conv1d`:

```text
(channels, sequence_length) = (1, 240)
```

- Implemented `CNNBaselineV1`, a simple 1D CNN with:
  - convolution layers
  - ReLU activations
  - max pooling
  - adaptive average pooling
  - final linear classifier

Key lesson:

- `Conv1d` expects the channel dimension before the sequence dimension, so each ECG window must be reshaped from `(240,)` to `(1, 240)`.
- A simple CNN is a useful first baseline because it proves the full pipeline works from processed ECG windows to class predictions.
- The first model does not need to be perfect; it mainly establishes a measurable reference point.

## Milestone 6 — Weighted training and evaluation pipeline

Implemented:

- Added a reusable CNN training script.
- Added weighted `CrossEntropyLoss` to handle class imbalance.
- Computed class weights from the training set.
- Added validation after each epoch.
- Saved the best model checkpoint based on validation macro F1.
- Added evaluation metrics:
  - loss
  - accuracy
  - macro F1
  - per-class precision
  - per-class recall
  - per-class F1
  - confusion matrix
- Added structured confusion matrix output for easier JSON inspection.

Key lesson:

- Accuracy is misleading for this dataset because `N` dominates the class distribution.
- Macro F1 is a better main metric because it gives each class equal importance.
- Per-class metrics and confusion matrices are necessary because the model can appear reasonable overall while completely failing minority arrhythmia classes.
- Weighted loss helps force the model to pay attention to rare classes, but it can also cause overprediction of minority classes.

## Milestone 7 — CNN Baseline V1 test evaluation

Implemented:

- Trained and evaluated `CNNBaselineV1`.
- Saved the model checkpoint to:

```text
artifacts/models/cnn_baseline_v1.pt
```

- Saved test metrics to:

```text
artifacts/results/cnn_baseline_v1_test_metrics.json
```

Test results:

| Metric | Value |
|---|---:|
| Test loss | 1.2953 |
| Test accuracy | 0.7177 |
| Test macro F1 | 0.2807 |

Per-class test F1:

| Class | F1 |
|---|---:|
| N | 0.8372 |
| S | 0.0287 |
| V | 0.2450 |
| F | 0.0120 |

Key lesson:

- CNN V1 learned the majority `N` class well but struggled heavily with minority classes.
- `V` was detected better than `S` and `F`, but still had poor precision.
- `S` and `F` were almost unusable in the first baseline.

## Milestone 8 — CNN Baseline V2 and shared model training

Implemented:

- Added `CNNBaselineV2`, a stronger CNN architecture.
- Increased the number of convolution channels.
- Added `BatchNorm1d` after convolution layers.
- Added dropout before the final classifier.
- Kept the same input/output interface as V1:

```text
Input:  (batch_size, 1, 240)
Output: (batch_size, 4)
```

- Refactored training so both CNN models can be trained using the same script.
- Added `--model-name` support for selecting:
  - `cnn_baseline_v1`
  - `cnn_baseline_v2`

Example commands:

```bash
python -m ecg_arrhythmia.training.cnn_training --model-name cnn_baseline_v1
python -m ecg_arrhythmia.training.cnn_training --model-name cnn_baseline_v2
```

Best validation macro F1:

| Model | Best validation macro F1 |
|---|---:|
| CNN Baseline V1 | 0.4892 |
| CNN Baseline V2 | 0.4919 |

Key lesson:

- Keeping the same model input/output contract made it easy to reuse the same training loop.
- Refactoring the training script avoided duplicating hundreds of lines of code.
- BatchNorm and dropout improved the model without making the architecture too large for future edge deployment.

## Milestone 9 — Shared CNN evaluation and CNN Baseline V2 test results

Implemented:

- Refactored CNN evaluation so both V1 and V2 can be evaluated using the same script.
- Added model-specific checkpoint loading.
- Added model-specific metrics output paths.
- Evaluated both CNN baselines on the held-out test set.

Example commands:

```bash
python -m ecg_arrhythmia.evaluation.evaluate_cnn --model-name cnn_baseline_v1
python -m ecg_arrhythmia.evaluation.evaluate_cnn --model-name cnn_baseline_v2
```

Saved outputs:

```text
artifacts/results/cnn_baseline_v1_test_metrics.json
artifacts/results/cnn_baseline_v2_test_metrics.json
```

Test results:

| Model | Test loss | Test accuracy | Test macro F1 |
|---|---:|---:|---:|
| CNN Baseline V1 | 1.2953 | 0.7177 | 0.2807 |
| CNN Baseline V2 | 1.0929 | 0.7121 | 0.3256 |

Per-class test F1:

| Class | CNN V1 F1 | CNN V2 F1 | Change |
|---|---:|---:|---:|
| N | 0.8372 | 0.8268 | -0.0104 |
| S | 0.0287 | 0.0537 | +0.0250 |
| V | 0.2450 | 0.4118 | +0.1668 |
| F | 0.0120 | 0.0100 | -0.0020 |

Key lesson:

- CNN V2 is the stronger CNN baseline overall.
- Test accuracy decreased slightly, but macro F1 improved from `0.2807` to `0.3256`.
- This is a better trade-off for an imbalanced arrhythmia classification task.
- The largest improvement came from class `V`, where F1 increased from `0.2450` to `0.4118`.
- `S` improved slightly but remains weak.
- `F` remains extremely poor and difficult to judge because the test set contains very few `F` examples.

## Milestone 10 — Per-beat normalisation experiment

Implemented:

- Added optional per-beat z-score normalisation during beat extraction.
- Each beat window can now be normalised using its own mean and standard deviation:

```text
normalised_beat = (beat - beat.mean()) / beat.std()
```

- Kept normalisation optional so the original raw-signal baseline remains reproducible.
- Saved normalised processed data separately from the original dataset.
- Created separate normalised train/validation/test splits.
- Updated training and evaluation scripts so models can be trained and evaluated using custom split directories and checkpoint paths.
- Trained and evaluated normalised versions of both CNN baselines.

saved outputs to 
```text
artifacts/models/cnn_baseline_v1_normalised.pt
artifacts/models/cnn_baseline_v2_normalised.pt
artifacts/results/cnn_baseline_v1_normalised_test_metrics.json
artifacts/results/cnn_baseline_v2_normalised_test_metrics.json
```

Test results:

| Model | Normalised? | Test loss | Test accuracy | Test macro F1 |
|:---:|:---:|:---:|:---:|:---:|
| CNN Baseline V1 | No | 1.2953 | 0.7177 | 0.2807 |
| CNN Baseline V1 | Yes | 0.9969 | 0.8684 | 0.3737 |
| CNN Baseline V2 | No | 1.0929 | 0.7121 | 0.3256 |
| CNN Baseline V2 | Yes | 1.1696 | 0.7349 | 0.3131 |

Per-class test F1:

| Class | V1 raw | V1 normalised | V2 raw | V2 normalised |
|:---:|:---:|:---:|:---:|:---:|
| N | 0.8372 | 0.9298 | 0.8268 | 0.8406 |
| S | 0.0287 | 0.0049 | 0.0537 | 0.0404 |
| V | 0.2450 | 0.5600 | 0.4118 | 0.3713 |
| F | 0.0120 | 0.0000 | 0.0100 | 0.0000 |

key lesson:
- Per-beat normalisation significantly improved CNN Baseline V1 overall.
- V1 macro F1 increased from 0.2807 to 0.3737.
- The largest improvement came from class V, where F1 increased from 0.2450 to 0.5600.
- Normalisation also improved V1 accuracy from 0.7177 to 0.8684.
- CNN Baseline V2 did not benefit from normalisation overall; macro F1 decreased from 0.3256 to 0.3131.
- S remains very weak across all experiments, suggesting that morphology-only beat windows are not enough for supraventricular beat detection. RR intervals can help here because supraventricular beats are often reflected more clearly in rhythm/timing patterns than in QRS morphology. They can occur prematurely with a shortened previous RR interval, while their QRS morphology may still look close to normal, which is why they are often predicted as N. So RR features could provide context that the CNN is currently missing.
- F remains difficult to interpret because the test set contains only a very small number of F beats.
- The next improvement should add rhythm/context information, such as RR interval features. Since the goal is real-time edge compute I'll add previous RR intervals only

## Milestone 11 — RR Interval Features and CNN Baseline V2 RR

### Implemented

Added RR interval features to the ECG preprocessing and CNN training pipeline.

Each beat-level sample can now include:

- the 240-sample ECG beat window
- RR timing features:
  - previous RR interval in seconds
  - RR ratio relative to recent rhythm

Updated the project so RR features flow through:

- beat extraction
- dataset building
- dataset loading and validation
- patient-wise train/validation/test splitting
- PyTorch `ECGDataset`
- CNN training
- CNN evaluation

Added a new model:

- `CNNBaselineV2RR`

This model combines ECG morphology features learned by the CNN with RR timing features before classification.

### Comparison Tested

To isolate the effect of RR intervals, the key comparison is:

| Model | RR Features? | Normalised? |
|:---:|:---:|:---:|
| CNN Baseline V2 | No | No |
| CNN Baseline V2 RR | Yes | No |

This comparison keeps normalisation fixed and only changes whether RR interval features are provided.

### Test Results

| Model | Test Loss | Test Accuracy | Test Macro F1 |
|:---:|:---:|:---:|:---:|
| CNN Baseline V2 | 1.0929 | 0.7121 | 0.3256 |
| CNN Baseline V2 + RR | 0.7311 | 0.8014 | 0.4113 |

### Per-Class F1

| Class | CNN V2 | CNN V2 + RR | Change |
|:---:|:---:|:---:|:---:|
| N | 0.8268 | 0.8936 | +0.0668 |
| S | 0.0537 | 0.2397 | +0.1860 |
| V | 0.4118 | 0.5097 | +0.0979 |
| F | 0.0100 | 0.0023 | -0.0077 |

### Key Findings

Adding RR interval features improved overall model performance.

Macro F1 increased from `0.3256` to `0.4113`, showing that the RR-enhanced model performed better across classes rather than only improving accuracy on the majority class.

The largest improvement was on the `S` class:

| Metric | CNN V2 | CNN V2 + RR |
|:---:|:---:|:---:|
| Precision | 0.0347 | 0.3096 |
| Recall | 0.1186 | 0.1955 |
| F1 | 0.0537 | 0.2397 |

This supports the original motivation outlined yesterday for adding RR intervals. Supraventricular ectopic beats can look morphologically similar to normal beats, so timing context helps the model identify them.

### Confusion Matrix Notes

For true `S` beats:

| Model | Predicted N | Predicted S | Predicted V | Predicted F |
|:---:|:---:|:---:|:---:|:---:|
| CNN V2 | 75 | 37 | 131 | 69 |
| CNN V2 + RR | 45 | 61 | 177 | 29 |

The RR model correctly identified more `S` beats, increasing correct `S` predictions from `37` to `61`.

It also reduced the number of `S` beats incorrectly predicted as `N` or `F`.

### Conclusion

`CNNBaselineV2RR` is the strongest CNN baseline so far.

The results show that adding RR interval features improves performance over the morphology-only CNN V2 baseline, especially for the minority `S` class.

Current best CNN model:

### Current Best Baseline

| Model | Test macro F1 |
|:---:|:---:|
| CNN V2 without RR | 0.3256 |
| CNN V2 + RR normalised | 0.3644 |
| CNN V1 + normalised | 0.3737 |
| CNN V2 + RR unnormalised | 0.4113 |

### Next steps

The next modelling stage should compare this RR-enhanced CNN against the transformer/sequence model, since the transformer should be able to use
surrounding beat context more naturally.
