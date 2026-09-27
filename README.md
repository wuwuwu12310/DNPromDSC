# DNPromDSC: *De Novo* Promoter Design for Strength Control Using Conditional Diffusion and Sequence Optimization

## Overview

`DNPromDSC` is a deep learning framework for the de novo design of 80-bp *Saccharomyces cerevisiae* promoter sequences with desired strength classes. It integrates promoter strength prediction, conditional denoising diffusion, and dynamic promoter optimization to generate promoters with controlled strengths while maintaining natural sequence characteristics.

The promoter strength prediction model combines raw sequence information with chaos game representation (CGR) and pseudo dinucleotide composition (PseDNC) features to predict strength classes and extract class-specific sequence information. The conditional diffusion model uses this information to generate promoters for the desired strength class. The dynamic promoter optimization model then refines the generated sequences through local nucleotide mutation, masked sequence regeneration, and conditional diffusion sampling, with candidate selection based on predicted strength and sequence composition.

## Description

The project includes the following directories and core files:

- `dataset/` contains the promoter data, five-fold datasets, and preprocessing scripts `data_processed_5fold.py` and `regression_63468_5fold.py`.
- `prediction/` contains the promoter strength prediction models and their training scripts: `model_optimized_predict.py`, `train_optimized_predict.py`, `model_dnabert2_mlp.py`, and `train_DNABert2_MLP.py`. Its `checkpoint_prediction/` and `checkpoint_dnabert2/` directories contain the corresponding checkpoints.
- `prior_condition/` contains `methods.py`, `model_optimized_predict.py`, and saved weights for extracting class-specific conditioning information from the prediction model.
- `Diffusion_to_optimized/` contains `train_diff_all.py` for diffusion training, `gen_all_10sets.py` for sequence generation, `optimized_all_10sets.py` for sequence optimization, and `evaluate_all_10sets.py` for evaluation. It also contains the predictor model definitions and the `results_diff/` directory.

## System Requirements

`DNPromDSC` was implemented, trained, and tested using Python 3.8.20 and PyTorch 2.4.1 with CUDA 12.1 on an NVIDIA RTX 4090 GPU. The provided package list includes the following libraries used by the preprocessing, prediction, generation, and evaluation scripts:

```text
Python         3.8.20
torch          2.4.1
transformers   4.32.1
scikit-learn   1.3.0
pandas         2.0.3
numpy          1.24.3
tqdm           4.65.0
scipy          1.11.1
```

For complete evaluation, install `biopython` and `Levenshtein` as well; `evaluate_all_10sets.py` uses them to calculate sequence alignment and edit-distance metrics. Their versions were not included in the provided package list. DNABERT-2 experiments require the local `bert_layers.py`, pretrained model and tokenizer files, and the corresponding trained checkpoints. Diffusion training also requires the saved class-position weights and base-preference arrays referenced in `train_diff_all.py`. Before running the scripts on another machine, update their configured data and checkpoint paths and select GPU indices available on that machine.

## Usage

### Datasets

The dataset used by `DNPromDSC` comprises 63,468 natural 80-bp *Saccharomyces cerevisiae* promoter sequences with measured expression strengths. The sequences and continuous strength values are stored in `dataset/merged_dataset63468.csv`; the five-class labels are stored with the sequences in `dataset/wrc_class5_63468.txt`.

Run `dataset/data_processed_5fold.py` to obtain five stratified classification folds in `dataset/wrc_processed_5fold/kfold5_new/`. Run `dataset/regression_63468_5fold.py` to obtain five continuous-strength folds in `dataset/wrc_regression_63468_5fold/` for DNABERT-2 regression training:

```bash
python dataset/data_processed_5fold.py
python dataset/regression_63468_5fold.py
```

### Model Training

We define the promoter strength prediction model in `prediction/model_optimized_predict.py`, where:

- The sequence feature extraction module uses convolutional layers and a Transformer encoder to process promoter sequences.
- The feature-processing and fusion modules combine sequence representations with chaos game representation (CGR) and pseudo dinucleotide composition (PseDNC) features for strength-class prediction.

Run `prediction/train_optimized_predict.py` to obtain five classification checkpoints (`best_f1.pth` through `best_f5.pth`) in its configured output directory. A separate DNABERT-2 regression model is defined in `prediction/model_dnabert2_mlp.py`. Run `prediction/train_DNABert2_MLP.py` to obtain `fold1/best_model.pth` through `fold5/best_model.pth` under `prediction/checkpoint_dnabert2/` for independent evaluation of designed promoters:

```bash
python prediction/train_optimized_predict.py
python prediction/train_DNABert2_MLP.py
```

Run `prior_condition/methods.py` with its required sequence-only classifier (`model_new_simple_gpt`) and five classifier checkpoints to obtain class-specific position weights and nucleotide preferences, including `class_position_weights_combined_5bins.npy` and `class_base_preference_from_mutagenesis.npy`. If these saved features are already available, they can be used directly. Run `Diffusion_to_optimized/train_diff_all.py` using the features and class-labeled sequences to obtain diffusion checkpoints (including `final.pt`) and `run_config.json` in its configured output directory:

```bash
python prior_condition/methods.py
python Diffusion_to_optimized/train_diff_all.py
```

### Model Testing

Run `Diffusion_to_optimized/gen_all_10sets.py` using a trained diffusion checkpoint to obtain ten independently generated sets of promoters and their generation summaries. Run `Diffusion_to_optimized/optimized_all_10sets.py` using the diffusion checkpoint and five classification predictor checkpoints to obtain ten optimized sets and optimization summaries. The optimization script generates its own candidates; it does not use the output of `gen_all_10sets.py` as input.

Run `Diffusion_to_optimized/evaluate_all_10sets.py` using the generated-sequence directory, DNABERT-2 model files, and five regression checkpoints to obtain predicted strengths, sequence metrics, and evaluation reports:

```bash
python Diffusion_to_optimized/gen_all_10sets.py
python Diffusion_to_optimized/optimized_all_10sets.py
python Diffusion_to_optimized/evaluate_all_10sets.py
```

Set the dataset, saved-feature, checkpoint, and output paths in these scripts to match your installation before running them. To regenerate the conditioning features, also provide `model_new_simple_gpt` and its sequence-only classifier checkpoints; they are not shown in the repository layout above. The classification training script saves `best_f*.pth`, while the provided prediction checkpoints are named `prediction_f*.pth`; verify the fold-to-file mapping before reuse. The repository does not currently include a separate test script for the prediction models.
