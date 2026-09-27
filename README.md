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

Run the commands below from the `prediction_wrc_c_diffusion_optimized/` directory. Check the input, checkpoint, and output paths in each script before running it on a different machine.

### Datasets

The classification dataset contains 63,468 promoter sequences grouped into five strength classes. The class-labeled sequences are stored in `dataset/wrc_class5_63468.txt`, and the merged data are stored in `dataset/merged_dataset63468.csv`. Prepare the five-fold datasets using:

```bash
python dataset/data_processed_5fold.py
python dataset/regression_63468_5fold.py
```

The processed datasets are stored in `dataset/wrc_processed_5fold/kfold5_new/` and `dataset/wrc_regression_63468_5fold/`, respectively.

### Model Training

The promoter strength prediction models can be trained with:

```bash
python prediction/train_optimized_predict.py
python prediction/train_DNABert2_MLP.py
```

The conditional diffusion model can be trained with:

```bash
python Diffusion_to_optimized/train_diff_all.py
```

### Model Testing

Generate promoter sequences, optimize them, and evaluate the results using:

```bash
python Diffusion_to_optimized/gen_all_10sets.py
python Diffusion_to_optimized/optimized_all_10sets.py
python Diffusion_to_optimized/evaluate_all_10sets.py
```

The repository structure shown does not yet include a separate test script for the prediction models. Their saved checkpoints are under `prediction/checkpoint_prediction/` and `prediction/checkpoint_dnabert2/`.
