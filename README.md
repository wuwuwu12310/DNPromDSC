# DNPromDSC: *De Novo* Promoter Design for Strength Control Using Conditional Diffusion and Sequence Optimization

## Overview

`DNPromDSC` is a novel *de novo* promoter design approach of strength control based on a conditional diffusion model and sequence optimization. It is used to design *Saccharomyces cerevisiae* promoters with specified strength classes.

The architecture of `DNPromDSC` consists of a promoter strength prediction model, a conditional diffusion model, and a promoter sequence optimization algorithm. The strength prediction model integrates statistical and physicochemical feature processing with feature extraction by deep learning networks to assist the conditional generation and optimization processes. The conditional diffusion model employs the trained strength prediction model to provide conditional sequence features, integrates the Transformer encoder and U-Net through the cross-attention mechanism for conditional noise prediction, and is trained by combining position-weighted denoising and adjacent-class competitive losses. The sequence optimization algorithm is a modified reinforcement learning-based evolution optimization process, which integrates the trained strength prediction model for evaluations and uses the trained conditional diffusion model for regeneration actions.

The overall framework of `DNPromDSC`  is shown in the following figure.

![Overall framework of DNPromDSC](./Model_architecture.png)

## Description

The project contains four main folders and several core files, which are briefly described as follows.

- The folder `dataset` contains the promoter datasets and five-fold datasets.
- The files `dataset/data_processed_5fold.py` and `dataset/regression_63468_5fold.py` are the data preprocessing scripts.
- The folder `prediction` contains the promoter strength prediction models and training scripts.
- The files `prediction/model_optimized_predict.py` and `prediction/train_optimized_predict.py` define and train the promoter strength prediction model.
- The files `prediction/model_dnabert2_mlp.py` and `prediction/train_DNABert2_MLP.py` define and train the DNABERT-2-based model.
- The folders `prediction/checkpoint_prediction` and `prediction/checkpoint_dnabert2` contain the corresponding model checkpoints.
- The folder `prior_condition` contains the files and saved weights used to extract class-specific conditioning information from the prediction model.
- The folder `Diffusion_to_optimized` contains the diffusion training, sequence generation, optimization, and evaluation scripts.
- The file `Diffusion_to_optimized/train_diff_all.py` is the diffusion training script.
- The file `Diffusion_to_optimized/gen_all_10sets.py` is the sequence generation script.
- The file `Diffusion_to_optimized/optimized_all_10sets.py` is the sequence optimization script.
- The file `Diffusion_to_optimized/evaluate_all_10sets.py` is the evaluation script.
- The folder `Diffusion_to_optimized/results_diff` contains the diffusion training results.



## System Requirements

`DNPromDSC` was implemented, trained, and tested using Python 3.8.20 and PyTorch 2.4.1 with CUDA 12.1 on an NVIDIA RTX 4090 GPU. The main libraries used by the preprocessing, prediction, generation, optimization, and evaluation scripts are:

```text
Python         3.8.20
torch          2.4.1+cu121
transformers   4.46.3
scikit-learn   1.3.2
pandas         1.5.0
numpy          1.23.5
tqdm           4.64.1
scipy          1.9.3
biopython      1.83
Levenshtein    0.25.1
```

DNABERT-2 experiments require the pretrained model and tokenizer files in `prediction/DNABERT_2/` and the corresponding trained checkpoints.

## Usage

## 1. Dataset and Data Preprocessing

The folder `dataset` contains 63,468 natural 80-bp *Saccharomyces cerevisiae* promoter sequences with measured strengths. The file `merged_dataset63468.csv` contains the continuous-strength data, and `wrc_class5_63468.txt` contains the five-class labels.

Run `dataset/data_processed_5fold.py` and `dataset/regression_63468_5fold.py` to prepare the classification and regression folds, respectively.

## 2. Promoter Strength Prediction Model Training

Run `prediction/train_optimized_predict.py` to train the five-fold promoter strength classification model defined in `prediction/model_optimized_predict.py`. This model combines sequence features with chaos game representation (CGR) and pseudo dinucleotide composition (PseDNC) features.

Run `prediction/train_DNABert2_MLP.py` to train the DNABERT-2 regression model used for independent evaluation.

## 3. Conditional Diffusion Model Training

Run `prior_condition/methods.py` with `prior_condition/model_optimized_predict.py` and the trained classification model to extract class-specific position weights and nucleotide preferences.

Run `Diffusion_to_optimized/train_diff_all.py` with these class-specific features to train the conditional diffusion model. The model incorporates the features into a U-Net through a Transformer encoder and cross-attention, and uses position-weighted denoising and adjacent-class competitive losses.

## 4. Promoter Sequence Generation

Run `Diffusion_to_optimized/gen_all_10sets.py` with a trained diffusion checkpoint to generate promoter sequences for the five target strength classes.

## 5. Promoter Sequence Optimization

Run `Diffusion_to_optimized/optimized_all_10sets.py` with the trained diffusion and classification models to optimize the generated promoter sequences.

## 6. In Silico Validation

Run `Diffusion_to_optimized/evaluate_all_10sets.py` with the trained DNABERT-2 regression model to evaluate the designed promoter sequences.
