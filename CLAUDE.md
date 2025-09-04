# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Python-based hackathon project for C# bug detection using machine learning. The project includes data filtering, synthetic dataset generation, model training, and evaluation capabilities.

## Key Scripts

**src/data_processing/generate_enhanced_csharp_dataset.py** - Synthetic dataset generator that:
- Creates working C# code samples and their buggy variants
- Generates balanced datasets for bug detection training
- Produces JSONL files with labeled code samples (0=working, 1=buggy)

**src/data_processing/analyze_training_data.py** - Training data analysis utility for examining dataset characteristics

**src/training/train_bug_detector.py** - Model training script that:
- Fine-tunes CodeBERT for C# bug detection
- Uses HuggingFace transformers for sequence classification
- Includes progress monitoring and metrics computation
- Configurable via YAML configuration files

**src/evaluation/bug_eval.py** - Model evaluation script that:
- Loads trained models for inference and evaluation
- Computes detailed classification metrics (accuracy, precision, recall, F1)
- Provides confusion matrix analysis

**src/evaluation/quick_eval.py** - Quick evaluation utility for model testing

**src/utils/code_checker.py** - Utility for code validation and checking

**src/streamlit_app.py** - Streamlit web interface for bug detection

## Running the Scripts

### C# Code Filtering
```bash
python filter_csharp_scored.py \
  --in_train input_train.jsonl \
  --in_valid input_valid.jsonl \
  --out_train output_train.jsonl \
  --out_valid output_valid.jsonl
```

#### Optional Parameters:
- `--min_csharp_score` (default: 2) - Minimum C# pattern matches required
- `--min_margin` (default: 1) - Minimum score difference between C# and other languages  
- `--others_cap` (default: 2) - Maximum allowed score for other languages
- `--progress_every` (default: 50,000) - Progress reporting interval
- `--max_keep` (default: 0) - Maximum records to keep (0 = unlimited)

### Dataset Generation
```bash
python src/data_processing/generate_enhanced_csharp_dataset.py
```
Generates training and validation files with labeled C# code samples.

### Model Training
```bash
python src/training/train_bug_detector.py config/bug.yaml
```
Trains a CodeBERT model using the configuration specified in `config/bug.yaml`.

### Model Evaluation
```bash
python src/evaluation/bug_eval.py config/bug.yaml
```
Evaluates a trained model and outputs classification metrics.

### Streamlit Web Interface
```bash
streamlit run src/streamlit_app.py
```
Launches the web interface for interactive bug detection.

## Code Architecture

### Data Processing Pipeline
- **Pattern Detection**: Uses compiled regex patterns to identify language-specific constructs
- **Scoring System**: Each language gets scored based on matched patterns in first 4000 characters
- **Filtering Logic**: C# code must meet minimum score and margin requirements while other languages stay below cap
- **Synthetic Generation**: Creates buggy variants of working C# code through systematic modifications

### Machine Learning Components
- **Model**: Fine-tuned CodeBERT (microsoft/codebert-base) for sequence classification
- **Training**: HuggingFace transformers with custom metrics and progress monitoring
- **Evaluation**: Comprehensive metrics including confusion matrix, precision, recall, F1-score
- **Configuration**: YAML-based configuration for model hyperparameters and file paths

## Configuration File (config/bug.yaml)

The training configuration includes:
- Model: microsoft/codebert-base
- Training files: data/processed/train_csharp_bugs_enhanced.jsonl, data/processed/valid_csharp_bugs_enhanced.jsonl
- Hyperparameters: 4 epochs, batch size 8, learning rate 2e-5
- Output directory: data/models/codebert_bug

## Dependencies

### Core Python Libraries
- `re`, `json`, `argparse`, `pathlib`, `collections`, `typing`
- `random`, `numpy`, `yaml`, `os`, `time`, `datetime`

### Machine Learning Libraries
- `torch` - PyTorch deep learning framework
- `transformers` - HuggingFace transformers library
- `datasets` - HuggingFace datasets library
- `sklearn` - Scikit-learn for metrics computation

Install ML dependencies with:
```bash
pip install torch transformers datasets scikit-learn pyyaml numpy
```