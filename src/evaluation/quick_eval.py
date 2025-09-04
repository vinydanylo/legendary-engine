#!/usr/bin/env python3
"""
Quick evaluation script with limited samples to get metrics fast
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import yaml
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

def load_model_and_tokenizer(config_path):
    """Load the trained model and tokenizer"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_dir = config['output_dir']
    base_model = config['model_name']
    
    print(f"Loading model from: {model_dir}")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        return model, tokenizer, config
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def load_validation_data(file_path, max_samples=500):
    """Load validation data with sample limit"""
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            data.append(json.loads(line))
    return data

def evaluate_model():
    """Quick evaluation with limited samples"""
    model, tokenizer, config = load_model_and_tokenizer('bug.yaml')
    
    if model is None:
        print("Failed to load model!")
        return
    
    # Load limited validation data
    print("Loading validation data...")
    valid_data = load_validation_data(config['valid_file'], max_samples=500)
    print(f"Loaded {len(valid_data)} validation samples")
    
    # Prepare data
    texts = [item['code'] for item in valid_data]
    labels = [item['label'] for item in valid_data]
    
    # Tokenize
    print("Tokenizing...")
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Predict
    print("Making predictions...")
    model.eval()
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch_inputs = {
                'input_ids': inputs['input_ids'][i:i+batch_size],
                'attention_mask': inputs['attention_mask'][i:i+batch_size]
            }
            
            outputs = model(**batch_inputs)
            batch_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            batch_preds = torch.argmax(batch_probs, dim=-1)
            
            predictions.extend(batch_preds.tolist())
            probabilities.extend(batch_probs.tolist())
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Class distribution
    label_counts = np.bincount(labels)
    pred_counts = np.bincount(predictions)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Samples evaluated: {len(labels)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Non-buggy  Buggy")
    print(f"Actual Non-buggy  {cm[0,0]:6d}   {cm[0,1]:5d}")
    print(f"       Buggy      {cm[1,0]:6d}   {cm[1,1]:5d}")
    
    print("\nClass Distribution:")
    print(f"True labels  - Non-buggy: {label_counts[0]:4d}, Buggy: {label_counts[1]:4d}")
    print(f"Predictions  - Non-buggy: {pred_counts[0]:4d}, Buggy: {pred_counts[1]:4d}")
    
    # Analyze prediction confidence
    probs_array = np.array(probabilities)
    non_buggy_conf = probs_array[:, 0]
    buggy_conf = probs_array[:, 1]
    
    print("\nPrediction Confidence Analysis:")
    print(f"Average confidence for non-buggy predictions: {non_buggy_conf[np.array(predictions) == 0].mean():.4f}")
    print(f"Average confidence for buggy predictions: {buggy_conf[np.array(predictions) == 1].mean():.4f}")
    
    # Find examples where model is very confident but wrong
    print("\nHigh-confidence errors (first 5):")
    max_conf = np.maximum(probs_array[:, 0], probs_array[:, 1])
    wrong_preds = np.array(predictions) != np.array(labels)
    high_conf_wrong = (max_conf > 0.9) & wrong_preds
    
    wrong_indices = np.where(high_conf_wrong)[0][:5]
    for idx in wrong_indices:
        print(f"Sample {idx}: True={labels[idx]}, Pred={predictions[idx]}, Conf={max_conf[idx]:.3f}")
        print(f"  Code: {texts[idx][:100]}...")

if __name__ == "__main__":
    evaluate_model()