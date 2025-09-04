#!/usr/bin/env python3
"""
Quick test script to check model predictions on division by zero cases
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yaml
import json

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

def predict_code(model, tokenizer, code_text):
    """Predict if code is buggy (1) or not (0)"""
    inputs = tokenizer(
        code_text, 
        truncation=True, 
        padding=True, 
        max_length=512, 
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    return predicted_class, confidence, predictions[0].tolist()

def test_division_cases():
    """Test various division by zero cases"""
    model, tokenizer, config = load_model_and_tokenizer('bug.yaml')
    
    if model is None:
        print("Failed to load model!")
        return
        
    # Test cases - the problematic code and variations
    test_cases = [
        # Your original case
        ("Your original case", 
         "public class Calculator { public int Divide(int numerator, int denominator) { return numerator / denominator; } public string GetElement(string[] array, int index) { return array[index]; }}"),
        
        # Simpler division case
        ("Simple division", 
         "public int Divide(int a, int b) { return a / b; }"),
        
        # Division with check (should be non-buggy)
        ("Division with check", 
         "public int Divide(int a, int b) { if (b == 0) throw new ArgumentException(); return a / b; }"),
        
        # Modulo by zero
        ("Modulo by zero", 
         "public int ModuloOperation(int a, int b) { return a % b; }"),
        
        # Percentage calculation
        ("Percentage calculation", 
         "public int CalculatePercentage(int value, int total) { return (value * 100) / total; }"),
        
        # Array access (another runtime error)
        ("Array access", 
         "public int GetFirstElement(int[] array) { return array[0]; }"),
        
        # Null reference (another runtime error) 
        ("Null reference", 
         "public string GetLength(string input) { return input.Length.ToString(); }"),
        
        # Working code for comparison
        ("Working addition", 
         "public int Add(int a, int b) { return a + b; }"),
        
        # Syntax error (should be buggy)
        ("Syntax error", 
         "public int Add(int a, int b) { return a + b }")  # Missing semicolon
    ]
    
    print("Testing division by zero and related cases:")
    print("=" * 60)
    
    for name, code in test_cases:
        prediction, confidence, raw_probs = predict_code(model, tokenizer, code)
        label = "BUGGY" if prediction == 1 else "NON-BUGGY"
        
        print(f"\nTest: {name}")
        print(f"Code: {code[:80]}{'...' if len(code) > 80 else ''}")
        print(f"Prediction: {label} (confidence: {confidence:.3f})")
        print(f"Raw probabilities: [Non-buggy: {raw_probs[0]:.3f}, Buggy: {raw_probs[1]:.3f}]")

if __name__ == "__main__":
    test_division_cases()