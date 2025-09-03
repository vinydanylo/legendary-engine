#!/usr/bin/env python3
"""
Command line tool for C# bug detection using trained CodeBERT model.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_dir):
    """Load the trained bug detection model and tokenizer."""
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        print(f"❌ Error: Model not found in {model_dir}")
        print("Please train the model first by running: python train_bug_detector.py bug.yaml")
        return None, None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()  # Set to evaluation mode
        return tokenizer, model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def predict_bug(code, tokenizer, model, max_length=512):
    """Predict if code contains bugs."""
    # Tokenize the code
    inputs = tokenizer(
        code,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=-1)
    bug_probability = probabilities[0][1].item()  # Probability of bug class (index 1)
    
    return bug_probability

def get_risk_level(probability, thresholds):
    """Determine risk level based on probability and thresholds."""
    if probability >= thresholds['high']:
        return "🔴 HIGH", "red"
    elif probability >= thresholds['medium']:
        return "🟡 MEDIUM", "yellow"
    else:
        return "🟢 LOW", "green"

def format_output(file_path, probability, risk_level, color, verbose=False):
    """Format the output message."""
    print(f"\n📁 File: {file_path}")
    print(f"🎯 Bug Probability: {probability:.4f}")
    print(f"⚠️  Risk Level: {risk_level}")
    
    if verbose:
        print(f"📊 Confidence: {probability*100:.2f}%")
        if probability >= 0.7:
            print("💡 Recommendation: High priority for manual review")
        elif probability >= 0.3:
            print("💡 Recommendation: Consider code review")
        else:
            print("💡 Recommendation: Code appears clean")

def main():
    parser = argparse.ArgumentParser(
        description="C# Bug Detection Tool using CodeBERT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_bugs.py --file sample.cs
  python detect_bugs.py --file MyClass.cs --verbose
  python detect_bugs.py --file buggy_code.cs --high-threshold 0.8
        """
    )
    
    parser.add_argument("--file", "-f", required=True, 
                       help="C# source file to analyze")
    parser.add_argument("--config", "-c", default="bug.yaml",
                       help="Configuration file (default: bug.yaml)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output")
    parser.add_argument("--high-threshold", type=float, default=0.7,
                       help="High risk threshold (default: 0.7)")
    parser.add_argument("--medium-threshold", type=float, default=0.3,
                       help="Medium risk threshold (default: 0.3)")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"❌ Error: File '{args.file}' not found")
        sys.exit(1)
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file '{args.config}' not found")
        sys.exit(1)
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    model_dir = config["output_dir"]
    max_length = config.get("max_seq_len", 512)
    
    print("🤖 C# Bug Detection Tool")
    print("=" * 40)
    
    # Load model
    print("📦 Loading trained model...")
    tokenizer, model = load_model(model_dir)
    if tokenizer is None or model is None:
        sys.exit(1)
    
    # Read the source file
    try:
        with open(args.file, "r", encoding="utf-8") as f:
            code = f.read()
        
        if not code.strip():
            print(f"❌ Error: File '{args.file}' is empty")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        sys.exit(1)
    
    # Predict
    print("🔍 Analyzing code...")
    probability = predict_bug(code, tokenizer, model, max_length)
    
    # Determine risk level
    thresholds = {
        'high': args.high_threshold,
        'medium': args.medium_threshold
    }
    risk_level, color = get_risk_level(probability, thresholds)
    
    # Output results
    format_output(args.file, probability, risk_level, color, args.verbose)
    
    # Exit with appropriate code
    if probability >= args.high_threshold:
        sys.exit(2)  # High risk
    elif probability >= args.medium_threshold:
        sys.exit(1)  # Medium risk
    else:
        sys.exit(0)  # Low risk

if __name__ == "__main__":
    main()