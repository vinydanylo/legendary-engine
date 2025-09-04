#!/usr/bin/env python3
"""
Consolidated C# bug detection tool combining all code checking functionality.
Supports command line usage, programmatic API, and streamlit integration.
"""

import os
import sys
import argparse
import json
import yaml
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def sanitize_code(code_text: str) -> str:
    """
    Sanitize code by removing comments and normalizing whitespace.
    Handles C#, Java, and other C-style languages.
    """
    # Remove single-line comments (// ...)
    code_text = re.sub(r'//.*?(?=\n|$)', '', code_text)

    # Remove multi-line comments (/* ... */)
    code_text = re.sub(r'/\*.*?\*/', '', code_text, flags=re.DOTALL)

    # Remove XML/HTML comments (<!-- ... -->)
    code_text = re.sub(r'<!--.*?-->', '', code_text, flags=re.DOTALL)

    # Remove all line breaks and normalize whitespace
    code_text = re.sub(r'\r\n|\r|\n', ' ', code_text)

    # Replace multiple whitespace characters with single spaces
    code_text = re.sub(r'\s+', ' ', code_text)

    # Strip leading and trailing whitespace
    code_text = code_text.strip()

    return code_text


class CodeChecker:
    """
    Consolidated code checker class for C# bug detection.
    """
    
    def __init__(self, model_dir: str = None, config_path: str = "bug.yaml", 
                 max_length: int = 512, threshold: float = None, device: str = None):
        """
        Initialize the code checker.
        
        Args:
            model_dir: Path to the trained model directory
            config_path: Path to configuration file
            max_length: Maximum sequence length for tokenization
            threshold: Threshold for bug classification
            device: Device to use for inference ('cpu' or 'cuda')
        """
        self.config_path = config_path
        self.max_length = max_length
        
        # Load configuration
        self.config = self._load_config()
        
        # Set model directory
        if model_dir:
            self.model_dir = model_dir
        else:
            self.model_dir = self.config.get('output_dir', 'codebert_bug')
            
        # Set device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Set threshold
        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = self._load_threshold()
            
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
        return {}
    
    def _load_threshold(self) -> float:
        """Load threshold from model directory or use default."""
        threshold_file = os.path.join(self.model_dir, "threshold.json")
        if os.path.exists(threshold_file):
            try:
                with open(threshold_file, 'r') as f:
                    data = json.load(f)
                    threshold = float(data.get("threshold", 0.5))
                    return max(0.0, min(1.0, threshold))
            except Exception:
                pass
        return 0.5
    
    def _load_model(self):
        """Load the trained model and tokenizer."""
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory '{self.model_dir}' not found! Please train the model first.")
        
        config_file = os.path.join(self.model_dir, "config.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Model config not found in {self.model_dir}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
    
    def predict_single(self, code: str) -> Dict:
        """
        Predict bug probability for a single code snippet.
        
        Args:
            code: The code to analyze
            
        Returns:
            Dict with keys: 'bug_probability', 'prediction', 'confidence', 'risk_level'
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded properly")
        
        # Sanitize the code
        sanitized_code = sanitize_code(code)
        
        # Tokenize
        inputs = self.tokenizer(
            sanitized_code,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            
            # Get probability of bug (class 1)
            bug_prob = probabilities[0][1].item()
            
            # Determine prediction based on threshold
            prediction = 1 if bug_prob > self.threshold else 0
            confidence = bug_prob if prediction == 1 else 1 - bug_prob
            
            # Determine risk level
            if bug_prob >= 0.7:
                risk_level = "HIGH"
                risk_color = "🔴"
            elif bug_prob >= 0.3:
                risk_level = "MEDIUM" 
                risk_color = "🟡"
            else:
                risk_level = "LOW"
                risk_color = "🟢"
        
        return {
            'bug_probability': bug_prob,
            'prediction': prediction,
            'confidence': confidence,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'threshold': self.threshold
        }
    
    def predict_batch(self, codes: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Predict bug probabilities for multiple code snippets.
        
        Args:
            codes: List of code snippets to analyze
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded properly")
        
        results = []
        sanitized_codes = [sanitize_code(code) for code in codes]
        
        with torch.no_grad():
            for start in range(0, len(sanitized_codes), batch_size):
                batch_codes = sanitized_codes[start:start + batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_codes,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Predict
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                
                # Process batch results
                for i, probs in enumerate(probabilities):
                    bug_prob = probs[1].item()
                    prediction = 1 if bug_prob > self.threshold else 0
                    confidence = bug_prob if prediction == 1 else 1 - bug_prob
                    
                    # Determine risk level
                    if bug_prob >= 0.7:
                        risk_level = "HIGH"
                        risk_color = "🔴"
                    elif bug_prob >= 0.3:
                        risk_level = "MEDIUM"
                        risk_color = "🟡" 
                    else:
                        risk_level = "LOW"
                        risk_color = "🟢"
                    
                    results.append({
                        'bug_probability': bug_prob,
                        'prediction': prediction,
                        'confidence': confidence,
                        'risk_level': risk_level,
                        'risk_color': risk_color,
                        'threshold': self.threshold
                    })
        
        return results
    
    def analyze_file(self, file_path: str) -> Dict:
        """
        Analyze a code file for bugs.
        
        Args:
            file_path: Path to the code file
            
        Returns:
            Analysis result dictionary
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            if not code.strip():
                raise ValueError(f"File '{file_path}' is empty")
            
            result = self.predict_single(code)
            result['file_path'] = file_path
            result['file_size'] = len(code)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error analyzing file '{file_path}': {e}")


def main():
    """Command line interface for the code checker."""
    parser = argparse.ArgumentParser(
        description="C# Bug Detection Tool using CodeBERT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python code_checker.py --file sample.cs
  python code_checker.py --file MyClass.cs --verbose
  python code_checker.py --file buggy_code.cs --threshold 0.8
  python code_checker.py --code "public int Divide(int a, int b) { return a / b; }"
        """
    )
    
    # Input options
    parser.add_argument("--file", "-f", help="C# source file to analyze")
    parser.add_argument("--code", "-c", help="Code snippet to analyze directly")
    
    # Configuration options
    parser.add_argument("--config", default="bug.yaml", help="Configuration file (default: bug.yaml)")
    parser.add_argument("--model-dir", help="Model directory path (overrides config)")
    parser.add_argument("--threshold", type=float, help="Prediction threshold (overrides config)")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--device", choices=['cpu', 'cuda', 'auto'], default='auto', help="Device to use")
    
    # Output options
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    # Validate input
    if not args.file and not args.code:
        parser.error("Either --file or --code must be specified")
    
    if args.file and args.code:
        parser.error("Cannot specify both --file and --code")
    
    # Set device
    device = None if args.device == 'auto' else args.device
    
    try:
        # Initialize checker
        checker = CodeChecker(
            model_dir=args.model_dir,
            config_path=args.config,
            max_length=args.max_length,
            threshold=args.threshold,
            device=device
        )
        
        # Perform analysis
        if args.file:
            result = checker.analyze_file(args.file)
            source = f"File: {args.file}"
        else:
            result = checker.predict_single(args.code)
            source = "Code snippet"
        
        # Output results
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n🤖 C# Bug Detection Results")
            print("=" * 40)
            print(f"📁 {source}")
            print(f"🎯 Bug Probability: {result['bug_probability']:.4f}")
            print(f"⚠️  Risk Level: {result['risk_color']} {result['risk_level']}")
            print(f"🔍 Prediction: {'Buggy' if result['prediction'] == 1 else 'Clean'}")
            print(f"📊 Confidence: {result['confidence']:.4f}")
            print(f"🎚️  Threshold: {result['threshold']}")
            
            if args.verbose:
                if 'file_size' in result:
                    print(f"📏 File Size: {result['file_size']} characters")
                    
                print(f"\n💡 Recommendations:")
                if result['bug_probability'] >= 0.7:
                    print("  - High priority for manual review")
                    print("  - Consider immediate code inspection")
                elif result['bug_probability'] >= 0.3:
                    print("  - Consider code review")
                    print("  - May benefit from additional testing")
                else:
                    print("  - Code appears clean")
                    print("  - Low priority for review")
        
        # Exit with appropriate code
        if result['bug_probability'] >= 0.7:
            sys.exit(2)  # High risk
        elif result['bug_probability'] >= 0.3:
            sys.exit(1)  # Medium risk
        else:
            sys.exit(0)  # Low risk
            
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()