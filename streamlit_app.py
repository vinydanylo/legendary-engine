import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yaml
import os
import numpy as np
import re

def sanitize_code(code_text: str) -> str:
    """
    Sanitize code by removing comments and all line breaks/whitespace.
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

@st.cache_resource
def load_model():
    """Load the trained CodeBERT model and tokenizer"""
    config_path = "bug.yaml"
    
    if not os.path.exists(config_path):
        st.error("Configuration file 'bug.yaml' not found!")
        return None, None
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_dir = config['output_dir']
    
    if not os.path.exists(model_dir):
        st.error(f"Model directory '{model_dir}' not found! Please train the model first.")
        return None, None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def predict_bug(code_text, model, tokenizer, threshold=0.5):
    """Predict if the code contains bugs"""
    if model is None or tokenizer is None:
        return None, None
    
    # Sanitize the code input
    sanitized_code = sanitize_code(code_text)
    
    # Tokenize the input
    inputs = tokenizer(
        sanitized_code,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        
        # Get probability of bug (class 1)
        bug_prob = probabilities[0][1].item()
        
        # Determine prediction based on threshold
        prediction = 1 if bug_prob > threshold else 0
        confidence = bug_prob if prediction == 1 else 1 - bug_prob
        
    return prediction, bug_prob, confidence

def main():
    st.set_page_config(
        page_title="C# Bug Detection",
        page_icon="🐛",
        layout="wide"
    )
    
    st.title("🐛 C# Bug Detection Tool")
    st.markdown("Detect potential bugs in your C# code using a fine-tuned CodeBERT model.")
    
    # Load model
    with st.spinner("Loading model..."):
        model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        st.stop()
    
    st.success("Model loaded successfully!")
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    threshold = st.sidebar.slider(
        "Prediction Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.01,
        help="Threshold for classifying code as buggy"
    )
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Enter C# Code")
        code_input = st.text_area(
            "Paste your C# code here:",
            height=400,
            placeholder="""public class Calculator 
{
    public int Add(int a, int b)
    {
        return a + b;
    }
    
    public int Divide(int a, int b)
    {
        if (b == 0)
            throw new ArgumentException("Division by zero");
        return a / b;
    }
}""",
            key="code_input"
        )
        
        analyze_button = st.button("🔍 Analyze Code", type="primary")
    
    with col2:
        st.header("Analysis Results")
        
        if analyze_button and code_input.strip():
            with st.spinner("Analyzing code..."):
                prediction, bug_prob, confidence = predict_bug(
                    code_input, model, tokenizer, threshold
                )
                
                if prediction is not None:
                    # Display results
                    if prediction == 1:
                        st.error("🚨 **Potential Bug Detected!**")
                        st.markdown(f"**Bug Probability:** {bug_prob:.2%}")
                    else:
                        st.success("✅ **Code Looks Good!**")
                        st.markdown(f"**Bug Probability:** {bug_prob:.2%}")
                    
                    st.markdown(f"**Confidence:** {confidence:.2%}")
                    
                    # Progress bar for bug probability
                    st.markdown("**Bug Probability Meter:**")
                    st.progress(bug_prob)
                    
                    # Additional details
                    with st.expander("📊 Detailed Analysis"):
                        st.markdown(f"- **Prediction:** {'Buggy' if prediction == 1 else 'Clean'}")
                        st.markdown(f"- **Bug Probability:** {bug_prob:.4f}")
                        st.markdown(f"- **Clean Probability:** {1-bug_prob:.4f}")
                        st.markdown(f"- **Threshold Used:** {threshold}")
                        st.markdown(f"- **Characters Analyzed:** {len(code_input)}")
                
        elif analyze_button:
            st.warning("Please enter some C# code to analyze.")
    
    # Examples section
    st.header("📚 Example Code Snippets")
    
    col_ex1, col_ex2 = st.columns(2)
    
    with col_ex1:
        st.subheader("🟢 Clean Code Example")
        clean_example = """public class MathUtils 
{
    public static int Add(int a, int b)
    {
        return a + b;
    }
    
    public static int Multiply(int x, int y)
    {
        return x * y;
    }
    
    public static double Divide(double numerator, double denominator)
    {
        if (Math.Abs(denominator) < double.Epsilon)
            throw new ArgumentException("Division by zero");
        return numerator / denominator;
    }
}"""
        st.code(clean_example, language="csharp")
        if st.button("Analyze Clean Example"):
            prediction, bug_prob, confidence = predict_bug(clean_example, model, tokenizer, threshold)
            if prediction is not None:
                st.write(f"Prediction: {'Buggy' if prediction == 1 else 'Clean'} (Prob: {bug_prob:.2%})")
    
    with col_ex2:
        st.subheader("🔴 Buggy Code Example")
        buggy_example = """public class Calculator 
{
    public int Divide(int numerator, int denominator)
    {
        // Bug: No division by zero check
        return numerator / denominator;
    }
    
    public string GetElement(string[] array, int index)
    {
        // Bug: No bounds checking
        return array[index];
    }
}"""
        st.code(buggy_example, language="csharp")
        if st.button("Analyze Buggy Example"):
            prediction, bug_prob, confidence = predict_bug(buggy_example, model, tokenizer, threshold)
            if prediction is not None:
                st.write(f"Prediction: {'Buggy' if prediction == 1 else 'Clean'} (Prob: {bug_prob:.2%})")
    
    # Model info
    with st.expander("ℹ️ Model Information"):
        st.markdown("""
        **Model:** CodeBERT (microsoft/codebert-base) fine-tuned for C# bug detection
        
        **Training Details:**
        - Trained on synthetic C# dataset with working and buggy code pairs
        - Evaluation Accuracy: 99%
        - Precision: 98.61%
        - Recall: 99.4%
        - F1-Score: 99%
        
        **Common Bug Types Detected:**
        - Division by zero
        - Array bounds violations
        - Null reference issues
        - Logic errors
        - Missing error handling
        """)

if __name__ == "__main__":
    main()