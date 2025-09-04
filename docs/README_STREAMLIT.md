# C# Bug Detection Streamlit App

A web-based interface for detecting potential bugs in C# code using a fine-tuned CodeBERT model.

## Features

- **Interactive Code Analysis**: Paste C# code and get instant bug detection results
- **Real-time Predictions**: Uses the trained CodeBERT model for sequence classification
- **Confidence Scoring**: Shows probability scores and confidence levels
- **Example Code Snippets**: Includes working and buggy examples to test
- **Adjustable Threshold**: Configure the prediction threshold via the sidebar
- **Detailed Metrics**: View model performance and analysis details

## Quick Start

### Prerequisites

Ensure you have trained the bug detection model first:
```bash
python train_bug_detector.py bug.yaml
```

### Install Dependencies

```bash
pip install streamlit
# or install all requirements
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run streamlit_app.py
```

The app will be available at:
- Local URL: http://localhost:8501
- Network URL: http://[your-ip]:8501

## How to Use

1. **Load Model**: The app automatically loads your trained model from the `codebert_bug` directory
2. **Enter Code**: Paste your C# code in the text area
3. **Analyze**: Click "🔍 Analyze Code" to get predictions
4. **View Results**: See if bugs are detected, with confidence scores and probability meters
5. **Adjust Settings**: Use the sidebar to modify the prediction threshold

## Model Performance

The current model achieves:
- **Accuracy**: 99%
- **Precision**: 98.61%
- **Recall**: 99.4%
- **F1-Score**: 99%

## Example Usage

### Clean Code (Should show low bug probability)
```csharp
public class MathUtils 
{
    public static int Add(int a, int b)
    {
        return a + b;
    }
}
```

### Buggy Code (Should show high bug probability)
```csharp
public class Calculator 
{
    public int Divide(int numerator, int denominator)
    {
        return numerator / denominator; // No division by zero check
    }
}
```

## Configuration

The app reads configuration from `bug.yaml`:
- Model directory: `output_dir`
- Model type: CodeBERT base model
- Max sequence length: 512 tokens

## Troubleshooting

1. **Model not found error**: Ensure you've trained the model using `train_bug_detector.py`
2. **Configuration error**: Check that `bug.yaml` exists and contains the correct `output_dir`
3. **Port already in use**: Use a different port: `streamlit run streamlit_app.py --server.port 8502`

## Architecture

- **Frontend**: Streamlit web interface
- **Backend**: PyTorch + Transformers (CodeBERT)
- **Model**: Fine-tuned microsoft/codebert-base for binary classification
- **Input**: C# source code (up to 512 tokens)
- **Output**: Bug probability + binary classification