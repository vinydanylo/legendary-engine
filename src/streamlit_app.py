import streamlit as st
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from code_checker import CodeChecker


@st.cache_resource
def load_code_checker():
    """Load the consolidated code checker"""
    try:
        checker = CodeChecker(config_path="config/bug.yaml")
        return checker
    except Exception as e:
        st.error(f"Error loading code checker: {str(e)}")
        return None


def main():
    st.set_page_config(
        page_title="C# Bug Detection",
        page_icon="🐛",
        layout="wide"
    )

    st.title("🐛 C# Bug Detection Tool")
    st.markdown(
        "Detect potential bugs in your C# code using a fine-tuned CodeBERT model.")

    # Load code checker
    with st.spinner("Loading model..."):
        checker = load_code_checker()

    if checker is None:
        st.stop()

    st.success("Model loaded successfully!")

    # Sidebar for settings
    st.sidebar.header("Settings")
    threshold = st.sidebar.slider(
        "Prediction Threshold",
        min_value=0.0,
        max_value=1.0,
        value=checker.threshold,
        step=0.01,
        help="Threshold for classifying code as buggy"
    )
    
    # Update threshold in checker if changed
    if threshold != checker.threshold:
        checker.threshold = threshold

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
                try:
                    result = checker.predict_single(code_input)
                    
                    # Display results
                    if result['prediction'] == 1:
                        st.error("🚨 **Potential Bug Detected!**")
                        st.markdown(f"**Bug Probability:** {result['bug_probability']:.2%}")
                    else:
                        st.success("✅ **Code Looks Good!**")
                        st.markdown(f"**Bug Probability:** {result['bug_probability']:.2%}")

                    st.markdown(f"**Confidence:** {result['confidence']:.2%}")
                    st.markdown(f"**Risk Level:** {result['risk_color']} {result['risk_level']}")

                    # Progress bar for bug probability
                    st.markdown("**Bug Probability Meter:**")
                    st.progress(result['bug_probability'])

                    # Additional details
                    with st.expander("📊 Detailed Analysis"):
                        st.markdown(f"- **Prediction:** {'Buggy' if result['prediction'] == 1 else 'Clean'}")
                        st.markdown(f"- **Bug Probability:** {result['bug_probability']:.4f}")
                        st.markdown(f"- **Clean Probability:** {1-result['bug_probability']:.4f}")
                        st.markdown(f"- **Risk Level:** {result['risk_level']}")
                        st.markdown(f"- **Threshold Used:** {result['threshold']}")
                        st.markdown(f"- **Characters Analyzed:** {len(code_input)}")
                        
                        # Recommendations
                        st.markdown("**💡 Recommendations:**")
                        if result['bug_probability'] >= 0.7:
                            st.markdown("  - High priority for manual review")
                            st.markdown("  - Consider immediate code inspection")
                        elif result['bug_probability'] >= 0.3:
                            st.markdown("  - Consider code review")
                            st.markdown("  - May benefit from additional testing")
                        else:
                            st.markdown("  - Code appears clean")
                            st.markdown("  - Low priority for review")
                            
                except Exception as e:
                    st.error(f"Error analyzing code: {str(e)}")

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
        if st.button("Analyze Clean Example", key="clean_example"):
            try:
                result = checker.predict_single(clean_example)
                st.write(f"Prediction: {'Buggy' if result['prediction'] == 1 else 'Clean'} "
                        f"(Prob: {result['bug_probability']:.2%}, Risk: {result['risk_color']} {result['risk_level']})")
            except Exception as e:
                st.error(f"Error: {str(e)}")

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
        if st.button("Analyze Buggy Example", key="buggy_example"):
            try:
                result = checker.predict_single(buggy_example)
                st.write(f"Prediction: {'Buggy' if result['prediction'] == 1 else 'Clean'} "
                        f"(Prob: {result['bug_probability']:.2%}, Risk: {result['risk_color']} {result['risk_level']})")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Model info
    with st.expander("ℹ️ Model Information"):
        st.markdown(f"""
        **Model:** CodeBERT (microsoft/codebert-base) fine-tuned for C# bug detection
        
        **Current Configuration:**
        - Model Directory: {checker.model_dir}
        - Max Length: {checker.max_length}
        - Device: {checker.device}
        - Default Threshold: {checker.threshold}
        
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