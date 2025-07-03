import streamlit as st
import pandas as pd
import numpy as np
from review_analyzer import ReviewAnalyzer
import time

# Initialize the review analyzer
@st.cache_resource
def load_analyzer():
    """Load and cache the review analyzer"""
    analyzer = ReviewAnalyzer()
    analyzer.initialize_model()
    return analyzer

def main():
    st.title("🔍 Fake Review Detector")
    st.markdown("---")
    
    st.markdown("""
    **Enter a product review below and our AI will analyze its authenticity.**
    
    Our machine learning model examines various linguistic patterns, sentiment markers, 
    and structural features to determine if a review appears genuine or suspicious.
    """)
    
    # Load the analyzer
    try:
        analyzer = load_analyzer()
    except Exception as e:
        st.error(f"Failed to load the analysis model: {str(e)}")
        st.stop()
    
    # Text input for review
    review_text = st.text_area(
        "Enter Product Review:",
        placeholder="Type or paste the product review you want to analyze...",
        height=150,
        help="Enter any product review text. The longer and more detailed the review, the more accurate the analysis."
    )
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Review", type="primary", use_container_width=True)
    
    if analyze_button and review_text.strip():
        # Show loading spinner
        with st.spinner("Analyzing review authenticity..."):
            time.sleep(0.5)  # Small delay for better UX
            
            try:
                # Perform analysis
                result = analyzer.analyze_review(review_text)
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Main result display
                col1, col2 = st.columns(2)
                
                with col1:
                    # Authenticity classification
                    if result['is_authentic']:
                        st.success("✅ **AUTHENTIC REVIEW**")
                    else:
                        st.error("⚠️ **SUSPICIOUS REVIEW**")
                
                with col2:
                    # Confidence score
                    confidence = result['confidence_score']
                    st.metric(
                        "Confidence Score", 
                        f"{confidence:.1f}%",
                        help="How confident the model is in its prediction"
                    )
                
                # Progress bar for confidence
                st.progress(confidence / 100)
                
                # Detailed analysis
                st.subheader("🔬 Detailed Analysis")
                
                # Feature analysis
                features = result['features']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Review Length", f"{features['review_length']} chars")
                    st.metric("Word Count", f"{features['word_count']} words")
                
                with col2:
                    st.metric("Sentiment Score", f"{features['sentiment_score']:.2f}")
                    st.metric("Readability", f"{features['readability_score']:.1f}")
                
                with col3:
                    st.metric("Exclamation Count", f"{features['exclamation_count']}")
                    st.metric("Capital Ratio", f"{features['capital_ratio']:.1%}")
                
                # Key factors
                st.subheader("🎯 Key Factors")
                factors = result['key_factors']
                
                for factor in factors:
                    if factor['impact'] == 'positive':
                        st.success(f"✅ {factor['description']}")
                    elif factor['impact'] == 'negative':
                        st.error(f"❌ {factor['description']}")
                    else:
                        st.info(f"ℹ️ {factor['description']}")
                
                # Risk indicators
                if result['risk_indicators']:
                    st.subheader("⚠️ Risk Indicators")
                    for indicator in result['risk_indicators']:
                        st.warning(f"• {indicator}")
                
                # Model explanation
                with st.expander("🤖 How does this work?"):
                    st.markdown("""
                    Our machine learning model analyzes multiple aspects of the review:
                    
                    **Text Features:**
                    - Length and structure patterns
                    - Vocabulary diversity and complexity
                    - Grammar and spelling patterns
                    
                    **Sentiment Analysis:**
                    - Emotional tone and intensity
                    - Sentiment consistency
                    - Balanced vs. extreme opinions
                    
                    **Linguistic Markers:**
                    - Use of superlatives and exaggerations
                    - Personal pronouns and specificity
                    - Technical vs. generic language
                    
                    **Behavioral Patterns:**
                    - Review timing and frequency patterns
                    - Common spam/fake review indicators
                    - Authenticity markers from genuine reviews
                    """)
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.info("Please try again or contact support if the issue persists.")
    
    elif analyze_button and not review_text.strip():
        st.warning("⚠️ Please enter a review to analyze.")
    
    # Sidebar with additional information
    with st.sidebar:
        st.header("📋 Tips for Better Analysis")
        st.markdown("""
        **For more accurate results:**
        - Enter complete reviews (not fragments)
        - Include reviews with at least 20 words
        - Paste the original text without modifications
        
        **What makes a review suspicious:**
        - Excessive use of superlatives
        - Generic, vague descriptions
        - Poor grammar with perfect spelling
        - Unnatural language patterns
        - Extreme sentiment without specifics
        """)
        
        st.markdown("---")
        st.markdown("**🔒 Privacy Notice**")
        st.markdown("Reviews are analyzed locally and not stored or shared.")

if __name__ == "__main__":
    main()
