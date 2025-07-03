import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import string
from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer

class ReviewAnalyzer:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.model_trainer = ModelTrainer()
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def initialize_model(self):
        """Initialize and train the model"""
        try:
            # Set NLTK data path to workspace directory
            import os
            nltk_data_dir = '/home/runner/workspace/.pythonlibs/nltk_data'
            os.makedirs(nltk_data_dir, exist_ok=True)
            if nltk_data_dir not in nltk.data.path:
                nltk.data.path.append(nltk_data_dir)
            
            # Download required NLTK data
            nltk.download('vader_lexicon', download_dir=nltk_data_dir, quiet=True)
            nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
            nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
            nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
            nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir, quiet=True)
            nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_dir, quiet=True)
            
            # Train the model
            self.model, self.scaler, self.feature_names = self.model_trainer.train_model()
            
        except Exception as e:
            raise Exception(f"Failed to initialize model: {str(e)}")
    
    def analyze_review(self, review_text):
        """Analyze a single review for authenticity"""
        try:
            # Extract features
            features = self.feature_extractor.extract_features(review_text)
            
            # Prepare features for prediction
            feature_vector = self._prepare_feature_vector(features)
            
            # Scale features
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # Make prediction
            prediction = self.model.predict(feature_vector_scaled)[0]
            prediction_proba = self.model.predict_proba(feature_vector_scaled)[0]
            
            # Calculate confidence score
            confidence_score = max(prediction_proba) * 100
            
            # Determine authenticity
            is_authentic = prediction == 1
            
            # Generate key factors and risk indicators
            key_factors = self._generate_key_factors(features, prediction_proba)
            risk_indicators = self._identify_risk_indicators(features)
            
            return {
                'is_authentic': is_authentic,
                'confidence_score': confidence_score,
                'features': features,
                'key_factors': key_factors,
                'risk_indicators': risk_indicators,
                'raw_prediction': prediction,
                'prediction_probabilities': prediction_proba.tolist()
            }
            
        except Exception as e:
            raise Exception(f"Analysis failed: {str(e)}")
    
    def _prepare_feature_vector(self, features):
        """Convert feature dictionary to vector for model prediction"""
        feature_order = [
            'review_length', 'word_count', 'sentence_count', 'avg_word_length',
            'sentiment_score', 'sentiment_magnitude', 'exclamation_count',
            'question_count', 'capital_ratio', 'punctuation_ratio',
            'unique_word_ratio', 'readability_score', 'personal_pronoun_count',
            'superlative_count', 'number_count', 'spelling_error_count'
        ]
        
        return [features.get(feature, 0) for feature in feature_order]
    
    def _generate_key_factors(self, features, prediction_proba):
        """Generate key factors that influenced the decision"""
        factors = []
        
        # Confidence in authentic vs suspicious
        authentic_prob = prediction_proba[1]
        suspicious_prob = prediction_proba[0]
        
        # Review length analysis
        if features['review_length'] > 500:
            factors.append({
                'description': 'Review has substantial length, indicating detailed experience',
                'impact': 'positive'
            })
        elif features['review_length'] < 50:
            factors.append({
                'description': 'Very short review may lack authentic detail',
                'impact': 'negative'
            })
        
        # Sentiment analysis
        if -0.1 <= features['sentiment_score'] <= 0.1:
            factors.append({
                'description': 'Balanced sentiment suggests objective review',
                'impact': 'positive'
            })
        elif abs(features['sentiment_score']) > 0.8:
            factors.append({
                'description': 'Extreme sentiment may indicate bias or fake review',
                'impact': 'negative'
            })
        
        # Capital letters usage
        if features['capital_ratio'] > 0.15:
            factors.append({
                'description': 'Excessive use of capital letters detected',
                'impact': 'negative'
            })
        
        # Exclamation marks
        if features['exclamation_count'] > 3:
            factors.append({
                'description': 'High number of exclamation marks may indicate fake enthusiasm',
                'impact': 'negative'
            })
        
        # Readability
        if 30 <= features['readability_score'] <= 70:
            factors.append({
                'description': 'Appropriate readability level for genuine reviews',
                'impact': 'positive'
            })
        
        # Vocabulary diversity
        if features['unique_word_ratio'] > 0.7:
            factors.append({
                'description': 'High vocabulary diversity suggests authentic expression',
                'impact': 'positive'
            })
        elif features['unique_word_ratio'] < 0.5:
            factors.append({
                'description': 'Low vocabulary diversity may indicate generic content',
                'impact': 'negative'
            })
        
        # Personal pronouns
        if features['personal_pronoun_count'] > 0:
            factors.append({
                'description': 'Use of personal pronouns indicates personal experience',
                'impact': 'positive'
            })
        
        return factors
    
    def _identify_risk_indicators(self, features):
        """Identify specific risk indicators for suspicious reviews"""
        indicators = []
        
        # Very short reviews
        if features['word_count'] < 10:
            indicators.append("Extremely short review with minimal content")
        
        # Excessive punctuation
        if features['punctuation_ratio'] > 0.2:
            indicators.append("Unusually high punctuation usage")
        
        # Too many superlatives
        if features['superlative_count'] > 5:
            indicators.append("Excessive use of superlative expressions")
        
        # Extreme sentiment with short length
        if abs(features['sentiment_score']) > 0.8 and features['word_count'] < 30:
            indicators.append("Extreme sentiment in very short review")
        
        # No personal pronouns in long reviews
        if features['word_count'] > 100 and features['personal_pronoun_count'] == 0:
            indicators.append("Long review without personal experience indicators")
        
        # Perfect spelling with poor grammar patterns
        if features['spelling_error_count'] == 0 and features['readability_score'] < 20:
            indicators.append("Perfect spelling with poor readability pattern")
        
        return indicators
