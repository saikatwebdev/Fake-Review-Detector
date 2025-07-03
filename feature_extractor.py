import re
import string
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
import numpy as np
from textstat import flesch_reading_ease

class FeatureExtractor:
    def __init__(self):
        # Set NLTK data path to workspace directory
        import os
        nltk_data_dir = '/home/runner/workspace/.pythonlibs/nltk_data'
        os.makedirs(nltk_data_dir, exist_ok=True)
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.append(nltk_data_dir)
            
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
        # Common superlative patterns
        self.superlatives = [
            'amazing', 'awesome', 'incredible', 'fantastic', 'outstanding',
            'excellent', 'perfect', 'wonderful', 'brilliant', 'superb',
            'magnificent', 'extraordinary', 'phenomenal', 'remarkable',
            'best', 'worst', 'greatest', 'terrible', 'horrible', 'awful'
        ]
        
        # Personal pronouns
        self.personal_pronouns = [
            'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours',
            'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves'
        ]
    
    def extract_features(self, text):
        """Extract comprehensive features from review text"""
        if not text or not text.strip():
            return self._get_empty_features()
        
        text = text.strip()
        
        # Basic text statistics
        features = {}
        features['review_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(sent_tokenize(text))
        
        # Average word length
        words = text.split()
        if words:
            features['avg_word_length'] = np.mean([len(word.strip(string.punctuation)) for word in words])
        else:
            features['avg_word_length'] = 0
        
        # Sentiment analysis
        sentiment_scores = self.sia.polarity_scores(text)
        features['sentiment_score'] = sentiment_scores['compound']
        features['sentiment_magnitude'] = abs(sentiment_scores['compound'])
        
        # Punctuation analysis
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        
        # Character pattern analysis
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['punctuation_ratio'] = sum(1 for c in text if c in string.punctuation) / len(text) if text else 0
        
        # Vocabulary diversity
        unique_words = set(word.lower().strip(string.punctuation) for word in words)
        features['unique_word_ratio'] = len(unique_words) / len(words) if words else 0
        
        # Readability
        try:
            features['readability_score'] = flesch_reading_ease(text)
        except:
            features['readability_score'] = 50  # Default middle value
        
        # Linguistic features
        features['personal_pronoun_count'] = self._count_personal_pronouns(text)
        features['superlative_count'] = self._count_superlatives(text)
        features['number_count'] = len(re.findall(r'\d+', text))
        
        # Spelling and grammar approximation
        features['spelling_error_count'] = self._estimate_spelling_errors(text)
        
        return features
    
    def _get_empty_features(self):
        """Return default features for empty text"""
        return {
            'review_length': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'sentiment_score': 0,
            'sentiment_magnitude': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'capital_ratio': 0,
            'punctuation_ratio': 0,
            'unique_word_ratio': 0,
            'readability_score': 50,
            'personal_pronoun_count': 0,
            'superlative_count': 0,
            'number_count': 0,
            'spelling_error_count': 0
        }
    
    def _count_personal_pronouns(self, text):
        """Count personal pronouns in text"""
        words = word_tokenize(text.lower())
        return sum(1 for word in words if word in self.personal_pronouns)
    
    def _count_superlatives(self, text):
        """Count superlative words and expressions"""
        text_lower = text.lower()
        count = 0
        for superlative in self.superlatives:
            count += text_lower.count(superlative)
        return count
    
    def _estimate_spelling_errors(self, text):
        """Simple spelling error estimation based on patterns"""
        # This is a simplified approach - in production, you might use a spell checker
        words = word_tokenize(text.lower())
        error_patterns = [
            r'\b\w*(.)\1{2,}\w*\b',  # Repeated characters (e.g., "goooood")
            r'\b\w*[0-9]+\w*\b',     # Numbers mixed with letters inappropriately
        ]
        
        error_count = 0
        for pattern in error_patterns:
            error_count += len(re.findall(pattern, text))
        
        # Check for very short words that might be typos
        very_short_words = [word for word in words if len(word) == 1 and word.isalpha()]
        error_count += len(very_short_words)
        
        return error_count
