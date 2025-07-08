import os
import re
import string
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
from textstat import flesch_reading_ease

class FeatureExtractor:
    def __init__(self):
        # Set NLTK data path to a safe writable directory
        nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.append(nltk_data_dir)

        # Initialize required resources
        try:
            nltk.data.find('sentiment/vader_lexicon')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('vader_lexicon', download_dir=nltk_data_dir, quiet=True)
            nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
            nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)

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
        features = {}

        # Basic text stats
        features['review_length'] = len(text)
        words = word_tokenize(text)
        features['word_count'] = len(words)
        features['sentence_count'] = len(sent_tokenize(text))

        # Average word length
        clean_words = [word.strip(string.punctuation) for word in words if word.isalpha()]
        features['avg_word_length'] = np.mean([len(word) for word in clean_words]) if clean_words else 0

        # Sentiment
        sentiment_scores = self.sia.polarity_scores(text)
        features['sentiment_score'] = sentiment_scores['compound']
        features['sentiment_magnitude'] = abs(sentiment_scores['compound'])

        # Punctuation and character features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['punctuation_ratio'] = sum(1 for c in text if c in string.punctuation) / len(text) if text else 0

        # Vocabulary diversity
        unique_words = set(word.lower() for word in clean_words)
        features['unique_word_ratio'] = len(unique_words) / len(clean_words) if clean_words else 0

        # Readability
        try:
            features['readability_score'] = flesch_reading_ease(text)
        except:
            features['readability_score'] = 50  # Default fallback

        # Custom linguistic features
        features['personal_pronoun_count'] = self._count_personal_pronouns(words)
        features['superlative_count'] = self._count_superlatives(text.lower())
        features['number_count'] = len(re.findall(r'\d+', text))
        features['spelling_error_count'] = self._estimate_spelling_errors(words)

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

    def _count_personal_pronouns(self, words):
        return sum(1 for word in words if word.lower() in self.personal_pronouns)

    def _count_superlatives(self, text_lower):
        return sum(text_lower.count(word) for word in self.superlatives)

    def _estimate_spelling_errors(self, words):
        # Simple rule-based spelling approximation
        error_patterns = [
            r'\b\w*(.)\1{2,}\w*\b',  # Repeated characters (e.g., "goooood")
            r'\b\w*[0-9]+\w*\b',     # Words with numbers
        ]
        error_count = 0
        for pattern in error_patterns:
            error_count += len([w for w in words if re.match(pattern, w)])

        # Suspected one-letter typos
        error_count += len([w for w in words if len(w) == 1 and w.isalpha()])
        return error_count