import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from feature_extractor import FeatureExtractor

class ModelTrainer:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
    
    def train_model(self):
        """Train the authenticity detection model with synthetic training data"""
        # Generate synthetic training data
        X, y = self._generate_training_data()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with accuracy: {accuracy:.3f}")
        
        # Feature names for reference
        feature_names = [
            'review_length', 'word_count', 'sentence_count', 'avg_word_length',
            'sentiment_score', 'sentiment_magnitude', 'exclamation_count',
            'question_count', 'capital_ratio', 'punctuation_ratio',
            'unique_word_ratio', 'readability_score', 'personal_pronoun_count',
            'superlative_count', 'number_count', 'spelling_error_count'
        ]
        
        return model, scaler, feature_names
    
    def _generate_training_data(self):
        """Generate synthetic training data for the model"""
        # Authentic review patterns
        authentic_samples = [
            "I bought this product last month and have been using it daily. The build quality is solid and it works as expected. Had a minor issue with setup but customer service was helpful. Overall satisfied with the purchase.",
            "After using this for 3 weeks, I can say it's decent. Not amazing but does what it's supposed to. The price point is fair for what you get. Would recommend for basic use.",
            "Mixed feelings about this product. Good points: fast delivery, nice packaging. Bad points: instructions unclear, some features don't work as advertised. Might return it.",
            "Excellent product! Been using it for 6 months now and it still works perfectly. Great value for money. My family loves it too. Highly recommend to anyone looking for this type of product.",
            "Okay product but nothing special. Quality is average, price is reasonable. Does the job but there are probably better alternatives out there. Customer service was responsive when I had questions.",
            "Love this product! Great quality, fast shipping, and exactly what I was looking for. The features work well and it's very user-friendly. Worth every penny. Will definitely buy from this company again.",
            "Disappointing purchase. The product arrived damaged and doesn't work properly. Tried contacting customer service but no response yet. Would not recommend based on my experience.",
            "Good product overall. Easy to use, nice design, and good value. Only complaint is that it's a bit smaller than I expected based on the photos. Still happy with the purchase though.",
        ]
        
        # Suspicious/fake review patterns
        suspicious_samples = [
            "AMAZING!!! BEST PRODUCT EVER!!! BUY NOW!!!",
            "Perfect perfect perfect! Love love love! Best quality! Fast shipping! Excellent seller!",
            "This product changed my life completely! Absolutely incredible! Everyone should buy this! Five stars!",
            "Great product. Good quality. Fast delivery. Recommended.",
            "Excellent! Amazing! Wonderful! Best purchase ever! So happy! Thank you!",
            "Outstanding product! Exceptional quality! Superb performance! Magnificent results! Brilliant design!",
            "Best product in the world! Nothing compares! Absolutely perfect! Buy immediately! Amazing seller!",
            "Good product good price good service good everything very satisfied highly recommend to everyone buying.",
            "Fantastic amazing incredible wonderful excellent superb outstanding brilliant magnificent phenomenal!",
            "Perfect product! No problems! Great quality! Fast shipping! Excellent customer service! Five stars!",
        ]
        
        # Generate more samples with variations
        all_samples = []
        all_labels = []
        
        # Authentic samples (label = 1)
        for sample in authentic_samples:
            all_samples.append(sample)
            all_labels.append(1)
            
            # Add variations
            variations = self._create_variations(sample)
            for variation in variations:
                all_samples.append(variation)
                all_labels.append(1)
        
        # Suspicious samples (label = 0)
        for sample in suspicious_samples:
            all_samples.append(sample)
            all_labels.append(0)
            
            # Add variations
            variations = self._create_variations(sample)
            for variation in variations:
                all_samples.append(variation)
                all_labels.append(0)
        
        # Extract features from all samples
        X = []
        y = []
        
        for sample, label in zip(all_samples, all_labels):
            features = self.feature_extractor.extract_features(sample)
            feature_vector = [
                features['review_length'], features['word_count'], 
                features['sentence_count'], features['avg_word_length'],
                features['sentiment_score'], features['sentiment_magnitude'], 
                features['exclamation_count'], features['question_count'], 
                features['capital_ratio'], features['punctuation_ratio'],
                features['unique_word_ratio'], features['readability_score'], 
                features['personal_pronoun_count'], features['superlative_count'], 
                features['number_count'], features['spelling_error_count']
            ]
            X.append(feature_vector)
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def _create_variations(self, text):
        """Create variations of text samples for more training data"""
        variations = []
        
        # Add punctuation variations
        if not text.endswith('!'):
            variations.append(text + "!")
        
        # Remove some punctuation
        no_punct = text.replace('!', '.').replace('?', '.')
        if no_punct != text:
            variations.append(no_punct)
        
        # Add minor modifications for authentic reviews
        if len(text.split()) > 10:  # Longer reviews
            # Add personal touches
            personal_additions = [
                " I would recommend this to others.",
                " Overall, I'm satisfied with my purchase.",
                " Hope this review helps!",
                " Just my honest opinion.",
            ]
            for addition in personal_additions[:2]:  # Limit variations
                variations.append(text + addition)
        
        return variations[:3]  # Limit to 3 variations per sample
