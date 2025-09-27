"""
Vector-based semantic learning module for improved keyword analysis.
This module enables the semantic analyzer to learn from training data instead of relying solely on hard-coded rules.
"""
from __future__ import annotations

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


class VectorSemanticLearner:
    """
    Advanced semantic learning system that can learn Main/Sub/Modifier patterns from training data.
    Uses vector embeddings and multi-output classification for sophisticated semantic understanding.
    """
    
    def __init__(self, model_path: str = "data/semantic_model.pkl"):
        self.model_path = Path(model_path)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),  # More sophisticated n-grams
            max_features=10000,
            stop_words='english',
            lowercase=True
        )
        
        # Multi-output classifier for Main/Sub/Modifier prediction
        self.classifier = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                class_weight='balanced'
            )
        )
        
        # Label encoders for each output
        self.main_encoder = {}
        self.sub_encoder = {}
        self.modifier_encoder = {}
        
        # Reverse mapping for prediction decoding
        self.main_decoder = {}
        self.sub_decoder = {}
        self.modifier_decoder = {}
        
        # Learned patterns and vocabularies
        self.learned_brands = set()
        self.learned_patterns = {}
        self.confidence_thresholds = {
            'main_topic': 0.7,
            'sub_topic': 0.6,
            'modifier': 0.5
        }
        
        # Load existing model if available
        self.load_model()
    
    def _prepare_encoders(self, df: pd.DataFrame) -> None:
        """Prepare label encoders for categorical outputs."""
        # Get unique values for each category
        main_topics = sorted(df['main_topic'].unique())
        sub_topics = sorted(df['sub_topic'].unique())
        modifiers = sorted(df['modifier'].unique())
        
        # Create encoders
        self.main_encoder = {label: idx for idx, label in enumerate(main_topics)}
        self.sub_encoder = {label: idx for idx, label in enumerate(sub_topics)}
        self.modifier_encoder = {label: idx for idx, label in enumerate(modifiers)}
        
        # Create decoders
        self.main_decoder = {idx: label for label, idx in self.main_encoder.items()}
        self.sub_decoder = {idx: label for label, idx in self.sub_encoder.items()}
        self.modifier_decoder = {idx: label for label, idx in self.modifier_encoder.items()}
    
    def _extract_learned_patterns(self, df: pd.DataFrame) -> None:
        """Extract brands and patterns from training data for enhanced semantic analysis."""
        # Learn brand patterns from Sub Topic column
        brand_keywords = df[df['main_topic'] == 'Branded']['sub_topic'].unique()
        self.learned_brands.update(brand_keywords)
        
        # Learn semantic patterns by analyzing keyword-to-output relationships
        for _, row in df.iterrows():
            keyword = str(row['keyword']).lower()
            main = row['main_topic']
            sub = row['sub_topic']
            modifier = row['modifier']
            
            # Extract key terms from keywords
            terms = keyword.split()
            for term in terms:
                if len(term) > 2:  # Skip very short terms
                    if term not in self.learned_patterns:
                        self.learned_patterns[term] = {
                            'main_counts': {},
                            'sub_counts': {},
                            'modifier_counts': {}
                        }
                    
                    # Count occurrences for pattern learning
                    pattern = self.learned_patterns[term]
                    pattern['main_counts'][main] = pattern['main_counts'].get(main, 0) + 1
                    pattern['sub_counts'][sub] = pattern['sub_counts'].get(sub, 0) + 1
                    pattern['modifier_counts'][modifier] = pattern['modifier_counts'].get(modifier, 0) + 1
    
    def train(self, training_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the vector-based semantic model on enhanced training data.
        
        Args:
            training_df: DataFrame with columns [keyword, main_topic, sub_topic, modifier]
            
        Returns:
            Dictionary with training metrics and results
        """
        if training_df.empty:
            return {'status': 'skipped', 'reason': 'No training data provided'}
        
        # Validate required columns
        required_cols = ['keyword', 'main_topic', 'sub_topic', 'modifier']
        missing_cols = [col for col in required_cols if col not in training_df.columns]
        if missing_cols:
            return {'status': 'error', 'reason': f'Missing columns: {missing_cols}'}
        
        # Clean data
        training_df = training_df.dropna(subset=required_cols)
        if training_df.empty:
            return {'status': 'skipped', 'reason': 'No valid training data after cleaning'}
        
        # Prepare encoders and extract patterns
        self._prepare_encoders(training_df)
        self._extract_learned_patterns(training_df)
        
        # Prepare features
        keywords = training_df['keyword'].astype(str).tolist()
        X = self.vectorizer.fit_transform(keywords)
        
        # Prepare targets (multi-output)
        y_main = [self.main_encoder[topic] for topic in training_df['main_topic']]
        y_sub = [self.sub_encoder[topic] for topic in training_df['sub_topic']]
        y_modifier = [self.modifier_encoder[modifier] for modifier in training_df['modifier']]
        
        y = np.column_stack([y_main, y_sub, y_modifier])
        
        # Split data for evaluation - handle small datasets gracefully
        if len(training_df) > 20:  # Only split if we have enough data
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y_main
                )
            except ValueError as e:
                # If stratified split fails, use regular split
                print(f"Warning: Stratified split failed ({e}), using regular split")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
        else:
            # For small datasets, use the same data for train and test
            X_train, X_test, y_train, y_test = X, X, y, y
        
        # Train model
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        
        # Calculate accuracy for each output
        main_accuracy = accuracy_score(y_test[:, 0], y_pred[:, 0])
        sub_accuracy = accuracy_score(y_test[:, 1], y_pred[:, 1])
        modifier_accuracy = accuracy_score(y_test[:, 2], y_pred[:, 2])
        
        # Save model
        self.save_model()
        
        return {
            'status': 'success',
            'training_samples': len(training_df),
            'main_topic_accuracy': main_accuracy,
            'sub_topic_accuracy': sub_accuracy,
            'modifier_accuracy': modifier_accuracy,
            'learned_brands': len(self.learned_brands),
            'learned_patterns': len(self.learned_patterns),
            'unique_main_topics': len(self.main_encoder),
            'unique_sub_topics': len(self.sub_encoder),
            'unique_modifiers': len(self.modifier_encoder)
        }
    
    def predict(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Predict semantic structure for keywords using the trained model.
        
        Args:
            keywords: List of keywords to analyze
            
        Returns:
            List of predictions with confidence scores
        """
        if not hasattr(self.classifier, 'estimators_') or not self.main_encoder:
            # Model not trained, return empty predictions
            return [{'main_topic': 'Betting', 'sub_topic': 'General', 'modifier': 'General', 'confidence': 0.0} 
                   for _ in keywords]
        
        # Vectorize keywords
        try:
            X = self.vectorizer.transform(keywords)
        except Exception:
            # Vectorizer not fitted, return default predictions
            return [{'main_topic': 'Betting', 'sub_topic': 'General', 'modifier': 'General', 'confidence': 0.0} 
                   for _ in keywords]
        
        # Get predictions and probabilities
        predictions = self.classifier.predict(X)
        
        # Try to get prediction probabilities for confidence scoring
        confidences = []
        try:
            probabilities = self.classifier.predict_proba(X)
            for i in range(len(keywords)):
                # Average confidence across all outputs
                main_conf = np.max(probabilities[0][i]) if len(probabilities) > 0 else 0.5
                sub_conf = np.max(probabilities[1][i]) if len(probabilities) > 1 else 0.5
                modifier_conf = np.max(probabilities[2][i]) if len(probabilities) > 2 else 0.5
                avg_confidence = (main_conf + sub_conf + modifier_conf) / 3
                confidences.append(avg_confidence)
        except Exception:
            # Fallback to default confidence
            confidences = [0.5] * len(keywords)
        
        # Decode predictions
        results = []
        for i, keyword in enumerate(keywords):
            pred = predictions[i]
            
            main_topic = self.main_decoder.get(pred[0], 'Betting')
            sub_topic = self.sub_decoder.get(pred[1], 'General')
            modifier = self.modifier_decoder.get(pred[2], 'General')
            
            results.append({
                'main_topic': main_topic,
                'sub_topic': sub_topic,
                'modifier': modifier,
                'confidence': confidences[i],
                'vector_based': True
            })
        
        return results
    
    def enhance_with_patterns(self, keyword: str) -> Dict[str, Any]:
        """
        Enhance prediction using learned patterns from training data.
        
        Args:
            keyword: Keyword to analyze
            
        Returns:
            Enhancement information with pattern-based insights
        """
        keyword_lower = keyword.lower()
        terms = keyword_lower.split()
        
        enhancements = {
            'pattern_matches': [],
            'brand_detected': None,
            'confidence_boost': 0.0
        }
        
        # Check for learned brand patterns
        for brand in self.learned_brands:
            if brand.lower() in keyword_lower:
                enhancements['brand_detected'] = brand
                enhancements['confidence_boost'] += 0.2
                break
        
        # Check for learned semantic patterns
        for term in terms:
            if term in self.learned_patterns:
                pattern = self.learned_patterns[term]
                
                # Find most common associations for this term
                if pattern['main_counts']:
                    most_common_main = max(pattern['main_counts'], key=pattern['main_counts'].get)
                    enhancements['pattern_matches'].append({
                        'term': term,
                        'suggests_main': most_common_main,
                        'count': pattern['main_counts'][most_common_main]
                    })
        
        return enhancements
    
    def save_model(self) -> None:
        """Save the trained model and encoders to disk."""
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'main_encoder': self.main_encoder,
            'sub_encoder': self.sub_encoder,
            'modifier_encoder': self.modifier_encoder,
            'main_decoder': self.main_decoder,
            'sub_decoder': self.sub_decoder,
            'modifier_decoder': self.modifier_decoder,
            'learned_brands': self.learned_brands,
            'learned_patterns': self.learned_patterns
        }
        
        self.model_path.parent.mkdir(exist_ok=True)
        joblib.dump(model_data, self.model_path)
    
    def load_model(self) -> bool:
        """Load trained model from disk if available."""
        if not self.model_path.exists():
            return False
        
        try:
            model_data = joblib.load(self.model_path)
            
            self.vectorizer = model_data['vectorizer']
            self.classifier = model_data['classifier']
            self.main_encoder = model_data['main_encoder']
            self.sub_encoder = model_data['sub_encoder']
            self.modifier_encoder = model_data['modifier_encoder']
            self.main_decoder = model_data['main_decoder']
            self.sub_decoder = model_data['sub_decoder']
            self.modifier_decoder = model_data['modifier_decoder']
            self.learned_brands = model_data.get('learned_brands', set())
            self.learned_patterns = model_data.get('learned_patterns', {})
            
            return True
        except Exception as e:
            print(f"Warning: Could not load semantic model: {e}")
            return False
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get statistics about the current training state."""
        if not hasattr(self.classifier, 'estimators_'):
            return {'status': 'not_trained'}
        
        return {
            'status': 'trained',
            'model_file_exists': self.model_path.exists(),
            'learned_brands_count': len(self.learned_brands),
            'learned_patterns_count': len(self.learned_patterns),
            'main_topics_count': len(self.main_encoder),
            'sub_topics_count': len(self.sub_encoder),
            'modifiers_count': len(self.modifier_encoder),
            'vectorizer_features': getattr(self.vectorizer, 'vocabulary_', {}) and len(self.vectorizer.vocabulary_),
            'learned_brands': list(self.learned_brands) if len(self.learned_brands) < 20 else list(self.learned_brands)[:20]
        }