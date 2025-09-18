"""
Fuzzy Intent Recognition - Understands misspelled keywords without modifying them
"""
from typing import List, Dict, Tuple, Set, Optional
from rapidfuzz import fuzz, process
import re

class FuzzyIntentRecognizer:
    """
    Uses fuzzy string matching to understand intent from misspelled keywords
    without modifying the original text.
    """
    
    def __init__(self):
        self.brand_vocabulary = self._build_brand_vocabulary()
        self.modifier_vocabulary = self._build_modifier_vocabulary()
        self.topic_vocabulary = self._build_topic_vocabulary()
        
        # Fuzzy matching thresholds
        self.brand_threshold = 75  # Minimum similarity score for brand matching
        self.modifier_threshold = 70  # Minimum similarity score for modifier matching
        self.topic_threshold = 65  # Minimum similarity score for topic matching
    
    def _build_brand_vocabulary(self) -> Dict[str, str]:
        """Build comprehensive brand vocabulary for fuzzy matching"""
        brands = {
            # Major betting brands
            'sportybet': 'sportybet',
            'betway': 'betway',
            'msport': 'msport',
            'betpawa': 'betpawa',
            'betika': 'betika',
            'powerbet': 'powerbet',
            'sportingbet': 'sportingbet',
            'bet9ja': 'bet9ja',
            'nairabet': 'nairabet',
            '1xbet': '1xbet',
            'melbet': 'melbet',
            
            # Common variations and abbreviations
            'sporty': 'sportybet',
            'ms': 'msport',
            'bet way': 'betway',
            'sport bet': 'sportybet',
        }
        return brands
    
    def _build_modifier_vocabulary(self) -> Dict[str, str]:
        """Build comprehensive modifier vocabulary for fuzzy matching"""
        modifiers = {
            # Login variations
            'login': 'Login',
            'log in': 'Login',
            'signin': 'Login',
            'sign in': 'Login',
            'log': 'Login',
            'logn': 'Login',
            'loging': 'Login',
            
            # App variations  
            'app': 'App',
            'application': 'App',
            'download': 'App',
            'apk': 'App',
            'mobile': 'App',
            'downlod': 'App',
            'downlaod': 'App',
            
            # Registration variations
            'register': 'Registration',
            'registration': 'Registration',
            'sign up': 'Registration',
            'signup': 'Registration',
            'join': 'Registration',
            'regster': 'Registration',
            'registr': 'Registration',
            
            # Geographic variations
            'ghana': 'Ghana',
            'gh': 'Ghana',
            'gha': 'Ghana',
            
            # Other common intents
            'code': 'Codes',
            'codes': 'Codes',
            'booking': 'Codes',
            'prediction': 'Predictions',
            'predictions': 'Predictions',
            'tips': 'Predictions',
            'aviator': 'Aviator',
            'crash': 'Aviator',
            'live': 'Live',
            'livescore': 'Live',
            'results': 'Live',
            'casino': 'Games',
            'games': 'Games',
            'slots': 'Games',
            
            # Broken links
            '.com': 'Broken Links',
            'www': 'Broken Links',
            'http': 'Broken Links',
        }
        return modifiers
    
    def _build_topic_vocabulary(self) -> Dict[str, str]:
        """Build comprehensive topic vocabulary for fuzzy matching"""
        topics = {
            # Sports terms
            'football': 'Sports',
            'soccer': 'Sports', 
            'sport': 'Sports',
            'sports': 'Sports',
            'league': 'Sports',
            'srl': 'Sports',
            
            # Casino terms
            'casino': 'Casino',
            'jackpot': 'Casino',
            'slots': 'Casino',
            'spin': 'Casino',
            'aviator': 'Casino',
            'crash': 'Casino',
            
            # Betting terms
            'bet': 'Betting',
            'betting': 'Betting',
            'odds': 'Betting',
            'wager': 'Betting',
            'stake': 'Betting',
        }
        return topics
    
    def fuzzy_match_brands(self, keyword: str, discovered_brands: Optional[Set[str]] = None) -> Tuple[Optional[str], int]:
        """
        Find the best matching brand using fuzzy matching
        Returns (matched_brand, confidence_score)
        """
        keyword_lower = keyword.lower()
        
        # Add discovered brands to vocabulary
        current_vocab = dict(self.brand_vocabulary)
        if discovered_brands:
            for brand in discovered_brands:
                current_vocab[brand] = brand
        
        # Try exact substring matches first (highest confidence)
        for variant, canonical in current_vocab.items():
            if variant in keyword_lower:
                return canonical, 95
        
        # Use fuzzy matching for misspellings
        words = keyword_lower.split()
        best_match = None
        best_score = 0
        
        for word in words:
            if len(word) > 2:  # Skip very short words
                matches = process.extractOne(
                    word, 
                    current_vocab.keys(), 
                    scorer=fuzz.ratio,
                    score_cutoff=self.brand_threshold
                )
                if matches and matches[1] > best_score:
                    best_match = current_vocab[matches[0]]
                    best_score = int(matches[1])
        
        return (best_match, best_score) if best_match else (None, 0)
    
    def fuzzy_match_modifiers(self, keyword: str) -> Tuple[str, int]:
        """
        Find the best matching modifier using fuzzy matching
        Returns (matched_modifier, confidence_score)
        """
        keyword_lower = keyword.lower()
        
        # Check for broken links first (exact patterns)
        if any(pattern in keyword_lower for pattern in ['.com', 'www.', 'http']):
            return 'Broken Links', 95
        
        # Try exact substring matches first
        for variant, canonical in self.modifier_vocabulary.items():
            if variant in keyword_lower:
                return canonical, 95
        
        # Use fuzzy matching for misspellings
        words = keyword_lower.split()
        best_match = None
        best_score = 0
        
        for word in words:
            if len(word) > 2:
                matches = process.extractOne(
                    word,
                    self.modifier_vocabulary.keys(),
                    scorer=fuzz.ratio,
                    score_cutoff=self.modifier_threshold
                )
                if matches and matches[1] > best_score:
                    best_match = self.modifier_vocabulary[matches[0]]
                    best_score = int(matches[1])
        
        return (best_match, best_score) if best_match else ('General', 50)
    
    def fuzzy_match_topics(self, keyword: str) -> Tuple[str, int]:
        """
        Find the best matching main topic using fuzzy matching
        Returns (matched_topic, confidence_score)
        """
        keyword_lower = keyword.lower()
        
        # Try exact substring matches first
        for variant, canonical in self.topic_vocabulary.items():
            if variant in keyword_lower:
                return canonical, 95
        
        # Use fuzzy matching for misspellings
        words = keyword_lower.split()
        best_match = None
        best_score = 0
        
        for word in words:
            if len(word) > 2:
                matches = process.extractOne(
                    word,
                    self.topic_vocabulary.keys(),
                    scorer=fuzz.ratio,
                    score_cutoff=self.topic_threshold
                )
                if matches and matches[1] > best_score:
                    best_match = self.topic_vocabulary[matches[0]]
                    best_score = int(matches[1])
        
        return (best_match, best_score) if best_match else ('Betting', 50)
    
    def analyze_intent(self, keyword: str, discovered_brands: Optional[Set[str]] = None) -> Dict[str, any]:
        """
        Analyze a keyword's intent using fuzzy matching
        Returns classification with confidence scores
        """
        # Get fuzzy matches for each component
        brand_match, brand_conf = self.fuzzy_match_brands(keyword, discovered_brands)
        modifier_match, modifier_conf = self.fuzzy_match_modifiers(keyword)
        topic_match, topic_conf = self.fuzzy_match_topics(keyword)
        
        # Determine main topic based on brand presence
        if brand_match and brand_conf > 80:
            main_topic = 'Branded'
            sub_topic = brand_match
            main_conf = brand_conf
        else:
            # Use topic classification
            main_topic = topic_match
            main_conf = topic_conf
            
            # Determine sub-topic based on main topic
            if main_topic == 'Sports':
                if any(term in keyword.lower() for term in ['football', 'soccer']):
                    sub_topic = 'Soccer/football'
                else:
                    sub_topic = 'General'
            elif main_topic == 'Casino':
                if 'aviator' in keyword.lower():
                    sub_topic = 'Crash Games'
                elif any(term in keyword.lower() for term in ['jackpot', 'spin']):
                    sub_topic = 'Slots'
                else:
                    sub_topic = 'Games'
            elif main_topic == 'Betting':
                if any(term in keyword.lower() for term in ['football', 'soccer']):
                    sub_topic = 'Soccer/football'
                else:
                    sub_topic = 'Sport'
            else:
                sub_topic = 'General'
        
        return {
            'main': main_topic,
            'sub': sub_topic,
            'modifier': modifier_match,
            'main_confidence': main_conf,
            'sub_confidence': 85,  # Default confidence for sub topics
            'modifier_confidence': modifier_conf,
            'original_keyword': keyword
        }
    
    def batch_analyze(self, keywords: List[str], discovered_brands: Optional[Set[str]] = None) -> List[Dict[str, any]]:
        """
        Analyze multiple keywords efficiently
        """
        return [self.analyze_intent(kw, discovered_brands) for kw in keywords]