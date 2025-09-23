"""URL pattern analysis for intent detection enhancement."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse


class UrlIntentAnalyzer:
    """Analyzes URL patterns to extract intent signals for better keyword classification."""
    
    def __init__(self):
        self.url_patterns = {
            # Lottery/Lucky Numbers patterns
            'lottery': [
                r'/luckynumbers',
                r'/lottery',
                r'/powerball',
                r'/daily-lotto',
                r'/uk49',
                r'/results'
            ],
            
            # Sports patterns  
            'sports': [
                r'/sport(?:/|$)',
                r'/betting',
                r'/fixtures',
                r'/soccer',
                r'/football',
                r'/rugby',
                r'/cricket',
                r'/tennis',
                r'/live.*sport',
                r'/event/livesport'
            ],
            
            # Casino patterns
            'casino': [
                r'/casino',
                r'/slots',
                r'/lobby/casino',
                r'/games',
                r'/live-casino',
                r'/roulette',
                r'/blackjack',
                r'/poker'
            ],
            
            # Login/Account patterns
            'login': [
                r'/login',
                r'/account/login',
                r'/signin',
                r'/loginform',
                r'/account/loginconfirmation'
            ],
            
            # App/Mobile patterns
            'app': [
                r'/app',
                r'/mobile',
                r'/download',
                r'/betway-app',
                r'/install'
            ],
            
            # Registration patterns
            'registration': [
                r'/register',
                r'/signup',
                r'/join',
                r'/account/register'
            ],
            
            # Promotions patterns
            'promotions': [
                r'/promo',
                r'/bonus',
                r'/offer',
                r'/latestpromos',
                r'/free-spins'
            ],
            
            # Live/In-Play patterns
            'live': [
                r'/live',
                r'/inplay',
                r'/event/inplay'
            ],
            
            # Horse Racing patterns
            'horse_racing': [
                r'/racing',
                r'/horses',
                r'/race',
                r'/tips'
            ],
            
            # Crash Games patterns (from Betway data)
            'crash_games': [
                r'/aviator',
                r'/jetx',
                r'/spaceman',
                r'/crash'
            ],
            
            # Betgames patterns
            'betgames': [
                r'/betgames',
                r'/lobby/betgames'
            ]
        }
        
        # Brand detection patterns
        self.brand_patterns = {
            'betway': r'betway\.co\.za|betway\.com',
            'hollywoodbets': r'hollywoodbets\.net|hollywoodbets\.co\.za',
            'sportingbet': r'sportingbet\.co\.za|sportingbet\.com',
            'supabets': r'supabets\.co\.za',
            'lottostar': r'lottostar\.co\.za',
            'playabets': r'playabets\.co\.za'
        }
    
    def analyze_url(self, url: str) -> Dict[str, Any]:
        """
        Analyze URL to extract intent signals
        
        Args:
            url: The URL to analyze
            
        Returns:
            Dictionary containing intent signals and brand information
        """
        if not url or not isinstance(url, str):
            return {'intent_signals': [], 'brand': None, 'confidence': 0}
            
        try:
            parsed_url = urlparse(url.lower())
            path = parsed_url.path
            domain = parsed_url.netloc
            
            # Detect brand from domain
            brand = self._detect_brand_from_domain(domain)
            
            # Extract intent signals from URL path
            intent_signals = self._extract_intent_signals(path)
            
            # Calculate confidence based on pattern matches
            confidence = self._calculate_confidence(intent_signals, path)
            
            return {
                'intent_signals': intent_signals,
                'brand': brand,
                'confidence': confidence,
                'domain': domain,
                'path': path
            }
            
        except Exception as e:
            return {'intent_signals': [], 'brand': None, 'confidence': 0, 'error': str(e)}
    
    def _detect_brand_from_domain(self, domain: str) -> Optional[str]:
        """Extract brand from domain name."""
        for brand, pattern in self.brand_patterns.items():
            if re.search(pattern, domain):
                return brand
        return None
    
    def _extract_intent_signals(self, path: str) -> List[str]:
        """Extract intent signals from URL path."""
        signals = []
        
        for intent_type, patterns in self.url_patterns.items():
            for pattern in patterns:
                if re.search(pattern, path, re.IGNORECASE):
                    signals.append(intent_type)
                    break  # Only add each intent type once
                    
        return signals
    
    def _calculate_confidence(self, intent_signals: List[str], path: str) -> int:
        """Calculate confidence score based on pattern matches."""
        if not intent_signals:
            return 0
            
        # Base confidence for having any match
        confidence = 60
        
        # Boost for each distinct signal (+10 per signal)
        confidence += len(intent_signals) * 10
        
        # Boost for highly specific patterns
        specific_patterns = [
            ('/luckynumbers', 25),
            ('/account/loginform', 25),
            ('/lobby/casino', 25),
            ('/sport/', 25),
            ('/betway-app', 25),
            ('/event/livesport', 25),
            ('/latestpromos', 20),
            ('/lobby/betgames', 20)
        ]
        
        for pattern, boost in specific_patterns:
            if pattern in path:
                confidence += boost
                
        # Additional boost for multiple distinct signals
        if len(intent_signals) > 1:
            confidence += 10
                
        # Cap at 95 to leave room for semantic analysis
        return min(confidence, 95)
    
    def enhance_semantic_classification(self, keyword: str, url: str, semantic_result: Dict) -> Dict:
        """
        Enhance semantic classification with URL intent analysis
        
        Args:
            keyword: The keyword being analyzed
            url: Associated ranking URL
            semantic_result: Result from semantic analyzer
            
        Returns:
            Enhanced classification result
        """
        url_analysis = self.analyze_url(url)
        
        if not url_analysis['intent_signals']:
            return semantic_result
            
        # Map URL intent signals to main topics
        intent_topic_mapping = {
            'lottery': 'Lottery',
            'sports': 'Sports', 
            'casino': 'Casino',
            'login': 'Branded',
            'app': 'Branded',
            'registration': 'Branded',
            'horse_racing': 'Horse Racing',
            'crash_games': 'Casino',
            'betgames': 'Casino'
        }
        
        # Map URL intent signals to modifiers
        intent_modifier_mapping = {
            'login': 'Login',
            'app': 'App',
            'registration': 'Registration',
            'promotions': 'Promotions',
            'live': 'Live',
            'crash_games': 'Crash Games',
            'betgames': 'Betgames'
        }
        
        # Enhance classification based on URL signals
        enhanced_result = semantic_result.copy()
        
        # Override logic: URL signal conflicts with semantic classification
        if url_analysis['confidence'] >= 70 and url_analysis['intent_signals']:
            primary_signal = url_analysis['intent_signals'][0]
            
            if primary_signal in intent_topic_mapping:
                new_main_topic = intent_topic_mapping[primary_signal]
                current_main = semantic_result.get('main_topic')
                
                # Apply targeted conflict rules for known error patterns
                should_override = False
                
                # Rule 1: Lottery URL but semantic says Casino/Betting
                if primary_signal == 'lottery' and current_main in ['Casino', 'Betting']:
                    should_override = True
                    
                # Rule 2: Sports URL but semantic says Casino/Betting  
                elif primary_signal == 'sports' and current_main in ['Casino', 'Betting']:
                    should_override = True
                    
                # Rule 3: Casino URL but semantic says Sports/Betting
                elif primary_signal == 'casino' and current_main in ['Sports', 'Betting']:
                    should_override = True
                    
                # Rule 4: General override for high confidence conflicts
                elif new_main_topic != current_main and url_analysis['confidence'] >= 80:
                    should_override = True
                
                if should_override:
                    enhanced_result['main_topic'] = new_main_topic
                    enhanced_result['url_override'] = True
                    enhanced_result['url_confidence'] = url_analysis['confidence']
        
        # Brand-based enhancement: leverage domain brand signals
        if url_analysis.get('brand') and url_analysis['intent_signals']:
            brand = url_analysis['brand']
            has_brand_signals = any(signal in ['login', 'app', 'registration'] 
                                  for signal in url_analysis['intent_signals'])
            
            # If brand + login/app/registration signals, ensure Branded classification
            if has_brand_signals or any(term in keyword.lower() for term in ['login', 'app', 'register']):
                enhanced_result['main_topic'] = 'Branded'
                enhanced_result['sub_topic'] = brand
                enhanced_result['url_override'] = True
                enhanced_result['url_confidence'] = url_analysis['confidence']
            # If already Branded but Sub is generic, set brand as Sub
            elif enhanced_result.get('main_topic') == 'Branded' and enhanced_result.get('sub_topic') in ['General', None, '']:
                enhanced_result['sub_topic'] = brand
        
        # Always enhance modifier with URL signals
        for signal in url_analysis['intent_signals']:
            if signal in intent_modifier_mapping:
                url_modifier = intent_modifier_mapping[signal]
                if url_modifier not in enhanced_result.get('modifier', ''):
                    enhanced_result['modifier'] = url_modifier
                    break
        
        # Add URL analysis metadata
        enhanced_result['url_analysis'] = url_analysis
        
        return enhanced_result
    
    def train_from_keyword_url_pairs(self, keyword_url_pairs: List[Tuple[str, str]]) -> Dict:
        """
        Analyze keyword-URL pairs to discover new patterns
        
        Args:
            keyword_url_pairs: List of (keyword, url) tuples
            
        Returns:
            Training insights and pattern suggestions
        """
        pattern_analysis = {
            'url_intent_correlation': {},
            'brand_specific_patterns': {},
            'new_pattern_suggestions': []
        }
        
        for keyword, url in keyword_url_pairs:
            url_analysis = self.analyze_url(url)
            
            if url_analysis['intent_signals']:
                # Track correlation between keywords and URL patterns
                for signal in url_analysis['intent_signals']:
                    if signal not in pattern_analysis['url_intent_correlation']:
                        pattern_analysis['url_intent_correlation'][signal] = []
                    pattern_analysis['url_intent_correlation'][signal].append(keyword)
        
        return pattern_analysis


def create_url_enhanced_analyzer():
    """Factory function to create URL-enhanced analyzer."""
    return UrlIntentAnalyzer()