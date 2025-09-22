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
            'hollywoodbets': 'hollywoodbets',
            'supabets': 'supabets',
            'playabets': 'playabets',
            
            # Common variations and abbreviations
            'sporty': 'sportybet',
            'ms': 'msport',
            'bet way': 'betway',
            'sport bet': 'sportybet',
            'hollywood': 'hollywoodbets',
            'supa': 'supabets',
            'playa': 'playabets',
            'batway': 'betway',
            'betwey': 'betway',
            'berway': 'betway',
            'beyway': 'betway',
            'betwa': 'betway',
            'betwat': 'betway',
            'betwy': 'betway',
            'bitway': 'betway',
            'bedway': 'betway',
            'bétway': 'betway',
            'betŵay': 'betway',
            'bètway': 'betway',
            'betaway': 'betway',
            'betwsy': 'betway',
            'betwau': 'betway',
            'bettway': 'betway',
            'bertway': 'betway',
            'betwqy': 'betway',
            'betwày': 'betway',
            'brtway': 'betway',
            'beatway': 'betway',
            'betawy': 'betway',
            'beteway': 'betway',
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
            'sign in': 'Login',
            
            # App variations  
            'app': 'App',
            'application': 'App',
            'download': 'App',
            'apk': 'App',
            'mobile': 'App',
            'downlod': 'App',
            'downlaod': 'App',
            'install': 'App',
            
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
            'south africa': 'South Africa',
            'sa': 'South Africa',
            'za': 'South Africa',
            
            # Betting Markets
            'over/under': 'Over/Under',
            'o/u': 'Over/Under',
            'over under': 'Over/Under',
            'ovr/undr': 'Over/Under',
            'asian handicap': 'Asian Handicap',
            'ah': 'Asian Handicap',
            'asian hcp': 'Asian Handicap',
            'asn handicap': 'Asian Handicap',
            'both teams to score': 'Both Teams To Score',
            'btts': 'Both Teams To Score',
            'both teams score': 'Both Teams To Score',
            'both to score': 'Both Teams To Score',
            'correct score': 'Correct Score',
            'cs': 'Correct Score',
            'correct scr': 'Correct Score',
            'exact score': 'Correct Score',
            'draw no bet': 'Draw No Bet',
            'dnb': 'Draw No Bet',
            'draw no': 'Draw No Bet',
            'drw no bet': 'Draw No Bet',
            'double chance': 'Double Chance',
            'dc': 'Double Chance',
            'dbl chance': 'Double Chance',
            'double chnce': 'Double Chance',
            'handicap': 'Handicap',
            'hcp': 'Handicap',
            'handi': 'Handicap',
            'handicp': 'Handicap',
            'total goals': 'Total Goals',
            'total gls': 'Total Goals',
            'tot goals': 'Total Goals',
            'ttl goals': 'Total Goals',
            'clean sheet': 'Clean Sheet',
            'clean sht': 'Clean Sheet',
            'cln sheet': 'Clean Sheet',
            'first goal scorer': 'First Goal Scorer',
            'fgs': 'First Goal Scorer',
            '1st goal': 'First Goal Scorer',
            'first scorer': 'First Goal Scorer',
            'anytime scorer': 'Anytime Scorer',
            'ats': 'Anytime Scorer',
            'anytime scr': 'Anytime Scorer',
            'any scorer': 'Anytime Scorer',
            'corners': 'Corners',
            'crnrs': 'Corners',
            'corner kicks': 'Corners',
            'corners bet': 'Corners',
            'cards': 'Cards',
            'yellow cards': 'Cards',
            'booking': 'Cards',
            'bookings': 'Cards',
            'penalties': 'Penalties',
            'pens': 'Penalties',
            'penalty': 'Penalties',
            'spot kicks': 'Penalties',
            
            # Slots Games
            'book of dead': 'Book Of Dead',
            'starburst': 'Starburst',
            'gonzo quest': 'Gonzo Quest',
            'gonzo\'s quest': 'Gonzo Quest',
            'mega moolah': 'Mega Moolah',
            'buffalo': 'Buffalo',
            'sweet bonanza': 'Sweet Bonanza',
            'gates of olympus': 'Gates Of Olympus',
            'wolf gold': 'Wolf Gold',
            'fire joker': 'Fire Joker',
            'twin spin': 'Twin Spin',
            'bonanza': 'Bonanza',
            'extra chilli': 'Extra Chilli',
            'divine fortune': 'Divine Fortune',
            'hall of gods': 'Hall Of Gods',
            
            # Crash Games
            'aviator': 'Aviator',
            'crash': 'Crash Games',
            'jetx': 'JetX',
            'spaceman': 'Spaceman',
            'mines': 'Mines',
            'plinko': 'Plinko',
            'limbo': 'Limbo',
            
            # Betgames
            'wheel of fortune': 'Wheel Of Fortune',
            'lucky 7': 'Lucky 7',
            'bet on poker': 'Bet On Poker',
            'war of bets': 'War Of Bets',
            'lucky 6': 'Lucky 6',
            
            # Lottery/Lucky Numbers
            'lucky numbers': 'Lucky Numbers',
            'results': 'Results',
            'draw results': 'Results',
            'winning numbers': 'Results',
            'latest draw': 'Results',
            'predictions': 'Predictions',
            'hot numbers': 'Predictions',
            'cold numbers': 'Predictions',
            'statistics': 'Statistics',
            'frequency': 'Statistics',
            'most drawn': 'Statistics',
            'overdue numbers': 'Statistics',
            'draw time': 'Draw Time',
            'next draw': 'Draw Time',
            'schedule': 'Draw Time',
            
            # Horse Racing
            'tips': 'Tips',
            'form': 'Form',
            'odds': 'Odds',
            'runners': 'Runners',
            'jockey': 'Jockey',
            'trainer': 'Trainer',
            'win': 'Win',
            'place': 'Place',
            'show': 'Show',
            'exacta': 'Exacta',
            'trifecta': 'Trifecta',
            'pick 6': 'Pick 6',
            
            # How-to content
            'how to': 'How To',
            'how to play': 'How To Play',
            'rules': 'Rules',
            'guide': 'Guide',
            'strategy': 'Strategy',
            'meaning': 'Meaning',
            
            # Other common intents
            'code': 'Codes',
            'codes': 'Codes',
            'live': 'Live',
            'livescore': 'Live',
            'casino': 'Games',
            'games': 'Games',
            'slots': 'Slots',
            
            # General categories
            'general': 'General',
            'contact': 'Contact',
            'support': 'Support',
            'help': 'Help',
            
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
            'match': 'Sports',
            'matches': 'Sports',
            'premier league': 'Sports',
            
            # Casino terms
            'casino': 'Casino',
            'jackpot': 'Casino',
            'slots': 'Casino',
            'slot': 'Casino',
            'spin': 'Casino',
            'aviator': 'Casino',
            'crash': 'Casino',
            'roulette': 'Casino',
            'blackjack': 'Casino',
            'baccarat': 'Casino',
            'poker': 'Casino',
            'betgames': 'Casino',
            'live casino': 'Casino',
            'live dealer': 'Casino',
            
            # Betting terms
            'bet': 'Betting',
            'betting': 'Betting',
            'odds': 'Betting',
            'wager': 'Betting',
            'stake': 'Betting',
            'markets': 'Betting',
            'handicap': 'Betting',
            
            # Horse Racing terms
            'horse': 'Horse Racing',
            'horses': 'Horse Racing',
            'racing': 'Horse Racing',
            'race': 'Horse Racing',
            'racecourse': 'Horse Racing',
            'jockey': 'Horse Racing',
            'trainer': 'Horse Racing',
            'thoroughbred': 'Horse Racing',
            'gallop': 'Horse Racing',
            'turf': 'Horse Racing',
            'derby': 'Horse Racing',
            
            # Lottery terms
            'lottery': 'Lottery',
            'lotto': 'Lottery',
            'powerball': 'Lottery',
            'mega millions': 'Lottery',
            'euromillions': 'Lottery',
            'lucky numbers': 'Lottery',
            'draw': 'Lottery',
            'numbers': 'Lottery',
            'uk49': 'Lottery',
            'uk 49': 'Lottery',
            'uk49s': 'Lottery',
            '49s': 'Lottery',
            'teatime': 'Lottery',
            'lunchtime': 'Lottery',
            'daily lotto': 'Lottery',
            'france lotto': 'Lottery',
            'russian lotto': 'Lottery',
            'gosloto': 'Lottery',
            'stoloto': 'Lottery',
            'keno': 'Lottery',
            
            # South African Racecourses
            'kenilworth': 'Horse Racing',
            'turffontein': 'Horse Racing',
            'scottsville': 'Horse Racing',
            'flamingo park': 'Horse Racing',
            'fairview': 'Horse Racing',
            'greyville': 'Horse Racing',
            'vaal': 'Horse Racing',
            'clairwood': 'Horse Racing',
            'arlington': 'Horse Racing',
            'milnerton': 'Horse Racing',
            'runners': 'Horse Racing',
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
    
    def _determine_sub_topic(self, keyword: str, main_topic: str) -> str:
        """Determine sub-topic based on main topic and keyword content"""
        keyword_lower = keyword.lower()
        
        if main_topic == 'Sports':
            if any(term in keyword_lower for term in ['football', 'soccer']):
                return 'Soccer/Football'
            elif any(term in keyword_lower for term in ['rugby', 'cricket', 'tennis']):
                return 'Other Sports'
            else:
                return 'General'
                
        elif main_topic == 'Casino':
            # Crash Games
            if any(term in keyword_lower for term in ['aviator', 'jetx', 'spaceman', 'crash', 'mines', 'plinko', 'limbo']):
                return 'Crash Games'
            # Slots
            elif any(term in keyword_lower for term in ['slots', 'slot', 'book of dead', 'starburst', 'gonzo', 'mega moolah', 'sweet bonanza', 'gates of olympus']):
                return 'Slots'
            # Betgames
            elif any(term in keyword_lower for term in ['betgames', 'wheel of fortune', 'lucky 7', 'lucky 6', 'bet on poker', 'war of bets']):
                return 'Betgames'
            # Live Casino
            elif any(term in keyword_lower for term in ['live', 'dealer', 'roulette', 'blackjack', 'baccarat']):
                return 'Live Casino'
            else:
                return 'Games'
                
        elif main_topic == 'Betting':
            if any(term in keyword_lower for term in ['football', 'soccer']):
                return 'Soccer/Football'
            elif any(term in keyword_lower for term in ['rugby', 'cricket', 'tennis']):
                return 'Other Sports'
            else:
                return 'Sport'
                
        elif main_topic == 'Horse Racing':
            # SA Racecourses
            if any(term in keyword_lower for term in ['kenilworth', 'turffontein', 'scottsville', 'flamingo park', 'fairview', 'greyville']):
                racecourse_map = {
                    'kenilworth': 'Kenilworth',
                    'turffontein': 'Turffontein', 
                    'scottsville': 'Scottsville',
                    'flamingo park': 'Flamingo Park',
                    'fairview': 'Fairview',
                    'greyville': 'Greyville'
                }
                for course, name in racecourse_map.items():
                    if course in keyword_lower:
                        return name
            elif any(term in keyword_lower for term in ['vaal', 'clairwood', 'arlington', 'milnerton']):
                racecourse_map = {
                    'vaal': 'Vaal',
                    'clairwood': 'Clairwood',
                    'arlington': 'Arlington', 
                    'milnerton': 'Milnerton'
                }
                for course, name in racecourse_map.items():
                    if course in keyword_lower:
                        return name
            else:
                return 'General Racing'
                
        elif main_topic == 'Lottery':
            # International Lotteries
            if any(term in keyword_lower for term in ['sa lottery', 'south african lottery', 'powerball sa', 'daily lotto']):
                return 'SA Lottery'
            elif any(term in keyword_lower for term in ['uk lottery', 'uk49', 'uk 49', '49s', 'teatime', 'lunchtime']):
                return 'UK49'
            elif any(term in keyword_lower for term in ['euromillions', 'euro millions']):
                return 'EuroMillions'
            elif any(term in keyword_lower for term in ['powerball', 'mega millions', 'usa lotto']):
                # Context-aware PowerBall classification - Default to SA unless explicit US markers
                if any(us_context in keyword_lower for us_context in ['usa', 'us ', 'power play', 'double play', 'multi-state', 'tennessee', 'florida', 'monday draw', 'wednesday draw', 'saturday draw']):
                    return 'USA Lotto'
                elif 'mega millions' in keyword_lower:
                    return 'USA Lotto'  # Mega Millions is only US
                else:
                    return 'SA Lottery'  # Default PowerBall to SA
            elif any(term in keyword_lower for term in ['france lotto', 'french lotto']):
                return 'France Lotto'
            elif any(term in keyword_lower for term in ['russian lotto', 'stoloto']):
                return 'Russian Lotto'
            elif any(term in keyword_lower for term in ['gosloto', 'gosolto']):
                return 'Gosloto'
            else:
                return 'General Lottery'
                
        else:
            return 'General'
    
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
            sub_topic = self._determine_sub_topic(keyword, main_topic)
        
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