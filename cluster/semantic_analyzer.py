"""
Smart semantic keyword analysis for generating Main/Sub/Modifier/Keyword structure.
Uses ML pattern recognition and fuzzy matching to discover intent patterns even from misspelled keywords.
"""
import re
from typing import Dict, List, Tuple, Set
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .fuzzy_intent import FuzzyIntentRecognizer
from .url_intent_analyzer import UrlIntentAnalyzer

class SemanticKeywordAnalyzer:
    def __init__(self):
        # Initialize fuzzy intent recognizer for handling misspellings
        self.fuzzy_recognizer = FuzzyIntentRecognizer()
        
        # Initialize URL intent analyzer for enhanced classification
        self.url_analyzer = UrlIntentAnalyzer()
        
        # Brand patterns - will be discovered from data  
        self.brand_patterns = set()
        
        # Load slots games for enhanced recognition
        self.load_slots_games()
        
        # Legacy pattern definitions (kept for fallback compatibility)
        self.intent_patterns = {
            'login': ['login', 'log in', 'signin', 'sign in', 'access'],
            'app': ['app', 'download', 'apk', 'mobile', 'application'],
            'registration': ['register', 'registration', 'sign up', 'signup', 'join'],
            'general': ['com', 'www', 'site', 'website'],
            'ghana': ['ghana', 'gh', 'gha'],
            'betting_markets': ['meaning', 'over', 'under', 'gg/ng', 'draw no bet', 'double chance'],
            'codes': ['code', 'booking', 'deposit', 'short code'],
            'predictions': ['prediction', 'tips', 'forecast'],
            'broken_links': ['.com', 'www.', 'http'],
            'aviator': ['aviator', 'crash', 'spribe'],
            'jackpot': ['jackpot', 'spin', 'win'],
            'live': ['live', 'livescore', 'results'],
            'games': ['games', 'casino', 'slots']
        }
        
        # Main topic categories
        self.main_topics = {
            'branded': ['sportybet', 'msport', 'betpawa', 'betway', 'betika', 'powerbet', 'sportingbet'],
            'betting': ['bet', 'betting', 'odds', 'wager', 'stake'],
            'sports': ['football', 'soccer', 'sport', 'srl', 'league'],
            'casino': ['casino', 'aviator', 'jackpot', 'slots', 'games', 'spin']
        }
    
    def load_slots_games(self):
        """Load slots games from the training data for enhanced recognition"""
        try:
            import pandas as pd
            slots_df = pd.read_csv('data/slots_training_data.csv')
            
            # Extract unique slot game names (from non-branded entries)
            casino_mask = slots_df['main_topic'] == 'Casino'
            casino_slots = slots_df.loc[casino_mask, 'keyword'].tolist()
            self.slots_games = set(str(game).lower().strip() for game in casino_slots if pd.notna(game))
            
            # Also extract branded slot patterns  
            branded_mask = slots_df['main_topic'] == 'Branded'
            branded_slots = slots_df.loc[branded_mask]
            self.branded_slot_patterns = {}
            
            for _, row in branded_slots.iterrows():
                keyword = str(row['keyword']).lower()
                brand = str(row['sub_topic']).lower()
                if brand not in self.branded_slot_patterns:
                    self.branded_slot_patterns[brand] = []
                self.branded_slot_patterns[brand].append(keyword)
                
        except Exception as e:
            # Fallback to basic slot recognition
            print(f"Warning: Could not load slots training data: {e}")
            self.slots_games = set()
            self.branded_slot_patterns = {}
        
    def discover_brands(self, keywords: List[str]) -> Set[str]:
        """Use ML to discover brand names from keyword patterns"""
        brands = set()
        
        # Look for consistent patterns that appear frequently
        word_freq = {}
        for keyword in keywords:
            words = keyword.lower().split()
            for word in words:
                if len(word) > 3:  # Skip very short words
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # More conservative: require absolute frequency (≥3 occurrences) AND relative frequency
        min_absolute = max(3, len(keywords) * 0.03)  # At least 3 times OR 3% of dataset
        potential_brands = [word for word, freq in word_freq.items() 
                          if freq >= min_absolute]
        
        # Comprehensive filter of common/generic words
        common_words = {
            'login', 'app', 'download', 'ghana', 'code', 'bet', 'sport', 'game',
            'football', 'casino', 'online', 'live', 'betting', 'jackpot', 'slots',
            'tips', 'prediction', 'today', 'results', 'games', 'aviator', 'spin'
        }
        
        # Only keep known betting brands or terms that frequently appear with brand indicators
        known_brands = {'sportybet', 'msport', 'betpawa', 'betway', 'betika', 'powerbet', 'sportingbet'}
        
        # Filter to known brands or words that appear with login/app/register (brand indicators)
        brand_context_words = set()
        for keyword in keywords:
            if any(indicator in keyword.lower() for indicator in ['login', 'app', 'register']):
                words = keyword.lower().split()
                brand_context_words.update(words)
        
        brands = (set(potential_brands) & known_brands) | (set(potential_brands) & brand_context_words)
        brands = brands - common_words
        
        self.brand_patterns.update(brands)
        return brands
        
    def is_slot_game(self, keyword: str) -> bool:
        """Check if keyword contains a slot game name or slot-related terms"""
        keyword_lower = keyword.lower().strip()
        
        # Check direct match with known slot games
        if keyword_lower in self.slots_games:
            return True
            
        # Check if any slot game name is contained in the keyword
        for slot_game in self.slots_games:
            if slot_game in keyword_lower or keyword_lower in slot_game:
                return True
                
        # Check for specific slot patterns
        slot_patterns = ['slots', 'slot machine', 'free spins', 'megaways', 'hold and win']
        if any(pattern in keyword_lower for pattern in slot_patterns):
            return True
            
        return False
    
    def is_slot_related(self, keyword: str) -> bool:
        """Check if keyword has slot-related terms (broader than specific games)"""
        keyword_lower = keyword.lower().strip()
        
        # First check if it's a specific slot game
        if self.is_slot_game(keyword):
            return True
            
        # Check for generic slot/casino gambling terms
        slot_related_terms = [
            'slot', 'slots', 'spin', 'spins', 'reel', 'reels',
            'jackpot', 'win', 'strike', 'lucky', 'fortune', 'gold',
            'bonus', 'free', 'wild', 'scatter', 'multiplier',
            'machine', 'game', 'casino', '777', 'fruit'
        ]
        
        # Check if keyword contains slot-related terms
        words = keyword_lower.split()
        for word in words:
            if word in slot_related_terms:
                return True
                
        return False
    
    def extract_modifier(self, keyword: str) -> str:
        """Extract the intent modifier from a keyword - optimized for branded slots"""
        keyword_lower = keyword.lower()
        
        # Check for broken links FIRST (higher priority)
        if any(pattern in keyword_lower for pattern in ['.com', 'www.', 'http']):
            return 'Broken Links'
        
        # Check if this is a slot-related + brand combination
        is_slot = self.is_slot_game(keyword)
        is_slot_related = self.is_slot_related(keyword)
        brand_found = self.has_brand(keyword)
        
        if brand_found and (is_slot or is_slot_related):
            return 'Slots'  # Branded slot games/terms get "Slots" modifier
        
        # Check geographic modifiers
        if any(pattern in keyword_lower for pattern in ['ghana', 'gh', 'gha']):
            return 'Ghana'
            
        # Check specific intent patterns (order matters)
        intent_priority = [
            ('Login', ['login', 'log in', 'signin', 'sign in']),
            ('App', ['app', 'download', 'apk', 'mobile', 'application']),
            ('Registration', ['register', 'registration', 'sign up', 'signup', 'join']),
            ('Betting markets', ['meaning', 'over', 'under', 'gg/ng', 'draw no bet', 'double chance']),
            ('Codes', ['code', 'booking', 'deposit', 'short code']),
            ('Predictions', ['prediction', 'tips', 'forecast']),
            ('Aviator', ['aviator', 'crash', 'spribe']),
            ('Slots', ['slots', 'slot machine', 'free spins']) if not brand_found else [],  # Only if not branded
            ('Games', ['games', 'casino']),
            ('Live', ['live', 'livescore', 'results'])
        ]
        
        for modifier_name, patterns in intent_priority:
            if any(pattern in keyword_lower for pattern in patterns):
                return modifier_name
        
        return 'General'
    
    def has_brand(self, keyword: str) -> bool:
        """Check if keyword contains a brand name"""
        keyword_lower = keyword.lower()
        
        # Check discovered brand patterns
        for brand in self.brand_patterns:
            if brand in keyword_lower and len(brand) > 4:  # Avoid short generic matches
                return True
                
        # Check known brands
        known_brands = ['sportybet', 'msport', 'betpawa', 'betway', 'betika', 'powerbet', 'sportingbet']
        for brand in known_brands:
            if brand in keyword_lower:
                return True
                
        return False
    
    def extract_main_topic(self, keyword: str) -> str:
        """Determine the main topic category - optimized for branded slots"""
        keyword_lower = keyword.lower()
        
        # Check if this contains both slot-related terms and a brand
        is_slot = self.is_slot_game(keyword)
        is_slot_related = self.is_slot_related(keyword)
        brand_found = self.has_brand(keyword)
        
        # PRIORITY: Branded slots (specific games OR slot-related terms + brand)
        if brand_found and (is_slot or is_slot_related):
            return 'Branded'
        
        # Standard brand check (non-slot branded content)
        if brand_found:
            return 'Branded'
        
        # Check if it's a slot game without brand
        if is_slot:
            return 'Casino'
        
        # Check casino patterns (more specific)
        casino_patterns = ['casino', 'aviator', 'jackpot', 'spin']
        if any(pattern in keyword_lower for pattern in casino_patterns):
            return 'Casino'
        
        # Check sports patterns
        sports_patterns = ['football', 'soccer', 'sport', 'srl', 'league']
        if any(pattern in keyword_lower for pattern in sports_patterns):
            # If it also has betting context, it's betting, not pure sports
            betting_patterns = ['bet', 'betting', 'odds', 'wager']
            if any(pattern in keyword_lower for pattern in betting_patterns):
                return 'Betting'
            else:
                return 'Sports'
        
        # Default to betting for general gambling terms
        return 'Betting'
    
    def extract_sub_topic(self, keyword: str, main_topic: str) -> str:
        """Extract the sub-topic based on main topic - optimized for branded slots"""
        keyword_lower = keyword.lower()
        
        if main_topic == 'Branded':
            # Find the specific brand (prioritize known brands)
            known_brands = ['sportybet', 'msport', 'betpawa', 'betway', 'betika', 'powerbet', 'sportingbet']
            for brand in known_brands:
                if brand in keyword_lower:
                    return brand.title()  # Return properly capitalized
            
            # Check discovered brand patterns
            for brand in self.brand_patterns:
                if brand in keyword_lower:
                    return brand.title()
                    
            return 'Unknown Brand'
        
        elif main_topic == 'Sports':
            if any(word in keyword_lower for word in ['football', 'soccer']):
                return 'Soccer/football'
            return 'General'
        
        elif main_topic == 'Casino':
            # Check if it's a slot game
            if self.is_slot_game(keyword):
                return 'Slots'
            elif 'aviator' in keyword_lower:
                return 'Crash Games'
            elif any(word in keyword_lower for word in ['jackpot', 'spin']):
                return 'Slots'
            return 'Games'
        
        elif main_topic == 'Betting':
            if any(word in keyword_lower for word in ['football', 'soccer']):
                return 'Soccer/football'
            return 'Sport'
        
        return 'General'
    
    def semantic_clustering(self, keywords: List[str], n_clusters: int = 5) -> Dict[int, List[str]]:
        """Group semantically similar keywords using ML"""
        if not keywords:
            return {}
            
        # Use word-level TF-IDF for better semantic understanding
        vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            max_features=1000,
            stop_words=['the', 'and', 'or', 'in', 'on', 'at', 'to', 'for']
        )
        
        try:
            vectors = vectorizer.fit_transform(keywords)
            
            # Auto-determine number of clusters based on data
            if n_clusters is None:
                n_clusters = max(2, min(len(keywords) // 10, 20))
            
            # Ensure n_clusters doesn't exceed number of keywords
            n_clusters = min(n_clusters, len(keywords))
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(vectors)
            
            # Group keywords by cluster
            cluster_groups = {}
            for idx, cluster_id in enumerate(clusters):
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(keywords[idx])
                
            return cluster_groups
        
        except Exception as e:
            # Fallback: simple grouping
            print(f"Warning: Clustering failed: {e}")
            return {0: keywords}
    
    def analyze_keywords(self, keywords: List[str]) -> pd.DataFrame:
        """
        Main analysis function - converts keywords to 4-column structure
        Now with enhanced slot game recognition and branded slot optimization!
        """
        
        # Step 1: Discover brands from the data (traditional method)
        self.discover_brands(keywords)
        
        # Step 2: Use enhanced semantic analysis for each keyword
        results = []
        
        for keyword in keywords:
            try:
                # Check if this is a slot game or branded combination
                is_slot = self.is_slot_game(keyword)
                has_brand = self.has_brand(keyword)
                
                if is_slot or has_brand:
                    # Use enhanced semantic methods for slots/brands
                    main_topic = self.extract_main_topic(keyword)
                    sub_topic = self.extract_sub_topic(keyword, main_topic)
                    modifier = self.extract_modifier(keyword)
                    
                    results.append({
                        'Main': main_topic,
                        'Sub': sub_topic,
                        'Mod': modifier,
                        'Keyword': keyword
                    })
                else:
                    # Fall back to fuzzy recognizer for other keywords
                    intent_analysis = self.fuzzy_recognizer.analyze_intent(keyword, self.brand_patterns)
                    
                    results.append({
                        'Main': intent_analysis['main'],
                        'Sub': intent_analysis['sub'],
                        'Mod': intent_analysis['modifier'],
                        'Keyword': keyword
                    })
                    
            except Exception as e:
                # Graceful fallback for any errors
                print(f"Warning: Error analyzing '{keyword}': {e}")
                # Use basic semantic methods as fallback
                try:
                    main_topic = self.extract_main_topic(keyword)
                    sub_topic = self.extract_sub_topic(keyword, main_topic)
                    modifier = self.extract_modifier(keyword)
                    
                    results.append({
                        'Main': main_topic,
                        'Sub': sub_topic,
                        'Mod': modifier,
                        'Keyword': keyword
                    })
                except:
                    # Ultimate fallback
                    results.append({
                        'Main': 'Betting',
                        'Sub': 'General',
                        'Mod': 'General',
                        'Keyword': keyword
                    })
        
        df = pd.DataFrame(results)

        # Step 3: Add cluster IDs using semantic clustering
        # Group by Main+Sub for more meaningful clusters
        df['cluster_group'] = df['Main'] + '_' + df['Sub']
        cluster_id = 0
        cluster_map = {}
        
        for group_name, group_df in df.groupby('cluster_group'):
            group_keywords = group_df['Keyword'].tolist()
            
            # Use semantic clustering for this group
            clusters = self.semantic_clustering(group_keywords)
            
            for cluster_keywords in clusters.values():
                cluster_map.update({kw: cluster_id for kw in cluster_keywords})
                cluster_id += 1
        
        df['Cluster_ID'] = df['Keyword'].map(cluster_map)

        # Clean up temporary columns
        df = df.drop(['cluster_group'], axis=1)
        
        return df[['Main', 'Sub', 'Mod', 'Keyword']]  # Clean 4-column output
    
    def analyze_keywords_with_urls(self, keyword_url_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Enhanced analysis function that uses URL patterns to improve intent classification
        
        Args:
            keyword_url_pairs: List of (keyword, url) tuples
            
        Returns:
            DataFrame with same 4-column structure but enhanced accuracy from URL analysis
        """
        # Extract just keywords for brand discovery
        keywords = [pair[0] for pair in keyword_url_pairs]
        self.discover_brands(keywords)
        
        results = []
        
        for keyword, url in keyword_url_pairs:
            # Initialize default intent_analysis to avoid unbound errors
            intent_analysis = {'main': 'Betting', 'sub': 'General', 'modifier': 'General', 'main_confidence': 0.0, 'modifier_confidence': 0.0}
            
            try:
                # Check if this is a slot game or branded combination
                is_slot = self.is_slot_game(keyword)
                has_brand = self.has_brand(keyword)
                
                if is_slot or has_brand:
                    # Use enhanced semantic methods for slots/brands
                    main_topic = self.extract_main_topic(keyword)
                    sub_topic = self.extract_sub_topic(keyword, main_topic)
                    modifier = self.extract_modifier(keyword)
                    
                    semantic_result = {
                        'main_topic': main_topic,
                        'sub_topic': sub_topic,
                        'modifier': modifier
                    }
                    # Update intent_analysis for consistency
                    intent_analysis = {'main': main_topic, 'sub': sub_topic, 'modifier': modifier, 'main_confidence': 80.0, 'modifier_confidence': 80.0}
                else:
                    # Fall back to fuzzy recognizer for other keywords
                    intent_analysis = self.fuzzy_recognizer.analyze_intent(keyword, self.brand_patterns)
                    
                    semantic_result = {
                        'main_topic': intent_analysis['main'],
                        'sub_topic': intent_analysis['sub'],
                        'modifier': intent_analysis['modifier']
                    }
            except Exception as e:
                # Graceful fallback
                print(f"Warning: Error analyzing '{keyword}': {e}")
                semantic_result = {
                    'main_topic': 'Betting',
                    'sub_topic': 'General',
                    'modifier': 'General'
                }
                intent_analysis = {'main': 'Betting', 'sub': 'General', 'modifier': 'General', 'main_confidence': 0.0, 'modifier_confidence': 0.0}
            
            main_conf_raw = float(intent_analysis.get('main_confidence') or 0.0)
            modifier_conf_raw = float(intent_analysis.get('modifier_confidence') or 0.0)
            main_conf = max(0.0, min(1.0, main_conf_raw / 100.0))
            modifier_conf = max(0.0, min(1.0, modifier_conf_raw / 100.0))

            confidence_components = [value for value in (main_conf, modifier_conf) if value > 0]

            # Enhance with URL analysis if URL is provided
            if url and url.strip():
                enhanced_result = self.url_analyzer.enhance_semantic_classification(
                    keyword, url, semantic_result
                )

                # Use enhanced results if available
                main_topic = enhanced_result.get('main_topic', intent_analysis['main'])
                sub_topic = enhanced_result.get('sub_topic', intent_analysis['sub'])
                modifier = enhanced_result.get('modifier', intent_analysis['modifier'])

                # Track URL enhancement metadata
                url_enhanced = enhanced_result.get('url_override', False)
                url_confidence_raw = float(enhanced_result.get('url_confidence', 0) or 0.0)
                url_confidence = max(0.0, min(1.0, url_confidence_raw / 100.0 if url_confidence_raw > 1 else url_confidence_raw))
                if url_confidence > 0:
                    confidence_components.append(url_confidence)
            else:
                # Fall back to semantic-only analysis
                main_topic = intent_analysis['main']
                sub_topic = intent_analysis['sub']
                modifier = intent_analysis['modifier']
                url_enhanced = False
                url_confidence = 0.0

            intent_conf = float(sum(confidence_components) / len(confidence_components)) if confidence_components else main_conf

            results.append({
                'Main': main_topic,
                'Sub': sub_topic,
                'Mod': modifier,
                'Keyword': keyword,
                'URL': url,
                'URL_Enhanced': url_enhanced,
                'URL_Confidence': url_confidence,
                'Cluster_ID': None,
                'main_confidence': main_conf,
                'modifier_confidence': modifier_conf,
                'intent_conf': intent_conf,
            })

        df = pd.DataFrame(results)

        # Add cluster IDs using semantic clustering (same as original method)
        df['cluster_group'] = df['Main'] + '_' + df['Sub']
        cluster_id = 0
        cluster_map = {}
        
        for group_name, group_df in df.groupby('cluster_group'):
            group_keywords = group_df['Keyword'].tolist()
            clusters = self.semantic_clustering(group_keywords)
            
            for cluster_keywords in clusters.values():
                for kw in cluster_keywords:
                    cluster_map[kw] = cluster_id
                cluster_id += 1
        
        df['Cluster_ID'] = df['Keyword'].map(cluster_map)

        # Clean up temporary columns and return DataFrame with confidence metadata
        return df[['Main', 'Sub', 'Mod', 'Keyword', 'URL', 'URL_Enhanced', 'URL_Confidence', 'Cluster_ID', 'main_confidence', 'modifier_confidence', 'intent_conf']]


def analyze_keyword_file(csv_path: str) -> pd.DataFrame:
    """Convenience function for analyzing a CSV file"""
    df = pd.read_csv(csv_path)
    
    if 'keyword' not in df.columns:
        raise ValueError("CSV must contain a 'keyword' column")
    
    analyzer = SemanticKeywordAnalyzer()
    results = analyzer.analyze_keywords(df['keyword'].tolist())
    
    return results