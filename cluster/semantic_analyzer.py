"""
Smart semantic keyword analysis for generating Main/Sub/Modifier/Keyword structure.
Uses ML pattern recognition to discover intent and topic patterns.
"""
import re
from typing import Dict, List, Tuple, Set
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticKeywordAnalyzer:
    def __init__(self):
        # Intent modifiers - these reveal user intent
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
        
        # Brand patterns - will be discovered from data
        self.brand_patterns = set()
        
        # Main topic categories
        self.main_topics = {
            'branded': ['sportybet', 'msport', 'betpawa', 'betway', 'betika', 'powerbet', 'sportingbet'],
            'betting': ['bet', 'betting', 'odds', 'wager', 'stake'],
            'sports': ['football', 'soccer', 'sport', 'srl', 'league'],
            'casino': ['casino', 'aviator', 'jackpot', 'slots', 'games', 'spin']
        }
        
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
        
    def extract_modifier(self, keyword: str) -> str:
        """Extract the intent modifier from a keyword"""
        keyword_lower = keyword.lower()
        
        # Check for broken links FIRST (higher priority)
        if any(pattern in keyword_lower for pattern in ['.com', 'www.', 'http']):
            return 'Broken Links'
        
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
            ('Games', ['games', 'casino', 'slots']),
            ('Live', ['live', 'livescore', 'results'])
        ]
        
        for modifier_name, patterns in intent_priority:
            if any(pattern in keyword_lower for pattern in patterns):
                return modifier_name
        
        return 'General'
    
    def extract_main_topic(self, keyword: str) -> str:
        """Determine the main topic category"""
        keyword_lower = keyword.lower()
        
        # Check for explicit brand mentions (conservative)
        brand_found = False
        for brand in self.brand_patterns:
            if brand in keyword_lower and len(brand) > 4:  # Avoid short generic matches
                brand_found = True
                break
                
        # Also check known brands
        known_brands = ['sportybet', 'msport', 'betpawa', 'betway', 'betika', 'powerbet', 'sportingbet']
        for brand in known_brands:
            if brand in keyword_lower:
                brand_found = True
                break
        
        if brand_found:
            return 'Branded'
        
        # Check casino patterns first (more specific)
        casino_patterns = ['casino', 'aviator', 'jackpot', 'slots', 'spin']
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
        """Extract the sub-topic based on main topic"""
        keyword_lower = keyword.lower()
        
        if main_topic == 'Branded':
            # Find the specific brand
            for brand in self.brand_patterns:
                if brand in keyword_lower:
                    return brand
            
            # Check known brands from main_topics
            for brand in self.main_topics['branded']:
                if brand in keyword_lower:
                    return brand
        
        elif main_topic == 'Sports':
            if any(word in keyword_lower for word in ['football', 'soccer']):
                return 'Soccer/football'
            return 'General'
        
        elif main_topic == 'Casino':
            if 'aviator' in keyword_lower:
                return 'Crash Games'
            elif any(word in keyword_lower for word in ['jackpot', 'spin']):
                return 'Slots'
            return 'Games'
        
        elif main_topic == 'Betting':
            if any(word in keyword_lower for word in ['football', 'soccer']):
                return 'Soccer/football'
            return 'Sport'
        
        return 'General'
    
    def semantic_clustering(self, keywords: List[str], n_clusters: int = None) -> Dict[int, List[str]]:
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
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(vectors)
            
            # Group keywords by cluster
            cluster_groups = {}
            for idx, cluster_id in enumerate(clusters):
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(keywords[idx])
                
            return cluster_groups
        
        except:
            # Fallback: simple grouping
            return {0: keywords}
    
    def analyze_keywords(self, keywords: List[str]) -> pd.DataFrame:
        """Main analysis function - converts keywords to 4-column structure"""
        
        # Step 1: Discover brands from the data
        self.discover_brands(keywords)
        
        # Step 2: Analyze each keyword
        results = []
        
        for keyword in keywords:
            main_topic = self.extract_main_topic(keyword)
            sub_topic = self.extract_sub_topic(keyword, main_topic)
            modifier = self.extract_modifier(keyword)
            
            results.append({
                'Main': main_topic.title(),
                'Sub': sub_topic,
                'Mod': modifier.title(),
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
        
        # Clean up temporary column
        df = df.drop('cluster_group', axis=1)
        
        return df[['Main', 'Sub', 'Mod', 'Keyword']]  # Clean 4-column output


def analyze_keyword_file(csv_path: str) -> pd.DataFrame:
    """Convenience function for analyzing a CSV file"""
    df = pd.read_csv(csv_path)
    
    if 'keyword' not in df.columns:
        raise ValueError("CSV must contain a 'keyword' column")
    
    analyzer = SemanticKeywordAnalyzer()
    results = analyzer.analyze_keywords(df['keyword'].tolist())
    
    return results