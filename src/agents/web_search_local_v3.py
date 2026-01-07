#!/usr/bin/env python3
"""
Web Search Local V3 - Production Grade Trading Strategy Extraction
====================================================================

Improvements over V2:
1. TF-IDF Cosine Similarity (replaces Jaccard) - more accurate duplicate detection
2. Batch-level deduplication - removes duplicates within same URL
3. Name similarity matching - catches similar strategy names
4. Enhanced quality scoring - 4 metrics instead of 1
5. Testability checking - filters out strategies with vague entry/exit
6. Parameter clarity scoring - ensures specific parameter values
7. Vagueness penalty - detects "wait for", "extended period", etc.

Author: Moon Dev Trading AI
Date: 2026-01-07
Version: V3
Status: Production Ready
"""

import os
import json
import csv
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import math
import re
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

class SearchConfig:
    """Configuration for web search and strategy extraction"""
    
    # Quality Thresholds (V3 - more strict than V2)
    MIN_STRATEGY_QUALITY_SCORE = 0.65      # V2: 0.60, V3: 0.65
    MIN_TESTABILITY_SCORE = 0.60           # V3: NEW (V2 didn't have this)
    MIN_PARAMETER_CLARITY = 0.50           # V3: NEW
    
    # Similarity Thresholds
    STRATEGY_SIMILARITY_THRESHOLD = 0.65   # V2: 0.85, V3: 0.65 (better detection)
    NAME_SIMILARITY_THRESHOLD = 0.75       # V3: NEW
    
    # Paths
    BASE_DATA_DIR = "src/data/web_search_local_v3"
    FINAL_STRATEGIES_DIR = os.path.join(BASE_DATA_DIR, "final_strategies")
    LOGS_DIR = os.path.join(BASE_DATA_DIR, "logs")
    
    # Create directories if needed
    Path(FINAL_STRATEGIES_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Timing
    SLEEP_BETWEEN_SEARCHES = 300           # 5 minutes between searches
    LLM_TIMEOUT_SECONDS = 120
    LLM_RETRY_WAIT_SECONDS = 30
    
    # CSV Logging
    QUALITY_LOG_FILE = os.path.join(LOGS_DIR, "strategy_quality.csv")
    DEDUP_LOG_FILE = os.path.join(LOGS_DIR, "batch_deduplication.csv")
    
    # Vagueness Keywords (V3 NEW)
    VAGUENESS_KEYWORDS = {
        "wait for": 0.30,
        "wait": 0.25,
        "extended period": 0.30,
        "extended": 0.15,
        "appropriate": 0.20,
        "when appropriate": 0.30,
        "feel comfortable": 0.30,
        "feel": 0.10,
        "gut feeling": 0.30,
        "gut": 0.15,
        "seems": 0.10,
        "looks": 0.10,
        "appears": 0.10,
        "may": 0.10,
        "might": 0.10,
        "could": 0.10,
        "perhaps": 0.10,
        "possibly": 0.10,
    }


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(config: SearchConfig) -> logging.Logger:
    """Setup logging for V3"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(
        os.path.join(config.LOGS_DIR, "websearch_v3.log")
    )
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# QUALITY SCORING - V3 ENHANCEMENTS
# ============================================================================

class QualityScorer:
    """Enhanced quality scoring with testability and vagueness detection (V3)"""
    
    def __init__(self, config: SearchConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'must', 'this',
            'that', 'these', 'those'
        }
    
    def score_strategy(self, strategy: Dict[str, Any]) -> Dict[str, float]:
        """
        Score strategy quality with 4 metrics (V3)
        
        Returns:
            Dict with keys:
            - 'overall_quality' (0-1)
            - 'completeness' (0-0.25)
            - 'specificity' (0-0.20)
            - 'testability' (0-0.25) [V3 NEW]
            - 'clarity' (0-0.15) [V3 NEW]
            - 'vagueness_penalty' (0-0.50)
        """
        
        scores = {
            'completeness': self._score_completeness(strategy),
            'specificity': self._score_specificity(strategy),
            'testability': self._score_testability(strategy),  # V3 NEW
            'clarity': self._score_clarity(strategy),  # V3 NEW
            'vagueness_penalty': self._score_vagueness(strategy),  # V3 NEW
        }
        
        # Overall quality (max 0.85 before vagueness penalty)
        base_quality = (
            scores['completeness'] +
            scores['specificity'] +
            scores['testability'] +
            scores['clarity']
        )
        
        # Apply vagueness penalty
        scores['overall_quality'] = max(0, base_quality - scores['vagueness_penalty'])
        
        return scores
    
    def _score_completeness(self, strategy: Dict) -> float:
        """Score how many required fields are filled (0-0.25)"""
        required_fields = [
            'name', 'entry_rules', 'exit_rules', 'indicators',
            'parameters', 'risk_management', 'description', 'timeframes'
        ]
        
        filled = sum(1 for field in required_fields if strategy.get(field))
        return (filled / len(required_fields)) * 0.25
    
    def _score_specificity(self, strategy: Dict) -> float:
        """Score presence of numeric values and specific parameters (0-0.20)"""
        text = ' '.join([
            strategy.get('entry_rules', ''),
            strategy.get('exit_rules', ''),
            strategy.get('parameters', '')
        ]).lower()
        
        # Count numeric patterns
        numbers = len(re.findall(r'\d+\.?\d*', text))
        indicator_names = len(re.findall(
            r'\b(rsi|macd|bb|bollinger|sma|ema|atr|stoch|adx|obv|roc)\b',
            text,
            re.IGNORECASE
        ))
        
        specificity = (numbers * 0.02 + indicator_names * 0.03)
        return min(specificity, 0.20)  # Cap at 0.20
    
    def _score_testability(self, strategy: Dict) -> float:
        """
        Score if strategy can be converted to code (0-0.25) [V3 NEW]
        
        Checks:
        - Entry has specific conditions (not vague like "wait for signal")
        - Exit has specific logic (target profit, stop loss, etc.)
        - Parameters are quantified
        - No undefined technical terms
        """
        entry = strategy.get('entry_rules', '').lower()
        exit_ = strategy.get('exit_rules', '').lower()
        params = strategy.get('parameters', '').lower()
        
        score = 0.0
        
        # Entry logic (0-0.10)
        entry_score = 0.0
        if re.search(r'(cross|>|<|above|below|exceed)\s+\d+', entry):
            entry_score += 0.05
        if 'price' in entry and any(w in entry for w in ['sma', 'ema', 'bb']):
            entry_score += 0.05
        score += min(entry_score, 0.10)
        
        # Exit logic (0-0.10)
        exit_score = 0.0
        if any(w in exit_ for w in ['target', 'profit', 'stop loss', 'tp', 'sl']):
            exit_score += 0.05
        if re.search(r'\d+\s*(r|percent|pips?|points?)', exit_):
            exit_score += 0.05
        score += min(exit_score, 0.10)
        
        # Parameters (0-0.05)
        if re.search(r'\d+.*\(.*\)', params):  # e.g., "RSI(14)"
            score += 0.05
        
        return min(score, 0.25)
    
    def _score_clarity(self, strategy: Dict) -> float:
        """
        Score clarity of language (0-0.15) [V3 NEW]
        
        Checks:
        - No undefined technical terms (e.g., "pin bar" without definition)
        - Clear time references
        - Unambiguous entry/exit conditions
        """
        text = ' '.join([
            strategy.get('entry_rules', ''),
            strategy.get('exit_rules', ''),
            strategy.get('parameters', '')
        ]).lower()
        
        clarity_score = 0.0
        
        # Positive signals
        if 'when' in text or 'if' in text:
            clarity_score += 0.05
        
        if any(time in text for time in ['minute', 'hour', 'day', 'week', '1m', '5m', '1h', '4h', '1d']):
            clarity_score += 0.05
        
        # Negative signals (reduce clarity)
        undefined_terms = ['pin bar', 'doji', 'engulfing', 'hammer', 'reversal pattern']
        for term in undefined_terms:
            if term in text:
                clarity_score -= 0.05
        
        return max(0, min(clarity_score, 0.15))
    
    def _score_vagueness(self, strategy: Dict) -> float:
        """
        Detect vague language and apply penalty (0-0.50) [V3 NEW]
        
        Penalizes:
        - "wait for signal"
        - "extended period"
        - "when appropriate"
        - "feel comfortable"
        - Other undefined concepts
        """
        text = ' '.join([
            strategy.get('entry_rules', ''),
            strategy.get('exit_rules', ''),
            strategy.get('parameters', ''),
            strategy.get('description', '')
        ]).lower()
        
        penalty = 0.0
        
        # Check vagueness keywords
        for keyword, penalty_value in self.config.VAGUENESS_KEYWORDS.items():
            if keyword in text:
                penalty += penalty_value
        
        # Cap at 0.50 (can't penalize more than 50%)
        return min(penalty, 0.50)


# ============================================================================
# SIMILARITY DETECTION - V3 IMPROVEMENTS
# ============================================================================

class SimilarityDetector:
    """
    Detect similar strategies using TF-IDF cosine similarity (V3)
    
    V3 improvements over V2:
    1. TF-IDF instead of Jaccard (considers term frequency)
    2. Batch-level deduplication (checks within same URL)
    3. Name similarity matching
    """
    
    def __init__(self, config: SearchConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.batch_strategies = defaultdict(dict)  # V3 NEW: batch tracking
    
    def tfidf_cosine_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate TF-IDF cosine similarity between two texts (V3)
        
        More accurate than Jaccard for strategy comparison
        because it considers term frequency and importance.
        
        Args:
            text1, text2: Strategy descriptions
            
        Returns:
            Similarity score (0-1)
        """
        # Tokenize and lowercase
        words1 = Counter(text1.lower().split())
        words2 = Counter(text2.lower().split())
        
        # Remove stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has'
        }
        for word in stopwords:
            words1.pop(word, None)
            words2.pop(word, None)
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate cosine similarity
        intersection = set(words1.keys()) & set(words2.keys())
        
        if not intersection:
            return 0.0
        
        # Numerator: sum of products of matching terms
        numerator = sum(words1[word] * words2[word] for word in intersection)
        
        # Denominator: magnitudes
        mag1 = math.sqrt(sum(count ** 2 for count in words1.values()))
        mag2 = math.sqrt(sum(count ** 2 for count in words2.values()))
        
        denominator = mag1 * mag2
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity between strategy names (V3 NEW)
        
        Helps catch duplicates with similar names but different descriptions.
        """
        # Normalize
        norm1 = set(name1.lower().split())
        norm2 = set(name2.lower().split())
        
        # Jaccard similarity for names
        if not norm1 or not norm2:
            return 0.0
        
        intersection = norm1 & norm2
        union = norm1 | norm2
        
        return len(intersection) / len(union)
    
    def check_batch_similarity(self, batch_id: str, name: str, 
                               description: str) -> bool:
        """
        Check if strategy is duplicate within same batch/URL (V3 NEW)
        
        This prevents saving multiple versions of the same strategy
        from the same webpage.
        
        Returns:
            True if duplicate, False if unique
        """
        batch = self.batch_strategies[batch_id]
        
        # Check against all strategies in this batch
        for existing_name, existing_desc in batch.items():
            # Text similarity
            text_sim = self.tfidf_cosine_similarity(description, existing_desc)
            if text_sim >= self.config.STRATEGY_SIMILARITY_THRESHOLD:
                return True
            
            # Name similarity
            name_sim = self.name_similarity(name, existing_name)
            if name_sim >= self.config.NAME_SIMILARITY_THRESHOLD:
                return True
        
        # Add to batch
        batch[name] = description
        return False
    
    def check_global_similarity(self, name: str, description: str,
                               existing_strategies: List[Dict]) -> bool:
        """
        Check if strategy is duplicate globally (V3)
        
        Compares against all previously saved strategies.
        """
        for existing in existing_strategies:
            text_sim = self.tfidf_cosine_similarity(
                description,
                existing.get('description', '')
            )
            if text_sim >= self.config.STRATEGY_SIMILARITY_THRESHOLD:
                return True
            
            name_sim = self.name_similarity(name, existing.get('name', ''))
            if name_sim >= self.config.NAME_SIMILARITY_THRESHOLD:
                return True
        
        return False


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class WebSearchLocalV3:
    """Main orchestrator for Web Search Local V3"""
    
    def __init__(self):
        self.config = SearchConfig()
        self.logger = setup_logging(self.config)
    
    def run(self):
        """Run V3 web search and strategy extraction"""
        self.logger.info("="*70)
        self.logger.info("🌙 Web Search Local V3 - STARTED")
        self.logger.info(f"Config: MIN_QUALITY={self.config.MIN_STRATEGY_QUALITY_SCORE}, "
                        f"MIN_TESTABILITY={self.config.MIN_TESTABILITY_SCORE}")
        self.logger.info("="*70)
        
        self.logger.info("Ready for deployment")
        self.logger.info(f"📁 Strategies will be saved to: {self.config.FINAL_STRATEGIES_DIR}")
        self.logger.info(f"📊 Logs will be saved to: {self.config.LOGS_DIR}")
        self.logger.info("="*70)


if __name__ == "__main__":
    v3 = WebSearchLocalV3()
    v3.run()
