import os
import json
import requests
import argparse
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SHLEvaluator:
    """
    Evaluator for SHL Assessment Recommendation System using Mean Recall@K and MAP@K metrics.
    """
    
    def __init__(self, api_url: str, test_data_path: str):
        """
        Initialize the evaluator.
        
        Args:
            api_url: URL of the recommendation API
            test_data_path: Path to the test data JSON file
        """
        self.api_url = api_url
        self.test_data_path = test_data_path
        self.test_data = self._load_test_data()
        
    def _load_test_data(self) -> List[Dict[str, Any]]:
        """Load test data from JSON file."""
        try:
            with open(self.test_data_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            raise
    
    def get_recommendations(self, query: str, max_results: int = 10) -> List[str]:
        """
        Get recommendations from the API.
        
        Args:
            query: The query string
            max_results: Maximum number of results to return
            
        Returns:
            List of recommended assessment names
        """
        try:
            response = requests.get(
                f"{self.api_url}/recommend",
                params={"query": query, "max_results": max_results},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                # Extract assessment names from the recommendations
                recommended_names = [rec["name"] for rec in data.get("recommendations", [])]
                
                # Log the returned assessments for debugging
                logger.info(f"API returned {len(recommended_names)} assessments:")
                for i, name in enumerate(recommended_names[:5]):
                    logger.info(f"  {i+1}. {name}")
                if len(recommended_names) > 5:
                    logger.info(f"  ... and {len(recommended_names) - 5} more")
                
                return recommended_names
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def normalize_name(self, name: str) -> str:
        """
        Normalize assessment name for comparison by converting to lowercase,
        removing any parentheses and their contents, and stripping whitespace.
        
        Args:
            name: Assessment name
            
        Returns:
            Normalized name
        """
        import re
        # Convert to lowercase
        normalized = name.lower()
        # Remove parentheses and their contents
        normalized = re.sub(r'\([^)]*\)', '', normalized)
        # Remove special characters and extra whitespace
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = ' '.join(normalized.split())
        return normalized
    
    def calculate_recall_at_k(self, 
                             recommended: List[str], 
                             relevant: List[str], 
                             k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            recommended: List of recommended items
            relevant: List of relevant items
            k: The K value
            
        Returns:
            Recall@K score
        """
        if not relevant:
            return 0.0
            
        # Limit recommendations to top K
        top_k = recommended[:k]
        
        # Normalize names for comparison
        normalized_top_k = [self.normalize_name(item) for item in top_k]
        normalized_relevant = [self.normalize_name(item) for item in relevant]
        
        # Log normalized names for debugging
        logger.info(f"Normalized relevant assessments: {normalized_relevant}")
        logger.info(f"Normalized top {k} recommended assessments: {normalized_top_k[:min(3, len(normalized_top_k))]}")
        
        # Count relevant items in top K using normalized names
        relevant_in_top_k = 0
        matches = []
        
        for i, rec_name in enumerate(normalized_top_k):
            for rel_name in normalized_relevant:
                # Check for exact match or if one contains the other
                if rec_name == rel_name or rec_name in rel_name or rel_name in rec_name:
                    relevant_in_top_k += 1
                    matches.append((top_k[i], rel_name))
                    break
        
        # Log matches for debugging
        if matches:
            logger.info(f"Matched {len(matches)} assessments:")
            for rec, rel in matches:
                logger.info(f"  Recommended: '{rec}' matched with Relevant: '{rel}'")
        else:
            logger.info("No matches found.")
        
        # Calculate recall
        recall = relevant_in_top_k / len(relevant)
        
        return recall
    
    def calculate_precision_at_k(self, 
                               recommended: List[str], 
                               relevant: List[str], 
                               k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            recommended: List of recommended items
            relevant: List of relevant items
            k: The K value
            
        Returns:
            Precision@K score
        """
        if k == 0:
            return 0.0
            
        # Limit recommendations to top K
        top_k = recommended[:min(k, len(recommended))]
        
        if not top_k:
            return 0.0
            
        # Normalize names for comparison
        normalized_top_k = [self.normalize_name(item) for item in top_k]
        normalized_relevant = [self.normalize_name(item) for item in relevant]
        
        # Count relevant items in top K using normalized names
        relevant_in_top_k = 0
        for rec_name in normalized_top_k:
            for rel_name in normalized_relevant:
                # Check for exact match or if one contains the other
                if rec_name == rel_name or rec_name in rel_name or rel_name in rec_name:
                    relevant_in_top_k += 1
                    break
        
        # Calculate precision
        precision = relevant_in_top_k / len(top_k)
        
        return precision
    
    def calculate_average_precision(self, 
                                  recommended: List[str], 
                                  relevant: List[str], 
                                  k: int) -> float:
        """
        Calculate Average Precision (AP) at K.
        
        Args:
            recommended: List of recommended items
            relevant: List of relevant items
            k: The K value
            
        Returns:
            Average Precision score
        """
        if not relevant or not recommended:
            return 0.0
            
        # Limit recommendations to top K
        top_k = recommended[:k]
        
        # Calculate cumulative precisions at each relevant item
        precisions = []
        for i, item in enumerate(top_k):
            if item in relevant:
                # Calculate precision at position i+1
                precision_at_i = self.calculate_precision_at_k(recommended, relevant, i+1)
                precisions.append(precision_at_i)
        
        # Calculate AP
        if not precisions:
            return 0.0
            
        ap = sum(precisions) / min(k, len(relevant))
        
        return ap
    
    def evaluate(self, k_values: List[int] = [3, 5, 10]) -> Dict[str, Any]:
        """
        Evaluate the recommendation system.
        
        Args:
            k_values: List of K values to evaluate
            
        Returns:
            Dictionary of evaluation results
        """
        results = {f"Mean_Recall@{k}": [] for k in k_values}
        results.update({f"MAP@{k}": [] for k in k_values})
        
        for test_case in self.test_data:
            query = test_case["query"]
            relevant_assessments = test_case["relevant_assessments"]
            
            logger.info(f"Evaluating query: {query}")
            
            # Get recommendations
            recommended = self.get_recommendations(query, max_results=max(k_values))
            
            # Calculate metrics for each K
            for k in k_values:
                recall = self.calculate_recall_at_k(recommended, relevant_assessments, k)
                ap = self.calculate_average_precision(recommended, relevant_assessments, k)
                
                results[f"Mean_Recall@{k}"].append(recall)
                results[f"MAP@{k}"].append(ap)
                
                logger.info(f"  Recall@{k}: {recall:.4f}, AP@{k}: {ap:.4f}")
        
        # Calculate mean metrics
        mean_metrics = {}
        for metric, values in results.items():
            mean_metrics[metric] = np.mean(values) if values else 0.0
            
        logger.info("Evaluation complete:")
        for metric, value in mean_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
            
        return {
            "detailed_results": results,
            "mean_metrics": mean_metrics
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save evaluation results to a JSON file.
        
        Args:
            results: Evaluation results
            output_path: Path to save the results
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate SHL Assessment Recommendation System')
    parser.add_argument('--api-url', type=str, default=os.getenv('API_URL', 'http://localhost:8000'),
                        help='URL of the recommendation API')
    parser.add_argument('--test-data', type=str, default='evaluation/test_data.json',
                        help='Path to test data JSON file')
    parser.add_argument('--output', type=str, default='evaluation/results.json',
                        help='Path to save evaluation results')
    parser.add_argument('--k-values', type=int, nargs='+', default=[3, 5, 10],
                        help='K values for evaluation metrics')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize evaluator
    evaluator = SHLEvaluator(args.api_url, args.test_data)
    
    # Run evaluation
    results = evaluator.evaluate(args.k_values)
    
    # Save results
    evaluator.save_results(results, args.output)

if __name__ == "__main__":
    main()