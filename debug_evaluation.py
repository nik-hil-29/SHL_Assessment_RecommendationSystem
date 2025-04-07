import requests
import json
import re
import pandas as pd
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def normalize_name(name: str) -> str:
    """Normalize assessment name for comparison."""
    # Convert to lowercase
    normalized = name.lower()
    # Remove parentheses and their contents
    normalized = re.sub(r'\([^)]*\)', '', normalized)
    # Remove special characters and extra whitespace
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    normalized = ' '.join(normalized.split())
    return normalized

def get_recommendations(api_url: str, query: str, max_results: int = 10) -> List[str]:
    """Get recommendations from the API."""
    try:
        response = requests.get(
            f"{api_url}/recommend",
            params={"query": query, "max_results": max_results},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("recommendations", [])
        else:
            logger.error(f"API error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return []

def check_matches(recommendations: List[Dict[str, Any]], relevant_assessments: List[str]) -> Dict[str, Any]:
    """Check for matches between recommendations and relevant assessments."""
    # Normalize relevant assessment names
    normalized_relevant = [normalize_name(item) for item in relevant_assessments]
    
    # Check each recommendation
    matches = []
    near_matches = []
    
    for rec in recommendations:
        rec_name = rec["name"]
        normalized_rec = normalize_name(rec_name)
        
        # Check for exact matches
        for i, rel_name in enumerate(normalized_relevant):
            if normalized_rec == rel_name:
                matches.append((rec_name, relevant_assessments[i], "exact"))
                break
            elif normalized_rec in rel_name or rel_name in normalized_rec:
                near_matches.append((rec_name, relevant_assessments[i], "partial"))
                break
    
    return {
        "exact_matches": matches,
        "near_matches": near_matches,
        "total_matches": len(matches) + len(near_matches),
        "total_possible": len(relevant_assessments),
        "recall": (len(matches) + len(near_matches)) / len(relevant_assessments) if relevant_assessments else 0
    }

def main():
    # Load test data
    try:
        with open("evaluation/test_data.json", "r") as f:
            test_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return
    
    # API URL
    api_url = "http://localhost:8000"
    
    # Process each test case
    results = []
    
    for case in test_data:
        query = case["query"]
        relevant = case["relevant_assessments"]
        
        logger.info(f"Processing query: {query}")
        logger.info(f"Relevant assessments: {relevant}")
        
        # Get recommendations
        recommendations = get_recommendations(api_url, query, max_results=10)
        
        # Check for matches
        match_results = check_matches(recommendations, relevant)
        
        # Print results
        logger.info(f"Results for query: '{query[:50]}...'")
        logger.info(f"Total matches: {match_results['total_matches']} out of {match_results['total_possible']} (Recall: {match_results['recall']:.2f})")
        
        if match_results["exact_matches"]:
            logger.info("Exact matches:")
            for rec, rel, _ in match_results["exact_matches"]:
                logger.info(f"  Recommended: '{rec}' matches '{rel}'")
        
        if match_results["near_matches"]:
            logger.info("Partial matches:")
            for rec, rel, _ in match_results["near_matches"]:
                logger.info(f"  Recommended: '{rec}' partially matches '{rel}'")
        
        # Show recommendations
        logger.info("Top 5 recommendations:")
        for i, rec in enumerate(recommendations[:5]):
            logger.info(f"  {i+1}. {rec['name']} (Duration: {rec['duration']}, Type: {rec['test_type']})")
        
        # Store results
        results.append({
            "query": query,
            "relevant": relevant,
            "recommendations": [r["name"] for r in recommendations],
            "exact_matches": match_results["exact_matches"],
            "near_matches": match_results["near_matches"],
            "recall": match_results["recall"]
        })
    
    # Calculate overall metrics
    avg_recall = sum(r["recall"] for r in results) / len(results) if results else 0
    logger.info(f"Average recall across all queries: {avg_recall:.2f}")
    
    # Save detailed results
    with open("evaluation/debug_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Debug results saved to evaluation/debug_results.json")

if __name__ == "__main__":
    main()