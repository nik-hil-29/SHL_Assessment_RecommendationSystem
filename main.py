from recommendation_system import SHLRecommendationSystem
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if all necessary environment variables and files exist."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
        return False
    
    # Check if processed data file exists
    processed_data_path = "FinalDataSource/processed_assessments.json"
    if not os.path.exists(processed_data_path):
        logger.error(f"Processed data file {processed_data_path} not found.")
        return False
    
    return True

def main():
    """Main function to demonstrate the SHL Recommendation System."""
    # Check environment first
    if not check_environment():
        return
    
    # Set up the SHL Recommendation System
    processed_data_path = "FinalDataSource/processed_assessments.json"  # Adjust path as needed
    db_directory = "chroma_db"
    
    try:
        # Initialize system
        recommendation_system = SHLRecommendationSystem(
            processed_data_path=processed_data_path,
            db_directory=db_directory
        )
        
        # Test queries
        test_queries = [
            "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
            "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes."
            "Need a SQL database test for experienced developers",
            "I am hiring for an analyst and wants applications to screen using Cognitive and personality tests, what options are available within 45 mins. "
        ]
        
        for query in test_queries:
            print(f"\n\n{'='*80}\nTESTING QUERY: {query}\n{'='*80}\n")
            recommendation_system.display_recommendations(query, max_results=5)
        
    except Exception as e:
        logger.error(f"Error in SHL Recommendation System demonstration: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()