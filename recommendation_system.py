import os
import json
import logging
import time
from typing import List, Dict, Any
import pandas as pd
from IPython.display import display, HTML
from vector_store import AssessmentVectorStore
from gemini_integration import GeminiIntegration
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SHLRecommendationSystem:
    """
    Combined system for SHL assessment recommendations using vector search and LLM enhancement.
    """
    
    def __init__(self, 
                 processed_data_path: str = "processed_assessments.json",
                 db_directory: str = "chroma_db",
                 collection_name: str = "shl_assessments"):
        """
        Initialize the recommendation system.
        
        Args:
            processed_data_path: Path to the processed assessment data JSON file
            db_directory: Directory to store the Chroma database
            collection_name: Name of the vector collection
        """
        self.google_api_key = os.environ.get("GOOGLE_API_KEY")
        
        if not self.google_api_key:
            logger.error("No Google API key provided. Set GOOGLE_API_KEY environment variable.")
            raise ValueError("Google API key is required")
        
        # Initialize vector store
        self.vector_store = AssessmentVectorStore(
            processed_data_path=processed_data_path,
            db_directory=db_directory,
            collection_name=collection_name,
            google_api_key=self.google_api_key
        )
        
        # Initialize Gemini integration
        self.gemini_integration = GeminiIntegration(google_api_key=self.google_api_key)
        
        # Initialize the system
        self._initialize_system()
        
        # Load test type mapping
        self.test_type_mapping = {
            "A": "Ability/Aptitude Test",
            "B": "Bio-Data and Situational Judgement",
            "C": "Competency-based Assessment",
            "D": "Development and 360-Degree Feedback",
            "E": "Assessment Center Exercise",
            "K": "Knowledge and Skills Test",
            "P": "Personality and Behaviour Assessment",
            "S": "Simulations"
        }
    
    def _initialize_system(self) -> None:
        """Initialize the recommendation system."""
        try:
            # Try to load existing vector database
            self.vector_store.load_vector_db()
            logger.info("Successfully initialized recommendation system with existing vector db")
        except Exception as e:
            logger.warning(f"Could not load existing vector database: {e}")
            
            # Create a new vector database
            try:
                logger.info("Creating new vector database from processed data")
                self.vector_store.create_vector_db()
                logger.info("Successfully created new vector database and initialized system")
            except Exception as create_error:
                logger.error(f"Error creating vector database: {create_error}")
                raise ValueError("Could not initialize the recommendation system") from create_error
    
    def get_recommendations(self, query: str, max_results: int = 10) -> pd.DataFrame:
        """
        Get assessment recommendations based on the query.
        
        Args:
            query: The natural language query or job description
            max_results: Maximum number of recommendations to return
            
        Returns:
            DataFrame with recommended assessments
        """
        # Extract max duration from query
        max_duration = self.gemini_integration.extract_duration_from_query(query)
        logger.info(f"Extracted max duration: {max_duration} minutes")
        
        # Expand the query for better retrieval
        expanded_query = self.gemini_integration.expand_query(query)
        logger.info(f"Expanded query: {expanded_query}")
        
        # Get initial search results
        search_results = self.vector_store.search_assessments(
            query=expanded_query, 
            k=30,  # Get more candidates to ensure enough unique results
            max_duration=max_duration
        )
        
        if not search_results:
            logger.warning("No assessment results found for the query")
            return pd.DataFrame({
                "Assessment Name": ["No matching assessments found"],
                "URL": [""],
                "Remote Testing": [""],
                "Adaptive/IRT Support": [""],
                "Duration": [""],
                "Test Type": [""]
            })
        
        # Deduplicate search results based on assessment name
        unique_results = []
        seen_names = set()
        for result in search_results:
            name = result.get("name", "Unnamed Assessment")
            if name not in seen_names:
                seen_names.add(name)
                unique_results.append(result)
        
        logger.info(f"Deduplicated results: {len(unique_results)} unique assessments from {len(search_results)} total")
        
        # Rank the results using Gemini
        ranked_results = self.gemini_integration.rank_assessments(
            query=query,
            search_results=unique_results,
            max_recommendations=max_results
        )
        
        # Format results as a DataFrame
        data = []
        for result in ranked_results:
            # Format the duration value
            duration = ""
            if result.get("assessment_time"):
                if isinstance(result["assessment_time"], (int, float)):
                    duration = f"{result['assessment_time']} minutes"
                else:
                    duration = str(result["assessment_time"])
            elif result.get("duration"):
                duration = f"{result['duration']} minutes"
            
            # Format test types with full descriptions
            test_types = result.get("test_type", "").split(", ")
            test_type_info = []
            for code in test_types:
                code = code.strip()
                if code in self.test_type_mapping:
                    test_type_info.append(f"{code} ({self.test_type_mapping[code]})")
                elif code:
                    test_type_info.append(code)
            
            test_type_str = ", ".join(test_type_info) if test_type_info else "Not specified"
            
            data.append({
                "Assessment Name": result.get("name", "Unnamed Assessment"),
                "URL": result.get("url", ""),
                "Remote Testing": "Yes" if result.get("remote_testing", "").lower() == "yes" else "No",
                "Adaptive/IRT Support": "Yes" if result.get("adaptive", "").lower() == "yes" else "No",
                "Duration": duration,
                "Test Type": test_type_str
            })
        
        return pd.DataFrame(data)
    
    def display_recommendations(self, query: str, max_results: int = 10) -> None:
        """
        Display assessment recommendations in a nice HTML table.
        
        Args:
            query: The natural language query or job description
            max_results: Maximum number of recommendations to return
        """
        recommendations_df = self.get_recommendations(query, max_results)
        
        # Create HTML table with hyperlinks
        html_table = recommendations_df.to_html(escape=False, index=False, render_links=True, classes="table table-striped")
        
        # Replace URL text with hyperlinks
        for i, row in recommendations_df.iterrows():
            url = row["URL"]
            if url and url.startswith("https://"):
                html_table = html_table.replace(f'<td>{url}</td>', f'<td><a href="{url}" target="_blank">{url}</a></td>')
        
        # Add styling
        styled_table = f"""
        <style>
        .table {{
            width: 100%;
            margin-bottom: 1rem;
            color: #212529;
            border-collapse: collapse;
        }}
        .table-striped tbody tr:nth-of-type(odd) {{
            background-color: rgba(0, 0, 0, 0.05);
        }}
        .table th, .table td {{
            padding: 0.75rem;
            vertical-align: top;
            border-top: 1px solid #dee2e6;
        }}
        .table thead th {{
            vertical-align: bottom;
            border-bottom: 2px solid #dee2e6;
            background-color: #4a89dc;
            color: white;
        }}
        .table a {{
            color: #4a89dc;
            text-decoration: none;
        }}
        .table a:hover {{
            text-decoration: underline;
        }}
        </style>
        <h3>SHL Assessment Recommendations for Query:</h3>
        <p><em>"{query}"</em></p>
        {html_table}
        """
        
        display(HTML(styled_table))
        
        # Also print a simple text version for environments without HTML support
        print(f"\nRecommended SHL Assessments for Query: '{query}'")
        for i, row in recommendations_df.iterrows():
            name = row["Assessment Name"]
            duration = row["Duration"]
            test_type = row["Test Type"]
            remote = row["Remote Testing"]
            adaptive = row["Adaptive/IRT Support"]
            
            print(f"{i+1}. {name}")
            print(f"   Duration: {duration}")
            print(f"   Test Type: {test_type}")
            print(f"   Remote Testing: {remote}")
            print(f"   Adaptive Support: {adaptive}")
            print()


def main():
    """Main function to demonstrate the SHL Recommendation System."""
    # Set up the SHL Recommendation System
    processed_data_path = "processed_assessments.json"  # Adjust path as needed
    db_directory = "chroma_db"
    
    try:
        # Initialize system
        recommendation_system = SHLRecommendationSystem(
            processed_data_path=processed_data_path,
            db_directory=db_directory
        )
        
        # Test queries
        test_queries = [
            "I need a Java programming assessment that can be completed in 30 minutes",
            "Looking for a personality assessment for managerial positions",
            "Need a SQL database test for experienced developers"
        ]
        
        for query in test_queries:
            print(f"\n\n{'='*80}\nTESTING QUERY: {query}\n{'='*80}\n")
            recommendation_system.display_recommendations(query, max_results=5)
            
            # Add a brief delay between queries
            time.sleep(2)
        
    except Exception as e:
        logger.error(f"Error in SHL Recommendation System demonstration: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()