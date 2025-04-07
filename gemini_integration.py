import os
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from IPython.display import display, HTML
from langchain_core.documents import Document
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# Get the API key from the environment and ensure it's a string
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise Exception("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeminiIntegration:
    """
    Integrates Google's Gemini model for enhanced assessment recommendations.
    """
    
    def __init__(self, google_api_key: Optional[str] = None):
        """
        Initialize the Gemini integration.
        
        Args:
            google_api_key: Google API key for Gemini access
        """
        self.google_api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
        
        if not self.google_api_key:
            logger.error("No Google API key provided. Set GOOGLE_API_KEY environment variable.")
            raise ValueError("Google API key is required")
        
        self.setup_gemini()
        self.test_type_mapping = self._get_test_type_mapping()
    
    def setup_gemini(self) -> None:
        """Set up the Gemini model."""
        try:
            logger.info("Setting up Gemini model")
            genai.configure(api_key=self.google_api_key)
            
            # Initialize Gemini model
            generation_config = {
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
            
            self.model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config=generation_config
            )
            
            logger.info("Gemini model setup successfully")
        except Exception as e:
            logger.error(f"Error setting up Gemini model: {e}")
            raise
    
    def _get_test_type_mapping(self) -> Dict[str, str]:
        """
        Get mappings of test type codes to descriptions.
        
        Returns:
            Dictionary mapping test type codes to descriptions
        """
        return {
            "A": "Ability/Aptitude Test",
            "B": "Bio-Data and Situational Judgement",
            "C": "Competency-based Assessment",
            "D": "Development and 360-Degree Feedback",
            "E": "Assessment Center Exercise",
            "K": "Knowledge and Skills Test",
            "P": "Personality and Behaviour Assessment",
            "S": "Simulations"
        }
    
    def _call_with_retry(self, func, *args, max_retries=3, **kwargs):
        """Call a function with retry logic."""
        retries = 0
        last_exception = None
        
        while retries < max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                wait_time = 2 ** retries  # Exponential backoff
                logger.warning(f"API call failed. Retrying in {wait_time} seconds. Error: {e}")
                time.sleep(wait_time)
                retries += 1
        
        # If we get here, all retries failed
        logger.error(f"API call failed after {max_retries} retries. Last error: {last_exception}")
        raise last_exception
    
    def extract_duration_from_query(self, query: str) -> Optional[int]:
        """
        Extract maximum assessment duration from query.
        
        Args:
            query: User query string
        
        Returns:
            Maximum duration in minutes if found, None otherwise
        """
        try:
            # Create a focused prompt for duration extraction
            user_prompt = f"""
            Extract the maximum assessment duration mentioned in this query: "{query}"
            
            If there is no duration mentioned, return "None".
            If a duration is mentioned, return only the number of minutes as an integer.
            
            Examples:
            - Input: "I need an assessment that takes less than 30 minutes"
              Output: 30
            - Input: "Find me assessments that can be done within 45 mins"
              Output: 45
            - Input: "I'm looking for a Java assessment"
              Output: None
            """
            
            # Use retry logic for the API call
            response = self._call_with_retry(
                self.model.generate_content,
                user_prompt,
                max_retries=3
            )
            
            # Extract the number from the result
            result_text = response.text.strip()
            if result_text.lower() == "none":
                return None
                
            # Try to parse the result as an integer
            try:
                duration = int(result_text)
                logger.info(f"Extracted duration from query: {duration} minutes")
                return duration
            except ValueError:
                logger.warning(f"Could not parse duration from result: {result_text}")
                return None
        except Exception as e:
            logger.error(f"Error extracting duration from query: {e}")
            return None
    
    def expand_query(self, query: str) -> str:
        """
        Expand and improve the query to enhance retrieval.
        
        Args:
            query: Original user query
        
        Returns:
            Expanded and improved query
        """
        try:
            # Create a focused prompt for query expansion
            user_prompt = f"""
            You are an expert in HR assessments and testing. Your task is to expand the following query to improve retrieval from a database of SHL assessments:
            
            Original query: {query}
            
            Consider including:
            1. Relevant assessment types (Knowledge Test, Ability/Aptitude Test, Behavioral Assessment, Competency-based Assessment, Personality Assessment, Simulation)
            2. Specific skills or competencies mentioned or implied
            3. Job roles or levels that might be relevant
            4. Any technical skills (programming languages, software, etc.)
            
            DO NOT add unmentioned specific durations, but DO preserve any duration constraints mentioned in the original query.
            DO NOT make up specific assessment names.
            
            Return only the expanded query without explanation or preamble.
            """
            
            # Use retry logic for the API call
            response = self._call_with_retry(
                self.model.generate_content,
                user_prompt,
                max_retries=3
            )
            
            expanded_query = response.text.strip()
            logger.info(f"Expanded query: {expanded_query}")
            return expanded_query
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return query  # Return original query if expansion fails
    
    def rank_assessments(self, query: str, search_results: List[Dict[str, Any]], max_recommendations: int = 10) -> List[Dict[str, Any]]:
        """
        Rank the assessment results based on relevance to query.
        
        Args:
            query: User query string
            search_results: List of assessment search results
            max_recommendations: Maximum number of recommendations to return
            
        Returns:
            List of ranked assessment results
        """
        if not search_results:
            logger.warning("No search results provided for ranking")
            return []
            
        # If we have fewer results than requested, return all of them
        if len(search_results) <= max_recommendations:
            return search_results
        
        try:
            # Prepare simplified context with search results to avoid token limits
            formatted_results = []
            for i, result in enumerate(search_results):
                # Format duration information
                duration_info = ""
                if result.get("assessment_time"):
                    if isinstance(result["assessment_time"], (int, float)):
                        duration_info = f"Assessment time: {result['assessment_time']} minutes"
                    else:
                        duration_info = f"Assessment time: {result['assessment_time']}"
                elif result.get("duration"):
                    duration_info = f"Duration: {result['duration']} minutes"
                else:
                    duration_info = "Duration: Not specified"
                
                # Create a shorter, more focused description of each assessment
                formatted_result = f"""
                Assessment {i}:
                Name: {result.get('name', 'Unnamed Assessment')}
                Type: {result.get('type', 'Unknown').capitalize()} Solution
                Test Types: {result.get('test_type', 'Unknown')}
                {duration_info}
                Remote Testing: {result.get('remote_testing', 'Unknown')}
                Adaptive: {result.get('adaptive', 'Unknown')}
                """
                formatted_results.append(formatted_result)
            
            context = "\n".join(formatted_results)
            
            # Create a focused prompt for ranking
            user_prompt = f"""
            You are an expert in HR assessments and testing. Rank these assessments based on their relevance to this query: "{query}"
            
            Available Assessment Options (referenced by index 0 to {len(search_results)-1}):
            {context}
            
            For each assessment, evaluate:
            1. How well it matches the skills/competencies in the query
            2. How appropriate the assessment type is for the scenario
            3. How well it meets any time constraints mentioned
            4. How relevant it is to any job role or level mentioned
            
            Return a JSON array of integers representing the indices of the top {max_recommendations} most relevant assessments in order of relevance.
            Example: [3, 0, 7, 1, 5]
            
            Only return the JSON array and nothing else.
            """
            
            # Use retry logic for the API call
            response = self._call_with_retry(
                self.model.generate_content,
                user_prompt,
                max_retries=3
            )
            
            response_text = response.text.strip()
            
            # Try to parse the response as a JSON array
            import re
            import json
            
            # Extract JSON array using regex
            json_array_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_array_match:
                json_array_str = json_array_match.group(0)
                try:
                    indices = json.loads(json_array_str)
                    # Validate indices are within range
                    valid_indices = [i for i in indices if 0 <= i < len(search_results)]
                    # Limit to requested number
                    valid_indices = valid_indices[:max_recommendations]
                    
                    # Return ranked results
                    return [search_results[i] for i in valid_indices]
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse response as JSON: {response_text}")
            
            # Fallback: return the original results up to max_recommendations
            logger.warning("Using fallback ranking method")
            return search_results[:max_recommendations]
            
        except Exception as e:
            logger.error(f"Error ranking assessments: {e}")
            # Return the original results up to max_recommendations
            return search_results[:max_recommendations]