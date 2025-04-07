"""
Data processor for SHL Assessment Recommendation System.
Processes the JSON files and prepares data for vector storage.
"""

import json
import os
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AssessmentDataProcessor:
    """
    Processes SHL assessment data from JSON files and prepares it for vector storage.
    """
    
    def __init__(self, prepackaged_path: str, individual_path: str):
        """
        Initialize with paths to the JSON data files.
        
        Args:
            prepackaged_path: Path to the prepackaged solutions JSON file
            individual_path: Path to the individual solutions JSON file
        """
        self.prepackaged_path = prepackaged_path
        self.individual_path = individual_path
        self.prepackaged_data = None
        self.individual_data = None
        self.processed_data = []
        
    def load_data(self) -> None:
        """Load data from JSON files."""
        try:
            logger.info(f"Loading prepackaged data from {self.prepackaged_path}")
            with open(self.prepackaged_path, 'r') as f:
                self.prepackaged_data = json.load(f)
                
            logger.info(f"Loading individual data from {self.individual_path}")
            with open(self.individual_path, 'r') as f:
                self.individual_data = json.load(f)
                
            logger.info("Data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def process_prepackaged_data(self) -> List[Dict[str, Any]]:
        """
        Process prepackaged assessment solutions data.
        
        Returns:
            List of processed assessment documents
        """
        if not self.prepackaged_data:
            logger.warning("Prepackaged data not loaded. Call load_data() first.")
            return []
        
        processed_docs = []
        
        for section in self.prepackaged_data:
            if "Pre_packaged_job_solutions" in section:
                for assessment in section["Pre_packaged_job_solutions"]:
                    # Clean and normalize test_type
                    test_type = assessment.get("test_type", "")
                    if test_type:
                        # Handle different formats (comma-separated or space-separated)
                        if "," in test_type:
                            test_types = [t.strip() for t in test_type.split(",")]
                        else:
                            test_types = [t.strip() for t in test_type.split()]
                        
                        # Create standardized string representation
                        test_type_normalized = ", ".join(sorted(test_types))
                    else:
                        test_type_normalized = ""
                    
                    # Prepare duration information
                    duration = assessment.get("duration")
                    assessment_time = assessment.get("assessment_time_duration")
                    
                    duration_info = ""
                    if duration:
                        duration_info = f"Duration: {duration} minutes. "
                    elif assessment_time:
                        if isinstance(assessment_time, (int, float)):
                            duration_info = f"Assessment time: {assessment_time} minutes. "
                        else:
                            duration_info = f"Assessment time: {assessment_time}. "
                    
                    # Prepare description field
                    desc = assessment.get("description", "No description available")
                    if not desc:
                        desc = "No description available"
                    
                    # Create document content with all relevant information
                    content = (
                        f"Assessment Name: {assessment.get('name', 'Unnamed Assessment')}\n"
                        f"Assessment Type: Prepackaged Job Solution\n"
                        f"Test Types: {test_type_normalized}\n"
                        f"{duration_info}"
                        f"Remote Testing: {assessment.get('remote_testing_support', 'Unknown')}\n"
                        f"Adaptive Support: {assessment.get('adaptive_support', 'Unknown')}\n"
                        f"Description: {desc}\n"
                        f"URL: {assessment.get('url', '')}"
                    )
                    
                    # Create metadata for filtering and identification
                    metadata = {
                        "name": assessment.get("name", ""),
                        "type": "prepackaged",
                        "test_type": test_type_normalized,
                        "remote_testing": assessment.get("remote_testing_support"),
                        "adaptive": assessment.get("adaptive_support"),
                        "url": assessment.get("url", ""),
                        "duration": assessment.get("duration"),
                        "assessment_time": assessment_time
                    }
                    
                    processed_docs.append({
                        "content": content,
                        "metadata": metadata
                    })
        
        logger.info(f"Processed {len(processed_docs)} prepackaged assessment documents")
        return processed_docs
    
    def process_individual_data(self) -> List[Dict[str, Any]]:
        """
        Process individual assessment solutions data.
        
        Returns:
            List of processed assessment documents
        """
        if not self.individual_data:
            logger.warning("Individual data not loaded. Call load_data() first.")
            return []
        
        processed_docs = []
        
        for section in self.individual_data:
            if "Individual_Test_Solutions" in section:
                for assessment in section["Individual_Test_Solutions"]:
                    # Clean and normalize test_type
                    test_type = assessment.get("test_type", "")
                    if test_type:
                        # Handle different formats (comma-separated or space-separated)
                        if "," in test_type:
                            test_types = [t.strip() for t in test_type.split(",")]
                        else:
                            test_types = [t.strip() for t in test_type.split()]
                        
                        # Create standardized string representation
                        test_type_normalized = ", ".join(sorted(test_types))
                    else:
                        test_type_normalized = ""
                    
                    # Prepare duration information
                    duration = assessment.get("duration")
                    assessment_time = assessment.get("assessment_time_duration")
                    
                    duration_info = ""
                    if duration:
                        duration_info = f"Duration: {duration} minutes. "
                    elif assessment_time:
                        if isinstance(assessment_time, (int, float)):
                            duration_info = f"Assessment time: {assessment_time} minutes. "
                        else:
                            duration_info = f"Assessment time: {assessment_time}. "
                    
                    # Prepare description field
                    desc = assessment.get("description", "No description available")
                    if not desc:
                        desc = "No description available"
                    
                    # Create document content with all relevant information
                    content = (
                        f"Assessment Name: {assessment.get('name', 'Unnamed Assessment')}\n"
                        f"Assessment Type: Individual Test Solution\n"
                        f"Test Types: {test_type_normalized}\n"
                        f"{duration_info}"
                        f"Remote Testing: {assessment.get('remote_testing_support', 'Unknown')}\n"
                        f"Adaptive Support: {assessment.get('adaptive_support', 'Unknown')}\n"
                        f"Description: {desc}\n"
                        f"URL: {assessment.get('url', '')}"
                    )
                    
                    # Create metadata for filtering and identification
                    metadata = {
                        "name": assessment.get("name", ""),
                        "type": "individual",
                        "test_type": test_type_normalized,
                        "remote_testing": assessment.get("remote_testing_support"),
                        "adaptive": assessment.get("adaptive_support"),
                        "url": assessment.get("url", ""),
                        "duration": assessment.get("duration"),
                        "assessment_time": assessment_time
                    }
                    
                    processed_docs.append({
                        "content": content,
                        "metadata": metadata
                    })
        
        logger.info(f"Processed {len(processed_docs)} individual assessment documents")
        return processed_docs
    
    def process_all_data(self) -> List[Dict[str, Any]]:
        """
        Process all assessment data and combine into a single list.
        
        Returns:
            List of all processed assessment documents
        """
        prepackaged_docs = self.process_prepackaged_data()
        individual_docs = self.process_individual_data()
        
        self.processed_data = prepackaged_docs + individual_docs
        logger.info(f"Total processed documents: {len(self.processed_data)}")
        
        return self.processed_data
    
    def get_test_type_mapping(self) -> Dict[str, str]:
        """
        Create a mapping of test type codes to descriptions.
        
        Returns:
            Dictionary mapping test type codes to descriptions
        """
        # Standard test type codes and their meanings
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
    
    def save_processed_data(self, output_path: str) -> None:
        """
        Save processed data to a JSON file.
        
        Args:
            output_path: Path to save the processed data
        """
        if not self.processed_data:
            logger.warning("No processed data to save. Call process_all_data() first.")
            return
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.processed_data, f, indent=2)
            logger.info(f"Processed data saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise

def main():
    """Main function to demonstrate the data processor."""
    # File paths - adjust these to your actual file locations
    prepackaged_path = "data_source/shl_enhanced_solutions_prepacksol.json"
    individual_path = "data_source/shl_enhanced_solutions.json"
    output_path = "FinalDataSource/processed_assessments.json"
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize and run the processor
    processor = AssessmentDataProcessor(prepackaged_path, individual_path)
    processor.load_data()
    processor.process_all_data()
    processor.save_processed_data(output_path)
    
    logger.info("Data processing complete.")

if __name__ == "__main__":
    main()