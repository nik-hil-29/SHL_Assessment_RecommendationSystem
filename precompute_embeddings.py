import json
import os
import logging
from vector_store import AssessmentVectorStore
from recommendation_system import SHLRecommendationSystem
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def precompute_embeddings():
    """
    Precompute embeddings for all assessment documents and save them to a file.
    This allows the Streamlit app to quickly load embeddings without computing them on startup.
    """
    try:
        # Initialize vector store
        processed_data_path = "FinalDataSource/processed_assessments.json"
        vector_store = AssessmentVectorStore(processed_data_path=processed_data_path)
        
        # Load documents
        logger.info("Loading assessment documents...")
        documents = vector_store.load_processed_data()
        
        if not documents:
            logger.error("No documents found to precompute embeddings")
            return
        
        # Prepare data
        ids = [f"doc_{i}" for i in range(len(documents))]
        texts = [doc.page_content for doc in documents]
        
        # Prepare metadata
        logger.info("Preparing metadata...")
        metadatas = []
        for doc in documents:
            metadata = {}
            for k, v in doc.metadata.items():
                if v is not None:
                    metadata[k] = str(v) if not isinstance(v, (int, float, bool)) else v
                else:
                    metadata[k] = ""
            metadatas.append(metadata)
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        embeddings = vector_store._embed_texts(texts)
        
        # Save precomputed data
        output_dir = "FinalDataSource"
        os.makedirs(output_dir, exist_ok=True)
        
        precomputed_data = {
            "ids": ids,
            "embeddings": embeddings,
            "texts": texts,
            "metadatas": metadatas
        }
        
        output_path = f"{output_dir}/precomputed_embeddings.json"
        logger.info(f"Saving precomputed embeddings to {output_path}...")
        
        with open(output_path, 'w') as f:
            json.dump(precomputed_data, f)
        
        logger.info(f"Successfully precomputed embeddings for {len(ids)} documents")
        
    except Exception as e:
        logger.error(f"Error precomputing embeddings: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    precompute_embeddings()
