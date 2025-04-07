import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from IPython.display import display, HTML
from langchain_core.documents import Document
import google.generativeai as genai
from dotenv import load_dotenv
import sys
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Attempt to use a compatible SQLite version
try:
    import sqlite3
    logger.info(f"Current SQLite version: {sqlite3.sqlite_version}")
    
    # Check SQLite version and attempt to upgrade
    if sqlite3.sqlite_version_info < (3, 35, 0):
        try:
            # Try to use pysqlite3
            import pysqlite3
            sys.modules['sqlite3'] = pysqlite3
            logger.info("Successfully upgraded SQLite using pysqlite3")
        except ImportError:
            logger.warning("Could not upgrade SQLite. Using system SQLite.")
except Exception as e:
    logger.error(f"Error checking SQLite version: {e}")

# Now import ChromaDB
import chromadb

# Get the API key from the environment and ensure it's a string
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise Exception("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")

class AssessmentVectorStore:
    """
    Creates and manages a vector database for SHL assessment data.
    """
    
    def __init__(self, 
                 processed_data_path: Optional[str] = None,
                 db_directory: str = "chroma_db",
                 collection_name: str = "shl_assessments",
                 google_api_key: Optional[str] = None):
        """
        Initialize the vector store.
        
        Args:
            processed_data_path: Path to the processed assessment data JSON file
            db_directory: Directory to store the Chroma database
            collection_name: Name of the vector collection
            google_api_key: Google API key for Gemini embeddings
        """
        self.processed_data_path = processed_data_path
        self.db_directory = db_directory
        self.collection_name = collection_name
        self.google_api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
        
        if not self.google_api_key:
            logger.warning("No Google API key provided. Set GOOGLE_API_KEY environment variable.")
        
        self.client = None
        self.collection = None
        self.retriever = None
        self.documents = []
        
        # Initialize ChromaDB client
        self._setup_client()
    
    def _setup_client(self) -> None:
        """Set up the ChromaDB client."""
        try:
            # Always use in-memory client for deployment
            self.client = chromadb.Client()
            logger.info("Using in-memory ChromaDB client for deployment compatibility")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {e}")
            raise
    
    def _get_or_create_collection(self) -> None:
        """Get or create the collection."""
        try:
            # Try to get the existing collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                count = self.collection.count()
                logger.info(f"Retrieved existing collection '{self.collection_name}' with {count} documents")
            except Exception as e:
                # Collection doesn't exist, create a new one
                logger.info(f"Collection '{self.collection_name}' not found. Creating a new one.")
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                )
                logger.info(f"Created new collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error getting or creating collection: {e}")
            raise
    
    def load_processed_data(self) -> List[Document]:
        """
        Load processed assessment data into Document objects.
        
        Returns:
            List of Document objects
        """
        if not self.processed_data_path:
            logger.error("No processed data path provided.")
            return []
            
        try:
            with open(self.processed_data_path, 'r') as f:
                data = json.load(f)
                
            documents = []
            for item in data:
                doc = Document(
                    page_content=item["content"],
                    metadata=item["metadata"]
                )
                documents.append(doc)
                
            logger.info(f"Loaded {len(documents)} documents from processed data")
            self.documents = documents
            return documents
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise
    
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using Google's embedding API."""
        embeddings = []
        batch_size = 10  # Process in smaller batches
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            logger.info(f"Embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
            
            try:
                batch_embeddings = []
                for text in batch:
                    # Handle empty text
                    if not text or text.strip() == "":
                        embedding = [0.0] * 768  # Default embedding dimension
                    else:
                        result = genai.embed_content(
                            model="models/text-embedding-004",
                            content=text,
                            task_type="retrieval_document"
                        )
                        embedding = result["embedding"]
                    batch_embeddings.append(embedding)
                
                embeddings.extend(batch_embeddings)
                
                # Add small delay to avoid rate limits
                import time
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error embedding batch {i//batch_size + 1}: {e}")
                # For failed batches, use zero embeddings
                for _ in range(len(batch)):
                    embeddings.append([0.0] * 768)
        
        return embeddings
    
    def _embed_query(self, query: str) -> List[float]:
        """Embed a query using Google's embedding API."""
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )
            return result["embedding"]
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return [0.0] * 768  # Return zero embedding as fallback
    
    def create_vector_db(self) -> None:
        """Create a vector database from processed documents."""
        # Load documents if not already loaded
        if not self.documents and self.processed_data_path:
            self.load_processed_data()
            
        if not self.documents:
            logger.error("No documents available for vector database creation.")
            return
        
        # Get or create collection
        self._get_or_create_collection()
        
        try:
            logger.info(f"Creating vector database with {len(self.documents)} documents")
            
            # Check if we have pre-computed embeddings
            embeddings_path = "FinalDataSource/precomputed_embeddings.json"
            
            if os.path.exists(embeddings_path):
                # Load pre-computed embeddings
                with open(embeddings_path, 'r') as f:
                    precomputed_data = json.load(f)
                    
                # Add documents to collection using pre-computed embeddings
                self.collection.add(
                    ids=precomputed_data["ids"],
                    embeddings=precomputed_data["embeddings"],
                    documents=precomputed_data["texts"],
                    metadatas=precomputed_data["metadatas"]
                )
                logger.info(f"Added {len(precomputed_data['ids'])} documents using pre-computed embeddings")
            
            else:
                # Process in smaller batches
                batch_size = 50
                
                for i in range(0, len(self.documents), batch_size):
                    batch = self.documents[i:i+batch_size]
                    logger.info(f"Processing batch {i//batch_size + 1}/{(len(self.documents) + batch_size - 1) // batch_size}")
                    
                    # Extract content and metadata
                    ids = [f"doc_{i+j}" for j in range(len(batch))]
                    texts = [doc.page_content for doc in batch]
                    
                    # Prepare metadata (ensure it's in correct format for ChromaDB)
                    metadatas = []
                    for doc in batch:
                        # Filter out None values and ensure all values are strings
                        metadata = {}
                        for k, v in doc.metadata.items():
                            if v is not None:
                                metadata[k] = str(v) if not isinstance(v, (int, float, bool)) else v
                            else:
                                metadata[k] = ""
                        metadatas.append(metadata)
                    
                    # Generate embeddings
                    embeddings = self._embed_texts(texts)
                    
                    # Add to collection
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=texts,
                        metadatas=metadatas
                    )
                    
                    logger.info(f"Added batch {i//batch_size + 1} to vector database")
                    
                    # Add small delay to avoid rate limits
                    import time
                    time.sleep(0.5)
            
            logger.info("Vector database created successfully")
            self.setup_retriever()
        except Exception as e:
            logger.error(f"Error creating vector database: {e}")
            raise
    
    def load_vector_db(self) -> None:
        """Load an existing vector database."""
        try:
            # Get the collection
            self._get_or_create_collection()
            
            # Check if collection exists and has documents
            if self.collection.count() == 0:
                logger.warning("Vector database is empty. Creating a new one.")
                if self.processed_data_path:
                    self.create_vector_db()
                else:
                    logger.error("No processed data path provided. Cannot create vector database.")
                    raise ValueError("Vector database is empty and no processed data path provided.")
            else:
                logger.info(f"Vector database loaded successfully with {self.collection.count()} documents")
                self.setup_retriever()
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            # Try to create a new database if loading fails
            if self.processed_data_path:
                logger.warning("Failed to load vector database. Creating a new one.")
                self.create_vector_db()
            else:
                raise
    
    def setup_retriever(self, k: int = 10) -> None:
        """
        Setup a retriever for the vector database.
        
        Args:
            k: Number of documents to retrieve
        """
        if not self.collection:
            logger.error("Collection not initialized. Call create_vector_db() or load_vector_db() first.")
            return
            
        try:
            logger.info(f"Setting up retriever with k={k}")
            # Nothing to do here, we'll use the collection directly for retrieval
            logger.info("Retriever setup successfully")
        except Exception as e:
            logger.error(f"Error setting up retriever: {e}")
            raise
    
    def retrieve(self, query: str, k: int = 10) -> List[Document]:
        """
        Retrieve relevant documents for the given query.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
        
        Returns:
            List of retrieved documents
        """
        if not self.collection:
            logger.warning("Collection not set up. Calling load_vector_db().")
            self.load_vector_db()
            
        try:
            logger.info(f"Retrieving documents for query: {query}")
            
            # Generate query embedding
            query_embedding = self._embed_query(query)
            
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            # Convert results to Document objects
            documents = []
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                metadata = results["metadatas"][0][i]
                content = results["documents"][0][i]
                documents.append(Document(page_content=content, metadata=metadata))
            
            logger.info(f"Retrieved {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []  # Return empty list on error
    
    def filter_by_duration(self, documents: List[Document], max_duration: Optional[int] = None) -> List[Document]:
        """
        Filter documents by assessment duration.
        
        Args:
            documents: List of documents to filter
            max_duration: Maximum assessment duration in minutes
        
        Returns:
            Filtered list of documents
        """
        if not max_duration:
            return documents
            
        filtered_docs = []
        for doc in documents:
            metadata = doc.metadata
            duration = metadata.get("duration")
            assessment_time = metadata.get("assessment_time")
            
            # If we have a numeric assessment time, use it for filtering
            if assessment_time is not None:
                try:
                    # Try to convert to float if it's a numeric string
                    if isinstance(assessment_time, str) and assessment_time.replace('.', '', 1).isdigit():
                        assessment_time = float(assessment_time)
                    
                    # If we have a numeric value, compare with max_duration
                    if isinstance(assessment_time, (int, float)) and assessment_time <= max_duration:
                        filtered_docs.append(doc)
                except (ValueError, TypeError):
                    # If conversion fails, include the document (benefit of doubt)
                    filtered_docs.append(doc)
            # If we have a numeric duration, use it for filtering
            elif duration is not None:
                try:
                    if isinstance(duration, str) and duration.replace('.', '', 1).isdigit():
                        duration = float(duration)
                    
                    if isinstance(duration, (int, float)) and duration <= max_duration:
                        filtered_docs.append(doc)
                except (ValueError, TypeError):
                    filtered_docs.append(doc)
            # If we don't have duration information, include the document
            else:
                filtered_docs.append(doc)
                
        logger.info(f"Filtered documents by duration ≤ {max_duration} minutes: {len(filtered_docs)} remaining from {len(documents)}")
        return filtered_docs
    
    def search_assessments(self, 
                          query: str, 
                          k: int = 10, 
                          max_duration: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for assessments based on the query.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            max_duration: Maximum assessment duration in minutes
        
        Returns:
            List of assessment information dictionaries
        """
        documents = self.retrieve(query, k=k)
        
        if not documents:
            logger.warning("No documents retrieved for query.")
            return []
        
        if max_duration:
            documents = self.filter_by_duration(documents, max_duration)
        
        results = []
        for doc in documents:
            results.append({
                "name": doc.metadata.get("name", "Unnamed Assessment"),
                "type": doc.metadata.get("type", "Unknown"),
                "test_type": doc.metadata.get("test_type", "Unknown"),
                "remote_testing": doc.metadata.get("remote_testing", "Unknown"),
                "adaptive": doc.metadata.get("adaptive", "Unknown"),
                "url": doc.metadata.get("url", ""),
                "duration": doc.metadata.get("duration"),
                "assessment_time": doc.metadata.get("assessment_time"),
                "content": doc.page_content
            })
        
        return results
                              
