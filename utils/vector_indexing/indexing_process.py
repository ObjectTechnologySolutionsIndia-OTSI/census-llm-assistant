import asyncio
import uuid
import logging
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

# Import your existing modules
from utils.emdeddings_inference.voyage_emded import AsyncVoyageEmbeddingsClient
from utils.vector_ingestions.pgvector_ingest import AsyncEmbeddingsDBManager

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmbeddingProcessor:
    """
    Main class to process documents and create embeddings using Voyage AI API
    and store them in PostgreSQL database.
    """
    
    def __init__(self, 
                 voyage_model: str = "voyage-3-large",
                 voyage_input_type: Optional[str] = "document",
                 db_config: Optional[Dict[str, str]] = None):
        """
        Initialize the embedding processor.
        
        Args:
            voyage_model: Voyage AI model to use for embeddings
            voyage_input_type: Input type for Voyage API ("document" or "query")
            db_config: Database configuration dictionary
        """
        self.voyage_model = voyage_model
        self.voyage_input_type = voyage_input_type
        
        # Database configuration from environment or provided config
        self.db_config = db_config or {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": os.getenv("POSTGRES_PORT", "5432"),
            "user": os.getenv("POSTGRES_USERNAME"),
            "password": os.getenv("POSTGRES_PASSWORD"),
            "database": os.getenv("POSTGRES_DBNAME"),
        }
        
        # Initialize clients
        self.voyage_client = None
        self.db_manager = None
        
    async def initialize(self):
        """Initialize the Voyage AI client and database manager."""
        try:
            # Initialize Voyage AI client
            self.voyage_client = AsyncVoyageEmbeddingsClient(
                model=self.voyage_model,
                input_type=self.voyage_input_type
            )
            
            # Initialize database manager
            self.db_manager = AsyncEmbeddingsDBManager(self.db_config)
            await self.db_manager.create_connection_pool()
            await self.db_manager.create_embeddings_table()
            
            logger.info("Successfully initialized Voyage AI client and database manager")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.voyage_client:
                await self.voyage_client.close()
            if self.db_manager:
                await self.db_manager.close_connection_pool()
            logger.info("Successfully cleaned up resources")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def validate_input_data(self, input_data: List[Dict[str, str]]) -> bool:
        """
        Validate that input data has the required structure.
        
        Args:
            input_data: List of dictionaries with keys: chunk, metadata_text, source_file
            
        Returns:
            bool: True if validation passes
        """
        required_keys = {'chunk', 'metadata_text', 'source_file'}
        
        if not input_data:
            logger.error("Input data is empty")
            return False
        
        for i, item in enumerate(input_data):
            if not isinstance(item, dict):
                logger.error(f"Item {i} is not a dictionary")
                return False
                
            missing_keys = required_keys - set(item.keys())
            if missing_keys:
                logger.error(f"Item {i} missing required keys: {missing_keys}")
                return False
                
            # Validate that values are strings
            for key in required_keys:
                if not isinstance(item[key], str):
                    logger.error(f"Item {i}: {key} must be a string")
                    return False
                    
            # Validate chunk is not empty
            if not item['chunk'].strip():
                logger.error(f"Item {i}: chunk cannot be empty")
                return False
        
        logger.info(f"Input validation passed for {len(input_data)} items")
        return True
    
    def extract_chunks(self, input_data: List[Dict[str, str]]) -> List[str]:
        """
        Extract chunks from input data.
        
        Args:
            input_data: List of dictionaries containing chunk data
            
        Returns:
            List[str]: List of text chunks
        """
        chunks = [item['chunk'] for item in input_data]
        logger.info(f"Extracted {len(chunks)} chunks from input data")
        return chunks
    
    async def get_embeddings_from_voyage(self, 
                                       chunks: List[str], 
                                       batch_size: int = 128) -> List[List[float]]:
        """
        Get embeddings from Voyage AI API.
        
        Args:
            chunks: List of text chunks to embed
            batch_size: Batch size for API calls
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            logger.info(f"Getting embeddings for {len(chunks)} chunks from Voyage AI")
            
            embeddings = await self.voyage_client.get_embeddings_batch(
                texts=chunks,
                batch_size=batch_size,
                input_type=self.voyage_input_type
            )
            
            logger.info(f"Successfully received {len(embeddings)} embeddings from Voyage AI")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings from Voyage AI: {e}")
            raise
    
    def combine_chunks_with_embeddings(self, 
                                     input_data: List[Dict[str, str]], 
                                     chunks: List[str], 
                                     embeddings: List[List[float]]) -> List[Dict[str, Any]]:
        """
        Combine chunks with embeddings and add UUIDs.
        
        Args:
            input_data: Original input data
            chunks: List of text chunks
            embeddings: List of embedding vectors
            
        Returns:
            List[Dict[str, Any]]: Updated data with IDs and embeddings
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})")
        
        if len(input_data) != len(chunks):
            raise ValueError(f"Mismatch between input_data ({len(input_data)}) and chunks ({len(chunks)})")
        
        # Create chunk-to-embedding mapping
        chunk_embedding_map = dict(zip(chunks, embeddings))
        
        # Update input data with IDs and embeddings
        updated_data = []
        for item in input_data:
            updated_item = item.copy()  # Create a copy to avoid modifying original
            
            # Add UUID
            updated_item['id'] = str(uuid.uuid4())
            
            # Add embedding
            chunk_text = item['chunk']
            if chunk_text in chunk_embedding_map:
                updated_item['embedding'] = chunk_embedding_map[chunk_text]
            else:
                logger.error(f"No embedding found for chunk: {chunk_text[:50]}...")
                raise ValueError(f"No embedding found for chunk")
            
            updated_data.append(updated_item)
        
        logger.info(f"Successfully combined {len(updated_data)} items with embeddings and IDs")
        return updated_data
    
    async def insert_into_database(self, data: List[Dict[str, Any]]) -> bool:
        """
        Insert data into PostgreSQL database.
        
        Args:
            data: List of dictionaries with embeddings data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Inserting {len(data)} records into database")
            
            success = await self.db_manager.bulk_insert_embeddings(data)
            
            if success:
                logger.info("Successfully inserted all records into database")
                
                # Get table statistics
                stats = await self.db_manager.get_table_stats()
                logger.info(f"Database stats after insertion: {stats}")
            else:
                logger.error("Failed to insert some records into database")
            
            return success
            
        except Exception as e:
            logger.error(f"Error inserting data into database: {e}")
            return False
    
    async def process_documents_to_embeddings(self, 
                                            input_data: List[Dict[str, str]],
                                            batch_size: int = 128,
                                            use_concurrent_db_insert: bool = False) -> bool:
        """
        Main function to process documents according to the specified logic.
        
        Args:
            input_data: List of dictionaries with keys: chunk, metadata_text, source_file
            batch_size: Batch size for Voyage API calls
            use_concurrent_db_insert: Whether to use concurrent database insertion
            
        Returns:
            bool: True if successful, False otherwise
        """
        
        try:
            logger.info("Starting document processing pipeline")
            
            # Step 1: Validate input data structure
            if not self.validate_input_data(input_data):
                return False
            
            # Step 2: Extract chunks from input data
            chunks = self.extract_chunks(input_data)
            
            # Step 3: Get embeddings from Voyage AI API
            embeddings = await self.get_embeddings_from_voyage(chunks, batch_size)
            
            if not embeddings:
                logger.error("Failed to get embeddings from Voyage AI")
                return False
            
            # Step 4: Combine chunks with embeddings and add UUIDs
            updated_data = self.combine_chunks_with_embeddings(input_data, chunks, embeddings)
            
            # Step 5: Insert into PostgreSQL database
            if use_concurrent_db_insert:
                success = await self.db_manager.bulk_insert_embeddings_concurrent(updated_data)
            else:
                success = await self.insert_into_database(updated_data)
            
            if success:
                logger.info("Successfully completed document processing pipeline")
                return True
            else:
                logger.error("Failed to complete document processing pipeline")
                return False
                
        except Exception as e:
            logger.error(f"Error in document processing pipeline: {e}")
            return False

async def main():
    """Example usage of the EmbeddingProcessor."""
    
    # Sample input data matching your specified structure
    input_data = [
        {
            "chunk": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            "metadata_text": "Introduction to ML concepts",
            "source_file": "ml_basics.pdf"
        },
        {
            "chunk": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
            "metadata_text": "Deep learning fundamentals", 
            "source_file": "deep_learning_guide.pdf"
        },
        {
            "chunk": "Natural language processing enables computers to understand, interpret, and generate human language.",
            "metadata_text": "NLP overview",
            "source_file": "nlp_handbook.pdf"
        },
        {
            "chunk": "Computer vision allows machines to interpret and understand visual information from the world.",
            "metadata_text": "Computer vision basics",
            "source_file": "cv_fundamentals.pdf"
        },
        {
            "chunk": "Reinforcement learning is a type of machine learning where agents learn to make decisions through trial and error.",
            "metadata_text": "RL introduction",
            "source_file": "reinforcement_learning.pdf"
        }
    ]
    
    # Initialize the processor
    processor = EmbeddingProcessor(
        voyage_model="voyage-3-large",
        voyage_input_type="document"
    )
    
    try:
        # Initialize components
        await processor.initialize()
        
        # Process documents and create embeddings
        success = await processor.process_documents_to_embeddings(
            input_data=input_data,
            batch_size=128,
            use_concurrent_db_insert=False  # Set to True for concurrent insertion
        )
        
        if success:
            print("✅ Successfully processed all documents and inserted embeddings into database!")
            
            # Optional: Demonstrate similarity search
            if processor.db_manager:
                # Search for similar embeddings using the first item's embedding
                # (In practice, you'd use a query embedding)
                stats = await processor.db_manager.get_table_stats()
                print(f"📊 Database statistics: {stats}")
        else:
            print("❌ Failed to process documents")
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Application error: {e}")
        
    finally:
        # Clean up resources
        await processor.cleanup()

# Example with error handling and logging
async def process_documents_with_error_handling(input_data: List[Dict[str, str]]):
    """
    Example function showing robust error handling.
    """
    processor = EmbeddingProcessor()
    
    try:
        await processor.initialize()
        
        # Process in smaller batches for better error handling
        batch_size = 50
        success = await processor.process_documents_to_embeddings(
            input_data=input_data,
            batch_size=batch_size
        )
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to process documents: {e}")
        return False
        
    finally:
        await processor.cleanup()

if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())
    
    # Example of processing a larger dataset
    # large_input_data = [...]  # Your large dataset
    # success = asyncio.run(process_documents_with_error_handling(large_input_data))