step 1: take input of user query along with options listed below

step 2: perform validation of options listed below
options : 
1. Enhance user query for optimal semantic search : (yes/no)
2. No.of variations of user query for semantic search: Number int  
3. system prompt for generating user query variations: string
   Note for option validations: Option 2 cannot be less than 0 if option 1 is set to yes
   option 3 cannot be empty if option 1 is set to yes


step 3: 
If option 1 is set to yes then call the Anthropic LLM model as listed below to generate the variations of user query for semantic search
from .anthropic_apis import AsyncAnthropicClient
class AsyncAnthropicClient:
   (method) async def chat(
      self: Self@AsyncAnthropicClient,
      system_prompt: str,
      messages: list
   ) -> str

Note : the input message must be passed in the following format
[{"role":"user", "content": <user query>}]

response of LLM model is in the following format
{ "search_strings": [   "Marital status by age and sex State District 2011",   "Primary census abstract marital status married women State 2011",   "Population by marital status age sex Andhra Pradesh 2011" ]}

step 4:
if option 1 is set to yes and step3 might return one or more than one item in the list search_strings then asynchronously or parallely perform semantic serach on pg vector database by following the below steps.
 1. generate the embedding for all the search strings using the voyageembedding.
 2. perform the semantic search on the pg vector to get top 100 simallar chuncks
 3. club the results of all the semantic searches and remove the duplicates.

if option 1 is set to no perform the below steps by taking user query as input
 1. generate the embedding for user query using the voyageembedding.
 2. perform the semantic search on the pg vector to get top 100 simallar chuncks
 3. club the results of all the semantic searches and remove the duplicates.
Note below is the existing code for voyage embedding:
```
"""
Voyage AI Embeddings Client
===========================

This module provides an asynchronous client for generating text embeddings using Voyage AI's embedding models.
Supports both single text and batch processing with proper error handling and resource management.

Author: Assistant
Date: 2025
"""

import os
import asyncio
from typing import List, Union, Optional
from dotenv import load_dotenv
import voyageai
import numpy as np

# Load environment variables
load_dotenv(override=True)


class AsyncVoyageEmbeddingsClient:
    """
    An asynchronous wrapper for Voyage AI's text embedding API.
    
    This client provides methods for generating embeddings from text using various Voyage AI models.
    Supports both single text and batch processing with automatic chunking for large batches.
    """
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "voyage-3-large",
                 input_type: Optional[str] = None,
                 truncation: bool = True):
        """
        Initialize the Voyage AI embeddings client.
        
        Args:
            api_key (str, optional): Voyage AI API key. If not provided, loads from VOYAGE_API_KEY env var.
            model (str): Embedding model name. Options include:
                - voyage-3-large (1024 dimensions, best performance)
                - voyage-3-lite (512 dimensions, faster/cheaper)
                - voyage-code-2 (1536 dimensions, optimized for code)
                - voyage-finance-2 (1024 dimensions, optimized for finance)
                - voyage-multilingual-2 (1024 dimensions, multilingual support)
                - voyage-law-2 (1024 dimensions, optimized for legal documents)
            input_type (str, optional): Type of input text. Options:
                - "query": For search queries
                - "document": For documents to be searched
                - None: Let the model decide
            truncation (bool): Whether to truncate input text if too long. Default is True.
        
        Raises:
            ValueError: If API key is not provided or found in environment.
        """
        self.api_key = api_key or os.getenv('VOYAGE_API_KEY')
        if not self.api_key:
            raise ValueError("Voyage AI API key is not set. Please set VOYAGE_API_KEY in environment or pass as argument.")
        
        self.client = voyageai.Client(api_key=self.api_key)
        self.model = model
        self.input_type = input_type
        self.truncation = truncation
        
        # Model specifications
        self.model_specs = {
            "voyage-3-large": {"dimensions": 1024, "context_length": 32000},
            "voyage-3-lite": {"dimensions": 512, "context_length": 32000},
            "voyage-code-2": {"dimensions": 1536, "context_length": 16000},
            "voyage-finance-2": {"dimensions": 1024, "context_length": 32000},
            "voyage-multilingual-2": {"dimensions": 1024, "context_length": 32000},
            "voyage-law-2": {"dimensions": 1024, "context_length": 32000},
        }
    
    async def get_embedding(self, text: str, input_type: Optional[str] = None) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text (str): Input text to embed.
            input_type (str, optional): Override default input type for this request.
            
        Returns:
            List[float]: Embedding vector as a list of floats.
            
        Raises:
            Exception: If the API call fails.
        """
        try:
            # Prepare request parameters
            params = {
                "model": self.model,
                "input": text,
                "truncation": self.truncation
            }
            
            # Add input_type if specified
            current_input_type = input_type or self.input_type
            if current_input_type:
                params["input_type"] = current_input_type
            
            # Execute in thread pool since voyageai is synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.client.embed([text], **{k: v for k, v in params.items() if k != 'input'})
            )
            
            return response.embeddings[0]
            
        except Exception as e:
            raise Exception(f"Failed to generate embedding for text: {str(e)}")
    
    async def get_embeddings_batch(self, 
                                   texts: List[str],
                                   batch_size: int = 128,
                                   input_type: Optional[str] = None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts (List[str]): List of texts to embed.
            batch_size (int): Number of texts to process in each batch. Default is 128.
            input_type (str, optional): Override default input type for this request.
            
        Returns:
            List[List[float]]: List of embedding vectors.
            
        Raises:
            Exception: If any API call fails.
        """
        if not texts:
            return []
        
        embeddings = []
        
        try:
            # Process texts in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Prepare request parameters
                params = {
                    "model": self.model,
                    "truncation": self.truncation
                }
                
                # Add input_type if specified
                current_input_type = input_type or self.input_type
                if current_input_type:
                    params["input_type"] = current_input_type
                
                # Execute in thread pool since voyageai is synchronous
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda batch=batch: self.client.embed(batch, **params)
                )
                
                # Extract embeddings from response
                batch_embeddings = response.embeddings
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            raise Exception(f"Failed to generate batch embeddings: {str(e)}")
    
    async def get_embeddings_concurrent(self, 
                                        texts: List[str],
                                        max_concurrent: int = 5,
                                        input_type: Optional[str] = None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts concurrently.
        
        Args:
            texts (List[str]): List of texts to embed.
            max_concurrent (int): Maximum number of concurrent requests. Default is 5.
            input_type (str, optional): Override default input type for this request.
            
        Returns:
            List[List[float]]: List of embedding vectors in the same order as input texts.
            
        Raises:
            Exception: If any API call fails.
        """
        if not texts:
            return []
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def get_single_embedding(text: str) -> List[float]:
            async with semaphore:
                return await self.get_embedding(text, input_type)
        
        # Execute all requests concurrently
        tasks = [get_single_embedding(text) for text in texts]
        embeddings = await asyncio.gather(*tasks)
        
        return embeddings
    
    def cosine_similarity(self, 
                         embedding1: List[float], 
                         embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1 (List[float]): First embedding vector.
            embedding2 (List[float]): Second embedding vector.
            
        Returns:
            float: Cosine similarity score between -1 and 1.
        """
        # Convert to numpy arrays for easier computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model.
        
        Returns:
            dict: Model specifications including dimensions and context length.
        """
        return self.model_specs.get(self.model, {"dimensions": "unknown", "context_length": "unknown"})
    
    async def close(self):
        """Close the client session (placeholder for consistency)."""
        # Voyage AI client doesn't require explicit closing
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Example usage and testing
async def main():
    """Example usage of the Voyage AI embeddings client."""
    
    async with AsyncVoyageEmbeddingsClient() as client:
        # Single text embedding
        print("=== Single Text Embedding ===")
        text = "The quick brown fox jumps over the lazy dog."
        embedding = await client.get_embedding(text)
        print(f"Text: {text}")
        print(f"Embedding dimensions: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        print(f"Model info: {client.get_model_info()}")
        
        # Batch embedding
        print("\n=== Batch Embeddings ===")
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand human language.",
            "Computer vision enables machines to interpret visual information."
        ]
        
        embeddings = await client.get_embeddings_batch(texts)
        print(f"Generated {len(embeddings)} embeddings")
        
        # Calculate similarities
        print("\n=== Similarity Analysis ===")
        for i, text1 in enumerate(texts):
            for j, text2 in enumerate(texts[i+1:], i+1):
                similarity = client.cosine_similarity(embeddings[i], embeddings[j])
                print(f"Similarity between text {i+1} and {j+1}: {similarity:.4f}")


# Test with different models and input types
async def test_different_models():
    """Test embeddings with different Voyage AI models."""
    
    models = [
        "voyage-3-large",
        "voyage-3-lite",
        "voyage-code-2",
        "voyage-finance-2",
        "voyage-multilingual-2",
    ]
    
    text = "Artificial intelligence is transforming the world."
    
    for model in models:
        try:
            async with AsyncVoyageEmbeddingsClient(model=model) as client:
                embedding = await client.get_embedding(text)
                model_info = client.get_model_info()
                print(f"Model: {model}")
                print(f"Expected dimensions: {model_info['dimensions']}")
                print(f"Actual embedding size: {len(embedding)}")
                print(f"Sample values: {embedding[:3]}")
                print("-" * 50)
        except Exception as e:
            print(f"Error with model {model}: {str(e)}")


# Test with different input types
async def test_input_types():
    """Test embeddings with different input types."""
    
    query_text = "What is machine learning?"
    document_text = "Machine learning is a method of data analysis that automates analytical model building."
    
    async with AsyncVoyageEmbeddingsClient() as client:
        # Generate embeddings with different input types
        query_embedding = await client.get_embedding(query_text, input_type="query")
        doc_embedding = await client.get_embedding(document_text, input_type="document")
        
        # Calculate similarity
        similarity = client.cosine_similarity(query_embedding, doc_embedding)
        
        print("=== Input Type Testing ===")
        print(f"Query: {query_text}")
        print(f"Document: {document_text}")
        print(f"Query-Document similarity: {similarity:.4f}")


# Test concurrent processing
async def test_concurrent_processing():
    """Test concurrent embedding generation."""
    
    texts = [
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret and analyze visual information.",
        "Robotics combines AI with mechanical engineering for autonomous systems.",
        "Machine learning algorithms learn patterns from data without explicit programming.",
        "Deep learning uses neural networks with multiple layers for complex pattern recognition."
    ]
    
    async with AsyncVoyageEmbeddingsClient() as client:
        print("=== Concurrent Processing Test ===")
        
        # Time the concurrent processing
        import time
        start_time = time.time()
        
        embeddings = await client.get_embeddings_concurrent(texts, max_concurrent=3)
        
        end_time = time.time()
        
        print(f"Processed {len(embeddings)} embeddings in {end_time - start_time:.2f} seconds")
        print(f"Average embedding dimensions: {len(embeddings[0])}")


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())
    
    # Uncomment to test different models
    # asyncio.run(test_different_models())
    
    # Uncomment to test input types
    # asyncio.run(test_input_types())
    
    # Uncomment to test concurrent processing
    # asyncio.run(test_concurrent_processing())
```

Note below is the existing code for semantic search
```
import asyncio
import asyncpg
import logging
from typing import List, Dict, Any, Optional
import uuid
from contextlib import asynccontextmanager
import sys
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("embeddings_insert.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


class AsyncEmbeddingsDBManager:
    def __init__(self, db_config: Dict[str, str]):
        """
        Initialize the async database manager with connection parameters.

        Args:
            db_config: Dictionary containing database connection parameters
                      {'host': 'localhost', 'port': '5432', 'database': 'your_db',
                       'user': 'your_user', 'password': 'your_password'}
        """
        self.db_config = db_config
        self.batch_size = 100
        self.connection_pool: Optional[asyncpg.Pool] = None

    async def create_connection_pool(self, min_size: int = 10, max_size: int = 20):
        """Create a connection pool for better performance."""
        try:
            self.connection_pool = await asyncpg.create_pool(
                min_size=min_size, max_size=max_size, **self.db_config
            )
            logger.info(
                f"Successfully created connection pool (min: {min_size}, max: {max_size})"
            )
        except Exception as e:
            logger.error(f"Error creating connection pool: {e}")
            raise

    async def close_connection_pool(self):
        """Close the connection pool."""
        if self.connection_pool:
            await self.connection_pool.close()
            logger.info("Connection pool closed")

    @asynccontextmanager
    async def get_db_connection(self):
        """Context manager for database connections."""
        if self.connection_pool:
            # Use connection from pool
            async with self.connection_pool.acquire() as conn:
                try:
                    logger.debug("Successfully acquired connection from pool")
                    yield conn
                except Exception as e:
                    logger.error(f"Error with pooled connection: {e}")
                    raise
        else:
            # Fallback to direct connection
            conn = None
            try:
                conn = await asyncpg.connect(**self.db_config)
                logger.info("Successfully connected to PostgreSQL database")
                yield conn
            except Exception as e:
                logger.error(f"Error connecting to PostgreSQL: {e}")
                raise
            finally:
                if conn:
                    await conn.close()
                    logger.info("Database connection closed")

    async def create_embeddings_table(self):
        """Create the embeddings table if it doesn't exist."""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS document_embeddings (
            id UUID PRIMARY KEY,
            chunk TEXT NOT NULL,
            embedding VECTOR(1024),  -- Adjust dimension based on your embedding model
            metadata_text TEXT,
            source_file TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_embeddings_source_file ON document_embeddings(source_file);
        CREATE INDEX IF NOT EXISTS idx_embeddings_created_at ON document_embeddings(created_at);
        
        -- Create a trigger to update the updated_at timestamp
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        
        DROP TRIGGER IF EXISTS update_embeddings_updated_at ON document_embeddings;
        CREATE TRIGGER update_embeddings_updated_at
            BEFORE UPDATE ON document_embeddings
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        """

        try:
            async with self.get_db_connection() as conn:
                await conn.execute(create_table_query)
                logger.info("Successfully created/verified document_embeddings table")
        except Exception as e:
            logger.error(f"Error creating table: {e}")
            raise

    async def validate_input_data(self, data: List[Dict[str, Any]]) -> bool:
        """
        Validate the input data structure.

        Args:
            data: List of dictionaries with required fields

        Returns:
            bool: True if validation passes
        """
        required_fields = {"id", "chunk", "embedding", "metadata_text", "source_file"}

        if not data:
            logger.warning("Input data is empty")
            return False

        for i, record in enumerate(data):
            if not isinstance(record, dict):
                logger.error(f"Record {i} is not a dictionary")
                return False

            missing_fields = required_fields - record.keys()
            if missing_fields:
                logger.error(f"Record {i} missing required fields: {missing_fields}")
                return False

            # Validate UUID format
            try:
                uuid.UUID(str(record["id"]))
            except ValueError:
                logger.error(f"Record {i} has invalid UUID format for 'id' field")
                return False

            # Validate embedding is a list/array
            if not isinstance(record["embedding"], (list, tuple)):
                logger.error(f"Record {i} embedding is not a list/array")
                return False

        logger.info(f"Input data validation passed for {len(data)} records")
        return True

    async def bulk_insert_embeddings(self, data: List[Dict[str, Any]]) -> bool:
        """
        Perform async bulk insert of embeddings data with batching.

        Args:
            data: List of dictionaries containing embedding data

        Returns:
            bool: True if all inserts successful
        """
        if not await self.validate_input_data(data):
            return False

        try:
            async with self.get_db_connection() as conn:
                total_records = len(data)
                successful_inserts = 0
                failed_inserts = 0

                logger.info(
                    f"Starting bulk insert of {total_records} records with batch size {self.batch_size}"
                )

                # Process data in batches
                for i in range(0, total_records, self.batch_size):
                    batch = data[i : i + self.batch_size]
                    batch_num = (i // self.batch_size) + 1

                    try:
                        # Prepare batch data for insertion
                        batch_records = []
                        for record in batch:
                            batch_records.append(
                                (
                                    record["id"],
                                    record["chunk"],
                                    json.dumps(
                                        record["embedding"]
                                    ),  # Convert list to JSON string for asyncpg
                                    record["metadata_text"],
                                    record["source_file"],
                                )
                            )

                        # Execute batch insert using asyncpg's executemany
                        insert_query = """
                        INSERT INTO document_embeddings (id, chunk, embedding, metadata_text, source_file)
                        VALUES ($1, $2, $3::vector, $4, $5)
                        ON CONFLICT (id) DO UPDATE SET
                            chunk = EXCLUDED.chunk,
                            embedding = EXCLUDED.embedding,
                            metadata_text = EXCLUDED.metadata_text,
                            source_file = EXCLUDED.source_file,
                            updated_at = CURRENT_TIMESTAMP
                        """

                        await conn.executemany(insert_query, batch_records)

                        successful_inserts += len(batch)

                        logger.info(
                            f"Batch {batch_num}: Successfully inserted {len(batch)} records "
                            f"({successful_inserts}/{total_records} total)"
                        )

                    except Exception as e:
                        logger.error(f"Batch {batch_num}: Error inserting records: {e}")
                        failed_inserts += len(batch)
                        continue

                logger.info(
                    f"Bulk insert completed. Success: {successful_inserts}, Failed: {failed_inserts}"
                )
                return failed_inserts == 0

        except Exception as e:
            logger.error(f"Unexpected error during bulk insert: {e}")
            return False

    async def bulk_insert_embeddings_concurrent(
        self, data: List[Dict[str, Any]], max_concurrent_batches: int = 5
    ) -> bool:
        """
        Perform concurrent bulk insert of embeddings data with batching.

        Args:
            data: List of dictionaries containing embedding data
            max_concurrent_batches: Maximum number of concurrent batch operations

        Returns:
            bool: True if all inserts successful
        """
        if not await self.validate_input_data(data):
            return False

        try:
            total_records = len(data)
            logger.info(
                f"Starting concurrent bulk insert of {total_records} records with batch size {self.batch_size}"
            )

            # Create batches
            batches = []
            for i in range(0, total_records, self.batch_size):
                batch = data[i : i + self.batch_size]
                batches.append(batch)

            # Process batches concurrently
            semaphore = asyncio.Semaphore(max_concurrent_batches)

            async def process_batch(batch_data: List[Dict[str, Any]], batch_num: int):
                async with semaphore:
                    try:
                        async with self.get_db_connection() as conn:
                            # Prepare batch data for insertion
                            batch_records = []
                            for record in batch_data:
                                batch_records.append(
                                    (
                                        record["id"],
                                        record["chunk"],
                                        json.dumps(record["embedding"]),
                                        record["metadata_text"],
                                        record["source_file"],
                                    )
                                )

                            # Execute batch insert
                            insert_query = """
                            INSERT INTO document_embeddings (id, chunk, embedding, metadata_text, source_file)
                            VALUES ($1, $2, $3::vector, $4, $5)
                            ON CONFLICT (id) DO UPDATE SET
                                chunk = EXCLUDED.chunk,
                                embedding = EXCLUDED.embedding,
                                metadata_text = EXCLUDED.metadata_text,
                                source_file = EXCLUDED.source_file,
                                updated_at = CURRENT_TIMESTAMP
                            """

                            await conn.executemany(insert_query, batch_records)

                            logger.info(
                                f"Batch {batch_num}: Successfully inserted {len(batch_data)} records"
                            )
                            return len(batch_data)

                    except Exception as e:
                        logger.error(f"Batch {batch_num}: Error inserting records: {e}")
                        return 0

            # Execute all batches concurrently
            tasks = [process_batch(batch, i + 1) for i, batch in enumerate(batches)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Calculate results
            successful_inserts = sum(
                result for result in results if isinstance(result, int)
            )
            failed_inserts = total_records - successful_inserts

            logger.info(
                f"Concurrent bulk insert completed. Success: {successful_inserts}, Failed: {failed_inserts}"
            )
            return failed_inserts == 0

        except Exception as e:
            logger.error(f"Unexpected error during concurrent bulk insert: {e}")
            return False

    async def get_table_stats(self) -> Dict[str, Any]:
        """Get statistics about the embeddings table."""
        try:
            async with self.get_db_connection() as conn:
                query = """
                    SELECT 
                        COUNT(*) as total_records,
                        COUNT(DISTINCT source_file) as unique_sources,
                        MIN(created_at) as earliest_record,
                        MAX(created_at) as latest_record
                    FROM document_embeddings
                """

                result = await conn.fetchrow(query)
                if result:
                    stats = {
                        "total_records": result["total_records"],
                        "unique_sources": result["unique_sources"],
                        "earliest_record": result["earliest_record"],
                        "latest_record": result["latest_record"],
                    }
                    logger.info(f"Table stats: {stats}")
                    return stats

        except Exception as e:
            logger.error(f"Error getting table stats: {e}")
            return {}

    async def search_similar_embeddings(
        self, query_embedding: List[float], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings using cosine similarity.

        Args:
            query_embedding: The embedding vector to search for
            limit: Maximum number of results to return

        Returns:
            List of dictionaries containing similar embeddings
        """
        try:
            async with self.get_db_connection() as conn:
                query = """
                    SELECT 
                        id,
                        chunk,
                        metadata_text,
                        source_file,
                        1 - (embedding <=> $1::vector) as similarity_score
                    FROM document_embeddings
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                """

                query_vector = json.dumps(query_embedding)
                results = await conn.fetch(query, query_vector, limit)

                return [dict(result) for result in results]

        except Exception as e:
            logger.error(f"Error searching similar embeddings: {e}")
            return []


async def main():
    """Example usage of the AsyncEmbeddingsDBManager."""

    # Database configuration
    db_config = {
        "host": os.getenv("POSTGRES_HOST"),
        "port": os.getenv("POSTGRES_PORT"),
        "user": os.getenv("POSTGRES_USERNAME"),
        "password": os.getenv("POSTGRES_PASSWORD"),
        "database": os.getenv("POSTGRES_DBNAME"),
    }

    # Sample data (replace with your actual data)
    sample_data = [
        {
            "id": str(uuid.uuid4()),
            "chunk": "This is a sample text chunk for embedding.",
            "embedding": [0.1, 0.2, 0.3] * 512,  # Example 1536-dimensional vector
            "metadata_text": "Sample metadata for document processing",
            "source_file": "sample_document.pdf",
        },
        {
            "id": str(uuid.uuid4()),
            "chunk": "Another sample text chunk with different content.",
            "embedding": [0.4, 0.5, 0.6] * 512,  # Example 1536-dimensional vector
            "metadata_text": "Different metadata for another chunk",
            "source_file": "another_document.pdf",
        },
        # Add more records as needed
    ]

    # Initialize the database manager
    db_manager = AsyncEmbeddingsDBManager(db_config)

    try:
        # Create connection pool
        await db_manager.create_connection_pool()

        # Create table if it doesn't exist
        await db_manager.create_embeddings_table()

        # Perform bulk insert (choose one method)
        # Method 1: Sequential batching
        success = await db_manager.bulk_insert_embeddings(sample_data)

        # Method 2: Concurrent batching (uncomment to use)
        # success = await db_manager.bulk_insert_embeddings_concurrent(sample_data, max_concurrent_batches=3)

        if success:
            logger.info("All records inserted successfully")

            # Get table statistics
            stats = await db_manager.get_table_stats()
            print(f"Table statistics: {stats}")

            # Example similarity search
            if sample_data:
                similar_results = await db_manager.search_similar_embeddings(
                    sample_data[0]["embedding"], limit=5
                )
                print(f"Similar embeddings found: {len(similar_results)}")
        else:
            logger.error("Some records failed to insert")

    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        # Clean up connection pool
        await db_manager.close_connection_pool()


if __name__ == "__main__":
    asyncio.run(main())

```
step 5:
with simillar logic of step 4 perform bm25 search 
Note: below is the existing code for bm25 search
```
import asyncio
import pickle
import os
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import aiofiles
import numpy as np
    class BM25Searcher:
        def __init__(self, index_file_path: str = "bm25_index.pkl"):
            self.index_file_path = index_file_path
            self.bm25_index = None
            self.corpus = []
            self.is_initialized = False
        
        async def initialize(self):
            """Initialize the searcher by loading the BM25 index"""
            if self.is_initialized:
                return
            
            try:
                if not os.path.exists(self.index_file_path):
                    raise FileNotFoundError(f"Index file {self.index_file_path} not found")
                
                async with aiofiles.open(self.index_file_path, 'rb') as f:
                    content = await f.read()
                    data = pickle.loads(content)
                    self.bm25_index = data['bm25_index']
                    self.corpus = data['corpus']
                    self.is_initialized = True
                    
                    print(f"Initialized searcher with {len(self.corpus)} documents")
            except Exception as e:
                print(f"Error initializing searcher: {e}")
                raise
        
        async def preprocess_query(self, query: str) -> List[str]:
            """Preprocess the query string"""
            # Simple tokenization - match the preprocessing used in indexing
            tokens = query.lower().split()
            return tokens
        
        async def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
            """
            Perform BM25 search on the loaded index
            
            Args:
                query: Search query string
                top_k: Number of top results to return
                
            Returns:
                List of dictionaries with structure: {'rank': int, 'chunk': str, 'score': float}
            """
            if not self.is_initialized:
                await self.initialize()
            
            if not query.strip():
                return []
            
            # Preprocess query
            tokenized_query = await self.preprocess_query(query)
            
            # Get BM25 scores for all documents
            scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top-k results
            top_k = min(top_k, len(scores))
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            # Format results
            results = []
            for rank, idx in enumerate(top_indices, 1):
                result = {
                    'rank': rank,
                    'chunk': self.corpus[idx],
                    'score': float(scores[idx])
                }
                results.append(result)
            
            return results
        
        async def batch_search(self, queries: List[str], top_k: int = 10) -> List[List[Dict[str, Any]]]:
            """
            Perform batch search for multiple queries
            
            Args:
                queries: List of query strings
                top_k: Number of top results to return for each query
                
            Returns:
                List of result lists, one for each query
            """
            if not self.is_initialized:
                await self.initialize()
            
            tasks = [self.search(query, top_k) for query in queries]
            results = await asyncio.gather(*tasks)
            return results
        
        async def get_index_stats(self) -> Dict[str, Any]:
            """Get statistics about the loaded index"""
            if not self.is_initialized:
                await self.initialize()
            
            return {
                'total_documents': len(self.corpus),
                'average_document_length': np.mean([len(doc.split()) for doc in self.corpus]),
                'index_file_path': self.index_file_path
            }


    async def main():
        """Example usage of the BM25 searcher"""
        # Initialize searcher
        searcher = BM25Searcher("bm25_index.pkl")
        
        # Example queries
        test_queries = [
            "python programming",
            "Google",
            "Android"
        ]
        
        print("=== Single Query Search ===")
        query = "python programming language"
        results = await searcher.search(query, top_k=3)
        
        print(f"Query: '{query}'")
        print(f"Top {len(results)} results:")
        for result in results:
            print(f"  Rank {result['rank']}: {result['chunk'][:50]}... (Score: {result['score']:.4f})")
        
        print("\n=== Batch Search ===")
        batch_results = await searcher.batch_search(test_queries, top_k=2)
        
        for i, (query, results) in enumerate(zip(test_queries, batch_results)):
            print(f"Query {i+1}: '{query}'")
            for result in results:
                print(f"  Rank {result['rank']}: {result['chunk'][:50]}... (Score: {result['score']:.4f})")
            print()
        
        print("=== Index Statistics ===")
        stats = await searcher.get_index_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")


    if __name__ == "__main__":
        # Run the async main function
        asyncio.run(main())

```
step 6:
    club the results from semantic search and bm25 search by removing the duplicates.

step 7:
 take the results from step 6 and call the reranker function to get the top 20 chuncks
 please find the existing code for reranker

 ```
 """
Voyage AI Reranking Client
==========================

This module provides an asynchronous client for document reranking using Voyage AI's reranking models.
Supports both single query and batch processing with proper error handling and resource management.

Author: Assistant
Date: 2025
"""

import os
import asyncio
from typing import List, Union, Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import voyageai

# Load environment variables
load_dotenv(override=True)


class RerankResult(BaseModel):
    """Container for reranking results."""
    document: str = Field(..., description="The document text")
    relevance_score: float = Field(..., description="Relevance score between 0 and 1")
    index: int = Field(..., description="Original index of the document")
    
    class Config:
        """Pydantic configuration."""
        frozen = True  # Make immutable like dataclass


class RerankResponse(BaseModel):
    """Container for reranking response."""
    results: List[RerankResult] = Field(..., description="List of reranked results")
    model: str = Field(..., description="Model used for reranking")
    usage: Optional[Dict[str, Any]] = Field(None, description="Usage information from API")
    
    class Config:
        """Pydantic configuration."""
        frozen = True  # Make immutable like dataclass


class AsyncVoyageRerankingClient:
    """
    An asynchronous wrapper for Voyage AI's document reranking API.
    
    This client provides methods for reranking documents based on relevance to a query.
    Supports both single query and batch processing with automatic chunking for large batches.
    """
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "rerank-2",
                 top_k: Optional[int] = None,
                 truncation: bool = True):
        """
        Initialize the Voyage AI reranking client.
        
        Args:
            api_key (str, optional): Voyage AI API key. If not provided, loads from VOYAGE_API_KEY env var.
            model (str): Reranking model name. Options include:
                - rerank-2 (latest reranking model)
                - rerank-1 (previous version)
            top_k (int, optional): Number of top results to return. If None, returns all results.
            truncation (bool): Whether to truncate input text if too long. Default is True.
        
        Raises:
            ValueError: If API key is not provided or found in environment.
        """
        self.api_key = api_key or os.getenv('VOYAGE_API_KEY')
        if not self.api_key:
            raise ValueError("Voyage AI API key is not set. Please set VOYAGE_API_KEY in environment or pass as argument.")
        
        self.client = voyageai.Client(api_key=self.api_key)
        self.model = model
        self.top_k = top_k
        self.truncation = truncation
        
        # Model specifications
        self.model_specs = {
            "rerank-2": {"max_documents": 1000, "context_length": 4000},
            "rerank-1": {"max_documents": 1000, "context_length": 4000},
        }
    
    async def rerank(self, 
                     query: str, 
                     documents: List[str],
                     top_k: Optional[int] = None,
                     model: Optional[str] = None) -> RerankResponse:
        """
        Rerank documents based on relevance to a query.
        
        Args:
            query (str): The search query.
            documents (List[str]): List of documents to rerank.
            top_k (int, optional): Number of top results to return. Overrides instance default.
            model (str, optional): Model to use for this request. Overrides instance default.
            
        Returns:
            RerankResponse: Reranking results with relevance scores.
            
        Raises:
            Exception: If the API call fails.
        """
        if not documents:
            return RerankResponse(results=[], model=model or self.model)
        
        try:
            # Prepare request parameters
            params = {
                "model": model or self.model,
                "truncation": self.truncation
            }
            
            # Add top_k if specified
            current_top_k = top_k or self.top_k
            if current_top_k is not None:
                params["top_k"] = current_top_k
            
            # Execute in thread pool since voyageai is synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.client.rerank(query, documents, **params)
            )
            
            # Convert response to our format
            results = []
            for result in response.results:
                results.append(RerankResult(
                    document=result.document,
                    relevance_score=result.relevance_score,
                    index=getattr(result, 'index', 0)  # Some responses might not have index
                ))
            
            return RerankResponse(
                results=results,
                model=self.model,
                usage=getattr(response, 'usage', None)
            )
            
        except Exception as e:
            raise Exception(f"Failed to rerank documents: {str(e)}")
    
    async def rerank_batch(self, 
                          queries: List[str], 
                          documents: List[str],
                          top_k: Optional[int] = None,
                          model: Optional[str] = None) -> List[RerankResponse]:
        """
        Rerank documents for multiple queries.
        
        Args:
            queries (List[str]): List of search queries.
            documents (List[str]): List of documents to rerank for each query.
            top_k (int, optional): Number of top results to return for each query.
            model (str, optional): Model to use for this request.
            
        Returns:
            List[RerankResponse]: List of reranking results for each query.
            
        Raises:
            Exception: If any API call fails.
        """
        if not queries:
            return []
        
        if not documents:
            return [RerankResponse(results=[], model=model or self.model) for _ in queries]
        
        return [RerankResponse(results=[], model=model or self.model) for _ in queries] if not documents else []
        
        results = []
        
        try:
            # Process each query sequentially to avoid rate limits
            for query in queries:
                result = await self.rerank(query, documents, top_k, model)
                results.append(result)
            
            return results
            
        except Exception as e:
            raise Exception(f"Failed to rerank batch: {str(e)}")
    
    async def rerank_concurrent(self, 
                               queries: List[str], 
                               documents: List[str],
                               max_concurrent: int = 3,
                               top_k: Optional[int] = None,
                               model: Optional[str] = None) -> List[RerankResponse]:
        """
        Rerank documents for multiple queries concurrently.
        
        Args:
            queries (List[str]): List of search queries.
            documents (List[str]): List of documents to rerank for each query.
            max_concurrent (int): Maximum number of concurrent requests. Default is 3.
            top_k (int, optional): Number of top results to return for each query.
            model (str, optional): Model to use for this request.
            
        Returns:
            List[RerankResponse]: List of reranking results in the same order as input queries.
            
        Raises:
            Exception: If any API call fails.
        """
        if not queries:
            return []
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def rerank_single_query(query: str) -> RerankResponse:
            async with semaphore:
                return await self.rerank(query, documents, top_k, model)
        
        # Execute all requests concurrently
        tasks = [rerank_single_query(query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        return results
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model.
        
        Returns:
            dict: Model specifications including max documents and context length.
        """
        return self.model_specs.get(self.model, {"max_documents": "unknown", "context_length": "unknown"})
    
    async def close(self):
        """Close the client session (placeholder for consistency)."""
        # Voyage AI client doesn't require explicit closing
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Example usage and testing
async def main():
    """Example usage of the Voyage AI reranking client."""
    
    async with AsyncVoyageRerankingClient() as client:
        print("=== Single Query Reranking ===")
        
        query = "When is Apple's conference call scheduled?"
        documents = [
            "The Mediterranean diet emphasizes fish, olive oil, and vegetables, believed to reduce chronic diseases.",
            "Photosynthesis in plants converts light energy into glucose and produces essential oxygen.",
            "20th-century innovations, from radios to smartphones, centered on electronic advancements.",
            "Rivers provide water, irrigation, and habitat for aquatic species, vital for ecosystems.",
            "Apple's conference call to discuss fourth fiscal quarter results and business updates is scheduled for Thursday, November 2, 2023 at 2:00 p.m. PT / 5:00 p.m. ET.",
            "Shakespeare's works, like 'Hamlet' and 'A Midsummer Night's Dream,' endure in literature."
        ]
        
        # Rerank documents
        reranking_response = await client.rerank(query, documents, top_k=3)
        
        print(f"Query: {query}")
        print(f"Model: {reranking_response.model}")
        print(f"Top {len(reranking_response.results)} results:")
        print("-" * 80)
        
        for i, result in enumerate(reranking_response.results, 1):
            print(f"Rank {i}:")
            print(f"Document: {result.document}")
            print(f"Relevance Score: {result.relevance_score:.4f}")
            print()


# Test with multiple queries
async def test_multiple_queries():
    """Test reranking with multiple queries."""
    
    queries = [
        "When is Apple's conference call scheduled?",
        "What are the benefits of the Mediterranean diet?",
        "How does photosynthesis work in plants?"
    ]
    
    documents = [
        "The Mediterranean diet emphasizes fish, olive oil, and vegetables, believed to reduce chronic diseases.",
        "Photosynthesis in plants converts light energy into glucose and produces essential oxygen.",
        "20th-century innovations, from radios to smartphones, centered on electronic advancements.",
        "Rivers provide water, irrigation, and habitat for aquatic species, vital for ecosystems.",
        "Apple's conference call to discuss fourth fiscal quarter results and business updates is scheduled for Thursday, November 2, 2023 at 2:00 p.m. PT / 5:00 p.m. ET.",
        "Shakespeare's works, like 'Hamlet' and 'A Midsummer Night's Dream,' endure in literature."
    ]
    
    async with AsyncVoyageRerankingClient() as client:
        print("=== Multiple Queries Reranking ===")
        
        # Process queries sequentially
        results = await client.rerank_batch(queries, documents, top_k=2)
        
        for i, (query, result) in enumerate(zip(queries, results), 1):
            print(f"Query {i}: {query}")
            print(f"Top {len(result.results)} results:")
            
            for j, doc_result in enumerate(result.results, 1):
                print(f"  {j}. Score: {doc_result.relevance_score:.4f}")
                print(f"     Document: {doc_result.document[:100]}...")
            print()


# Test concurrent processing
async def test_concurrent_reranking():
    """Test concurrent reranking with multiple queries."""
    
    queries = [
        "What are technological innovations?",
        "How do rivers support ecosystems?",
        "What is the Mediterranean diet?",
        "When is Apple's earnings call?"
    ]
    
    documents = [
        "The Mediterranean diet emphasizes fish, olive oil, and vegetables, believed to reduce chronic diseases.",
        "Photosynthesis in plants converts light energy into glucose and produces essential oxygen.",
        "20th-century innovations, from radios to smartphones, centered on electronic advancements.",
        "Rivers provide water, irrigation, and habitat for aquatic species, vital for ecosystems.",
        "Apple's conference call to discuss fourth fiscal quarter results and business updates is scheduled for Thursday, November 2, 2023 at 2:00 p.m. PT / 5:00 p.m. ET.",
        "Shakespeare's works, like 'Hamlet' and 'A Midsummer Night's Dream,' endure in literature."
    ]
    
    async with AsyncVoyageRerankingClient() as client:
        print("=== Concurrent Reranking Test ===")
        
        # Time the concurrent processing
        import time
        start_time = time.time()
        
        results = await client.rerank_concurrent(queries, documents, max_concurrent=2, top_k=2)
        
        end_time = time.time()
        
        print(f"Processed {len(results)} queries in {end_time - start_time:.2f} seconds")
        
        for i, (query, result) in enumerate(zip(queries, results), 1):
            print(f"\nQuery {i}: {query}")
            best_result = result.results[0] if result.results else None
            if best_result:
                print(f"Best match (score: {best_result.relevance_score:.4f}): {best_result.document[:80]}...")


# Test different models
async def test_different_models():
    """Test reranking with different models."""
    
    models = ["rerank-2", "rerank-1"]
    
    query = "What are the health benefits of diet?"
    documents = [
        "The Mediterranean diet emphasizes fish, olive oil, and vegetables, believed to reduce chronic diseases.",
        "Fast food consumption is linked to obesity and cardiovascular problems.",
        "Regular exercise and balanced nutrition are key to maintaining good health.",
        "Processed foods often contain high levels of sodium and preservatives."
    ]
    
    for model in models:
        try:
            async with AsyncVoyageRerankingClient(model=model) as client:
                result = await client.rerank(query, documents, top_k=2)
                
                print(f"=== Model: {model} ===")
                print(f"Query: {query}")
                print(f"Model info: {client.get_model_info()}")
                
                for i, doc_result in enumerate(result.results, 1):
                    print(f"Rank {i}: Score {doc_result.relevance_score:.4f}")
                    print(f"Document: {doc_result.document[:100]}...")
                print()
                
        except Exception as e:
            print(f"Error with model {model}: {str(e)}")


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())
    
    # Uncomment to test multiple queries
    # asyncio.run(test_multiple_queries())
    
    # Uncomment to test concurrent processing
    # asyncio.run(test_concurrent_reranking())
    
    # Uncomment to test different models
    # asyncio.run(test_different_models())
 ```
Note : while writing this program always remember to write more performance optimized code by using options like asynchronous batch job calling and make the code dynamic & also modulerize the code by spliting into multiple py files

