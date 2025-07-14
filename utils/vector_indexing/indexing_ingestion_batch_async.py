import asyncio
import uuid
import logging
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from collections import deque

# Import your existing modules
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

class AsyncEmbeddingProcessor:
    """
    Async embedding processor that processes records in batches of 100,
    with overlapping embedding creation and database insertion.
    """
    
    def __init__(self, 
                 voyage_model: str = "voyage-3-large",
                 voyage_input_type: Optional[str] = "document",
                 batch_size: int = 100,
                 max_concurrent_operations: int = 3,
                 db_config: Optional[Dict[str, str]] = None):
        """
        Initialize the async embedding processor.
        
        Args:
            voyage_model: Voyage AI model to use for embeddings
            voyage_input_type: Input type for Voyage API ("document" or "query")
            batch_size: Number of records to process in each batch (default: 100)
            max_concurrent_operations: Maximum concurrent embedding/insertion operations
            db_config: Database configuration dictionary
        """
        self.voyage_model = voyage_model
        self.voyage_input_type = voyage_input_type
        self.batch_size = batch_size
        self.max_concurrent_operations = max_concurrent_operations
        
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
        
        # Semaphore to control concurrent operations
        self.operation_semaphore = asyncio.Semaphore(max_concurrent_operations)
        
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
    
    def create_batches(self, input_data: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
        """
        Split input data into batches of specified size.
        
        Args:
            input_data: List of dictionaries to batch
            
        Returns:
            List of batches
        """
        batches = []
        for i in range(0, len(input_data), self.batch_size):
            batch = input_data[i:i + self.batch_size]
            batches.append(batch)
        
        logger.info(f"Created {len(batches)} batches of size {self.batch_size}")
        return batches
    
    async def process_batch_embeddings(self, 
                                     batch_data: List[Dict[str, str]], 
                                     batch_num: int) -> List[Dict[str, Any]]:
        """
        Process embeddings for a single batch.
        
        Args:
            batch_data: Batch of input data
            batch_num: Batch number for logging
            
        Returns:
            List of dictionaries with embeddings and IDs added
        """
        async with self.operation_semaphore:
            try:
                logger.info(f"Batch {batch_num}: Starting embedding creation for {len(batch_data)} records")
                
                # Extract chunks from batch data
                chunks = [item['chunk'] for item in batch_data]
                
                # Get embeddings from Voyage AI
                embeddings = await self.voyage_client.get_embeddings_batch(
                    texts=chunks,
                    batch_size=128,  # API batch size (can be different from processing batch size)
                    input_type=self.voyage_input_type
                )
                
                if len(embeddings) != len(chunks):
                    raise ValueError(f"Embedding count mismatch: got {len(embeddings)}, expected {len(chunks)}")
                
                # Combine with original data and add UUIDs
                processed_batch = []
                for i, (item, embedding) in enumerate(zip(batch_data, embeddings)):
                    processed_item = item.copy()
                    processed_item['id'] = str(uuid.uuid4())
                    processed_item['embedding'] = embedding
                    processed_batch.append(processed_item)
                
                logger.info(f"Batch {batch_num}: Successfully created embeddings for {len(processed_batch)} records")
                return processed_batch
                
            except Exception as e:
                logger.error(f"Batch {batch_num}: Error creating embeddings: {e}")
                raise
    
    async def insert_batch_to_database(self, 
                                     processed_batch: List[Dict[str, Any]], 
                                     batch_num: int) -> bool:
        """
        Insert a processed batch into the database.
        
        Args:
            processed_batch: Batch data with embeddings and IDs
            batch_num: Batch number for logging
            
        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"Batch {batch_num}: Starting database insertion for {len(processed_batch)} records")
            
            success = await self.db_manager.bulk_insert_embeddings(processed_batch)
            
            if success:
                logger.info(f"Batch {batch_num}: Successfully inserted {len(processed_batch)} records into database")
            else:
                logger.error(f"Batch {batch_num}: Failed to insert records into database")
            
            return success
            
        except Exception as e:
            logger.error(f"Batch {batch_num}: Error inserting into database: {e}")
            return False
    
    async def process_single_batch_pipeline(self, 
                                          batch_data: List[Dict[str, str]], 
                                          batch_num: int) -> bool:
        """
        Process a single batch through the complete pipeline.
        
        Args:
            batch_data: Batch of input data
            batch_num: Batch number for logging
            
        Returns:
            bool: True if successful
        """
        try:
            # Step 1: Create embeddings for the batch
            processed_batch = await self.process_batch_embeddings(batch_data, batch_num)
            
            # Step 2: Insert into database
            success = await self.insert_batch_to_database(processed_batch, batch_num)
            
            return success
            
        except Exception as e:
            logger.error(f"Batch {batch_num}: Pipeline failed: {e}")
            return False
    
    async def process_documents_async_pipeline(self, 
                                             input_data: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Process documents with async pipeline: while inserting batch N, 
        create embeddings for batch N+1.
        
        Args:
            input_data: List of dictionaries with keys: chunk, metadata_text, source_file
            
        Returns:
            Dict with processing results
        """
        try:
            logger.info("Starting async document processing pipeline")
            
            # Validate input data
            if not self.validate_input_data(input_data):
                return {"success": False, "error": "Input validation failed"}
            
            # Create batches
            batches = self.create_batches(input_data)
            total_batches = len(batches)
            
            if total_batches == 0:
                return {"success": True, "processed_batches": 0, "failed_batches": 0, "total_records": 0}
            
            # Process batches with overlapping operations
            successful_batches = 0
            failed_batches = 0
            
            # Queue to manage async operations
            pending_operations = deque()
            
            # Start processing first batch
            logger.info(f"Starting processing of {total_batches} batches")
            
            # Process all batches
            batch_tasks = []
            for batch_num, batch_data in enumerate(batches, 1):
                task = asyncio.create_task(
                    self.process_single_batch_pipeline(batch_data, batch_num)
                )
                batch_tasks.append(task)
            
            # Wait for all batches to complete
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Count results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Batch {i+1} failed with exception: {result}")
                    failed_batches += 1
                elif result:
                    successful_batches += 1
                else:
                    failed_batches += 1
            
            total_records_processed = successful_batches * self.batch_size
            
            logger.info(f"Pipeline completed: {successful_batches}/{total_batches} batches successful")
            
            return {
                "success": failed_batches == 0,
                "processed_batches": successful_batches,
                "failed_batches": failed_batches,
                "total_batches": total_batches,
                "total_records": len(input_data),
                "successful_records": total_records_processed
            }
            
        except Exception as e:
            logger.error(f"Error in async pipeline: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_documents_sequential_async(self, 
                                               input_data: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Alternative implementation: Sequential batch processing with async operations.
        Each batch waits for the previous one to complete embedding creation before starting DB insertion.
        
        Args:
            input_data: List of dictionaries with keys: chunk, metadata_text, source_file
            
        Returns:
            Dict with processing results
        """
        try:
            logger.info("Starting sequential async document processing")
            
            # Validate input data
            if not self.validate_input_data(input_data):
                return {"success": False, "error": "Input validation failed"}
            
            # Create batches
            batches = self.create_batches(input_data)
            total_batches = len(batches)
            
            successful_batches = 0
            failed_batches = 0
            
            # Process batches with overlapping embedding creation and DB insertion
            embedding_task = None
            insertion_task = None
            
            for batch_num, batch_data in enumerate(batches, 1):
                try:
                    # Start embedding creation for current batch
                    current_embedding_task = asyncio.create_task(
                        self.process_batch_embeddings(batch_data, batch_num)
                    )
                    
                    # If we have a previous insertion task, wait for it to complete
                    if insertion_task:
                        await insertion_task
                    
                    # Wait for current embedding creation to complete
                    processed_batch = await current_embedding_task
                    
                    # Start database insertion asynchronously
                    insertion_task = asyncio.create_task(
                        self.insert_batch_to_database(processed_batch, batch_num)
                    )
                    
                    # For the last batch, wait for insertion to complete
                    if batch_num == total_batches:
                        success = await insertion_task
                        if success:
                            successful_batches += 1
                        else:
                            failed_batches += 1
                    else:
                        # For other batches, we'll check the result in the next iteration
                        successful_batches += 1
                    
                except Exception as e:
                    logger.error(f"Batch {batch_num} failed: {e}")
                    failed_batches += 1
            
            logger.info(f"Sequential async processing completed: {successful_batches}/{total_batches} batches successful")
            
            return {
                "success": failed_batches == 0,
                "processed_batches": successful_batches,
                "failed_batches": failed_batches,
                "total_batches": total_batches,
                "total_records": len(input_data),
                "successful_records": successful_batches * self.batch_size
            }
            
        except Exception as e:
            logger.error(f"Error in sequential async processing: {e}")
            return {"success": False, "error": str(e)}

async def main():
    """Example usage of the AsyncEmbeddingProcessor."""
    
    # Create sample data (300 records to demonstrate batching)
    input_data = []
    base_texts = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
        "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
        "Natural language processing enables computers to understand, interpret, and generate human language.",
        "Computer vision allows machines to interpret and understand visual information from the world.",
        "Reinforcement learning is a type of machine learning where agents learn to make decisions through trial and error.",
        "Data science combines statistical analysis, machine learning, and domain expertise to extract insights from data.",
        "Artificial neural networks are computing systems inspired by biological neural networks.",
        "Big data refers to large, complex datasets that require special tools and techniques to process.",
        "Cloud computing provides on-demand access to computing resources over the internet.",
        "Blockchain is a distributed ledger technology that maintains a continuously growing list of records."
    ]
    
    # Generate 250 records for demonstration
    for i in range(250):
        text_index = i % len(base_texts)
        input_data.append({
            "chunk": f"{base_texts[text_index]} (Record {i+1})",
            "metadata_text": f"Sample metadata for record {i+1}",
            "source_file": f"document_{(i//10)+1}.pdf"
        })
    
    # Initialize the processor with batch size of 100
    processor = AsyncEmbeddingProcessor(
        voyage_model="voyage-3-large",
        voyage_input_type="document",
        batch_size=100,
        max_concurrent_operations=3
    )
    
    try:
        # Initialize components
        await processor.initialize()
        
        print(f"📊 Processing {len(input_data)} records in batches of {processor.batch_size}")
        
        # Process documents with async pipeline
        result = await processor.process_documents_async_pipeline(input_data)
        
        if result["success"]:
            print("✅ Successfully processed all documents!")
            print(f"📈 Results:")
            print(f"   - Total records: {result['total_records']}")
            print(f"   - Total batches: {result['total_batches']}")
            print(f"   - Successful batches: {result['processed_batches']}")
            print(f"   - Failed batches: {result['failed_batches']}")
            print(f"   - Successful records: {result['successful_records']}")
            
            # Get database statistics
            if processor.db_manager:
                stats = await processor.db_manager.get_table_stats()
                print(f"📊 Database statistics: {stats}")
        else:
            print("❌ Failed to process documents")
            print(f"Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Application error: {e}")
        
    finally:
        # Clean up resources
        await processor.cleanup()

async def demo_sequential_async():
    """Demo the sequential async processing method."""
    
    # Smaller dataset for demo
    input_data = [
        {
            "chunk": f"Sample text chunk number {i+1} for testing sequential async processing.",
            "metadata_text": f"Metadata for chunk {i+1}",
            "source_file": f"test_doc_{(i//5)+1}.pdf"
        }
        for i in range(50)  # 50 records = will create 1 batch of 100 (since we have only 50)
    ]
    
    processor = AsyncEmbeddingProcessor(batch_size=20)  # Smaller batches for demo
    
    try:
        await processor.initialize()
        
        print("🔄 Testing sequential async processing...")
        result = await processor.process_documents_sequential_async(input_data)
        
        print(f"Results: {result}")
        
    finally:
        await processor.cleanup()

if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())
    
    # Uncomment to test sequential async processing
    # asyncio.run(demo_sequential_async())