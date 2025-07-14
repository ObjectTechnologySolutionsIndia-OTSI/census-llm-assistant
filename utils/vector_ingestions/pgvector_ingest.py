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
