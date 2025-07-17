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
    def __init__(self, db_config: Dict[str, str], table_name: str = "document_embeddings"):
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
        if table_name:
            self.table_name = table_name
        else:
            self.table_name = "document_embeddings"

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
        try:
            async with self.get_db_connection() as conn:

                # 1. Create table with new structure
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id UUID PRIMARY KEY,
                        chunk TEXT NOT NULL,
                        embedding VECTOR(1024),
                        source_domain TEXT,
                        menu TEXT,
                        menu_item TEXT,
                        source_link TEXT,
                        file_name TEXT,
                        title TEXT,
                        source_file TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                print("✅ Table ensured with new structure")

                # 2. Create indexes
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_embeddings_source_file
                    ON {self.table_name}(source_file)
                """)
                print("✅ Index on source_file ensured")

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_embeddings_source_domain
                    ON {self.table_name}(source_domain)
                """)
                print("✅ Index on source_domain ensured")

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_embeddings_menu
                    ON {self.table_name}(menu)
                """)
                print("✅ Index on menu ensured")

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_embeddings_menu_item
                    ON {self.table_name}(menu_item)
                """)
                print("✅ Index on menu_item ensured")

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_embeddings_created_at
                    ON {self.table_name}(created_at)
                """)
                print("✅ Index on created_at ensured")

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_embeddings_title
                    ON {self.table_name}(title)
                """)
                print("✅ Index on title ensured")

                # 3. Create function
                await conn.execute("""
                    CREATE OR REPLACE FUNCTION update_updated_at_column()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.updated_at = CURRENT_TIMESTAMP;
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql
                """)
                print("✅ Trigger function ensured")

                # 4. Drop trigger (if exists)
                await conn.execute(f"""
                    DROP TRIGGER IF EXISTS update_embeddings_updated_at
                    ON {self.table_name}
                """)
                print("✅ Old trigger dropped (if existed)")

                # 5. Create trigger
                await conn.execute(f"""
                    CREATE TRIGGER update_embeddings_updated_at
                    BEFORE UPDATE ON {self.table_name}
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column()
                """)
                print("✅ Trigger created")

                logger.info("Successfully created or verified all embedding-related DB components")

        except Exception as e:
            logger.error(f"Error in DB setup: {e}")
            raise

    async def validate_input_data(self, data: List[Dict[str, Any]]) -> bool:
        """
        Validate the input data structure.

        Args:
            data: List of dictionaries with required fields

        Returns:
            bool: True if validation passes
        """
        required_fields = {"id", "chunk", "embedding", "source_file"}
        optional_fields = {"source_domain", "menu", "menu_item", "source_link", "file_name", "title"}

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

            # Set default values for optional fields if not provided
            for field in optional_fields:
                if field not in record:
                    record[field] = None

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
                                    json.dumps(record["embedding"]),  # Convert list to JSON string for asyncpg
                                    record.get("source_domain"),
                                    record.get("menu"),
                                    record.get("menu_item"),
                                    record.get("source_link"),
                                    record.get("file_name"),
                                    record.get("title"),
                                    record["source_file"],
                                )
                            )

                        # Execute batch insert using asyncpg's executemany
                        insert_query = f"""
                        INSERT INTO {self.table_name} (
                            id, chunk, embedding, source_domain, menu, menu_item, 
                            source_link, file_name, title, source_file
                        )
                        VALUES ($1, $2, $3::vector, $4, $5, $6, $7, $8, $9, $10)
                        ON CONFLICT (id) DO UPDATE SET
                            chunk = EXCLUDED.chunk,
                            embedding = EXCLUDED.embedding,
                            source_domain = EXCLUDED.source_domain,
                            menu = EXCLUDED.menu,
                            menu_item = EXCLUDED.menu_item,
                            source_link = EXCLUDED.source_link,
                            file_name = EXCLUDED.file_name,
                            title = EXCLUDED.title,
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
                                        record.get("source_domain"),
                                        record.get("menu"),
                                        record.get("menu_item"),
                                        record.get("source_link"),
                                        record.get("file_name"),
                                        record.get("title"),
                                        record["source_file"],
                                    )
                                )

                            # Execute batch insert
                            insert_query = f"""
                            INSERT INTO {self.table_name} (
                                id, chunk, embedding, source_domain, menu, menu_item, 
                                source_link, file_name, title, source_file
                            )
                            VALUES ($1, $2, $3::vector, $4, $5, $6, $7, $8, $9, $10)
                            ON CONFLICT (id) DO UPDATE SET
                                chunk = EXCLUDED.chunk,
                                embedding = EXCLUDED.embedding,
                                source_domain = EXCLUDED.source_domain,
                                menu = EXCLUDED.menu,
                                menu_item = EXCLUDED.menu_item,
                                source_link = EXCLUDED.source_link,
                                file_name = EXCLUDED.file_name,
                                title = EXCLUDED.title,
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
                query = f"""
                    SELECT 
                        COUNT(*) as total_records,
                        COUNT(DISTINCT source_file) as unique_source_files,
                        COUNT(DISTINCT source_domain) as unique_source_domains,
                        COUNT(DISTINCT menu) as unique_menus,
                        COUNT(DISTINCT menu_item) as unique_menu_items,
                        MIN(created_at) as earliest_record,
                        MAX(created_at) as latest_record
                    FROM {self.table_name}
                """

                result = await conn.fetchrow(query)
                if result:
                    stats = {
                        "total_records": result["total_records"],
                        "unique_source_files": result["unique_source_files"],
                        "unique_source_domains": result["unique_source_domains"],
                        "unique_menus": result["unique_menus"],
                        "unique_menu_items": result["unique_menu_items"],
                        "earliest_record": result["earliest_record"],
                        "latest_record": result["latest_record"],
                    }
                    logger.info(f"Table stats: {stats}")
                    return stats

        except Exception as e:
            logger.error(f"Error getting table stats: {e}")
            return {}

    async def search_similar_embeddings(
        self, query_embedding: List[float], limit: int = 10, filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings using cosine similarity with optional filters.

        Args:
            query_embedding: The embedding vector to search for
            limit: Maximum number of results to return
            filters: Optional filters (e.g., {'source_domain': 'example.com', 'menu': 'products'})

        Returns:
            List of dictionaries containing similar embeddings
        """
        try:
            async with self.get_db_connection() as conn:
                base_query = f"""
                    SELECT 
                        id,
                        chunk,
                        source_domain,
                        menu,
                        menu_item,
                        source_link,
                        file_name,
                        title,
                        source_file,
                        1 - (embedding <=> $1::vector) as similarity_score
                    FROM {self.table_name}
                """
                
                where_conditions = []
                params = [json.dumps(query_embedding)]
                param_counter = 2

                if filters:
                    for key, value in filters.items():
                        if value is not None:
                            where_conditions.append(f"{key} = ${param_counter}")
                            params.append(value)
                            param_counter += 1

                if where_conditions:
                    query = f"{base_query} WHERE {' AND '.join(where_conditions)} ORDER BY embedding <=> $1::vector LIMIT ${param_counter}"
                else:
                    query = f"{base_query} ORDER BY embedding <=> $1::vector LIMIT ${param_counter}"
                
                params.append(limit)
                
                results = await conn.fetch(query, *params)
                return [dict(result) for result in results]

        except Exception as e:
            logger.error(f"Error searching similar embeddings: {e}")
            return []

    async def get_records_by_filters(self, filters: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get records by applying filters on the new columns.

        Args:
            filters: Dictionary of filters (e.g., {'source_domain': 'example.com', 'menu': 'products'})
            limit: Maximum number of results to return

        Returns:
            List of dictionaries containing filtered records
        """
        try:
            async with self.get_db_connection() as conn:
                base_query = f"""
                    SELECT 
                        id, chunk, source_domain, menu, menu_item, 
                        source_link, file_name, title, source_file,
                        created_at, updated_at
                    FROM {self.table_name}
                """
                
                where_conditions = []
                params = []
                param_counter = 1

                for key, value in filters.items():
                    if value is not None:
                        where_conditions.append(f"{key} = ${param_counter}")
                        params.append(value)
                        param_counter += 1

                if where_conditions:
                    query = f"{base_query} WHERE {' AND '.join(where_conditions)} ORDER BY created_at DESC LIMIT ${param_counter}"
                else:
                    query = f"{base_query} ORDER BY created_at DESC LIMIT ${param_counter}"
                
                params.append(limit)
                
                results = await conn.fetch(query, *params)
                return [dict(result) for result in results]

        except Exception as e:
            logger.error(f"Error getting records by filters: {e}")
            return []


async def main():
    """Example usage of the AsyncEmbeddingsDBManager with new table structure."""

    # Database configuration
    db_config = {
        "host": os.getenv("POSTGRES_HOST"),
        "port": os.getenv("POSTGRES_PORT"),
        "user": os.getenv("POSTGRES_USERNAME"),
        "password": os.getenv("POSTGRES_PASSWORD"),
        "database": os.getenv("POSTGRES_DBNAME"),
    }

    # Sample data with new structure
    sample_data = [
        {
            "id": str(uuid.uuid4()),
            "chunk": "This is a sample text chunk for embedding from the products page.",
            "embedding": [0.1, 0.2, 0.3] * 341 + [0.1, 0.2, 0.3][:1024-(341*3)],  # Ensure exactly 1024 dimensions
            "source_domain": "example.com",
            "menu": "products",
            "menu_item": "laptops",
            "source_link": "https://example.com/products/laptops",
            "file_name": "products_laptops.html",
            "title": "Latest Laptops Collection",
            "source_file": "sample_document.pdf",
        },
        {
            "id": str(uuid.uuid4()),
            "chunk": "Another sample text chunk with different content from services.",
            "embedding": [0.4, 0.5, 0.6] * 341 + [0.4, 0.5, 0.6][:1024-(341*3)],  # Ensure exactly 1024 dimensions
            "source_domain": "example.com",
            "menu": "services",
            "menu_item": "support",
            "source_link": "https://example.com/services/support",
            "file_name": "services_support.html",
            "title": "Customer Support Services",
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

            # # Example similarity search with filters
            # if sample_data:
            #     # Search without filters
            #     similar_results = await db_manager.search_similar_embeddings(
            #         sample_data[0]["embedding"], limit=5
            #     )
            #     print(f"Similar embeddings found: {len(similar_results)}")

            #     # Search with filters
            #     filtered_results = await db_manager.search_similar_embeddings(
            #         sample_data[0]["embedding"], 
            #         limit=5,
            #         filters={"source_domain": "example.com", "menu": "products"}
            #     )
            #     print(f"Filtered similar embeddings found: {len(filtered_results)}")

            #     # Get records by filters
            #     records_by_filter = await db_manager.get_records_by_filters(
            #         {"menu": "products"}, limit=10
            #     )
            #     print(f"Records with menu='products': {len(records_by_filter)}")

        else:
            logger.error("Some records failed to insert")

    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        # Clean up connection pool
        await db_manager.close_connection_pool()


if __name__ == "__main__":
    asyncio.run(main())