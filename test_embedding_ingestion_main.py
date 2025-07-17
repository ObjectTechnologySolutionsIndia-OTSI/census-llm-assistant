import asyncio

# from utils.vector_indexing.indexing_process import main
from utils.vector_indexing.indexing_ingestion_batch_async import AsyncEmbeddingProcessor, logger
from utils.bm25_inference.BM25_indexer import BM25Indexer
import json


async def main():
    """Example usage of the AsyncEmbeddingProcessor."""
    
    # Create sample data (300 records to demonstrate batching)
    # input_data = []
    # base_texts = [
    #     "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
    #     "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
    #     "Natural language processing enables computers to understand, interpret, and generate human language.",
    #     "Computer vision allows machines to interpret and understand visual information from the world.",
    #     "Reinforcement learning is a type of machine learning where agents learn to make decisions through trial and error.",
    #     "Data science combines statistical analysis, machine learning, and domain expertise to extract insights from data.",
    #     "Artificial neural networks are computing systems inspired by biological neural networks.",
    #     "Big data refers to large, complex datasets that require special tools and techniques to process.",
    #     "Cloud computing provides on-demand access to computing resources over the internet.",
    #     "Blockchain is a distributed ledger technology that maintains a continuously growing list of records."
    # ]
    
    # # Generate 250 records for demonstration
    # for i in range(250):
    #     text_index = i % len(base_texts)
    #     input_data.append({
    #         "chunk": f"{base_texts[text_index]} (Record {i+1})",
    #         "metadata_text": f"Sample metadata for record {i+1}",
    #         "source_file": f"document_{(i//10)+1}.pdf"
    #     })
    
    
    embeddings_table_name = "census_data_embeddings_v1"
    bm25_index_path = f"./utils/bm25_inference/{embeddings_table_name}.pkl"

    with open('processed_chunks.jsonl', 'r', encoding='utf-8') as file:
        input_data = json.load(file)
    # input_data = []
    
    # Initialize the processor with batch size of 100
    processor = AsyncEmbeddingProcessor(
        voyage_model="voyage-3-large",
        voyage_input_type="document",
        batch_size=10,
        max_concurrent_operations=3,
        table_name=embeddings_table_name
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
    # ==============================================
    try:
        
        logger.info("BM25 Indexer Starting indexing process...")
        sample_chunks = [item['chunk'] for item in input_data]
    
        # Initialize indexer
        indexer = BM25Indexer(bm25_index_path)

        # Process chunks
        await indexer.process_chunks(sample_chunks)

        logger.info("Indexing completed successfully!")
    
    except Exception as e:
        logger.error(f"BM25 Indexer error: {e}")
        print(f"❌ BM25 Indexer error: {e}")
        



if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())