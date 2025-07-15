from utils.bm25_inference import BM25_indexer
import json

async def main():
    # Example chunks to index
    with open('all_folders_data.json', 'r', encoding='utf-8') as file:
        input_data = json.load(file)
        
    sample_chunks = [item['chunk'] for item in input_data]
    
    # Initialize indexer
    indexer = BM25_indexer.BM25Indexer("bm25_index.pkl")

    # Process chunks
    await indexer.process_chunks(sample_chunks)

    print("Indexing completed successfully!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())