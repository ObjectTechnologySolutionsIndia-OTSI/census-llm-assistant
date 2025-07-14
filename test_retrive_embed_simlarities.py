from utils.emdeddings_inference.gemini_emded import AsyncGeminiEmbeddingsClient

async def main():
    async with AsyncGeminiEmbeddingsClient() as client:
        # Single text embedding
        texts = ["The quick brown fox jumps over the lazy dog.","The quick black fox jumps over the lazy cat." ]
        embeddings = await client.get_embeddings_batch(texts=texts, task_type="document")
        
        for text, embedding in zip(texts, embeddings):
            print(f"Text: {text}")
            print(f"Embedding dimensions: {len(embedding)}")
            print(embedding)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

