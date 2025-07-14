import asyncio
import pickle
import os
from typing import List
from rank_bm25 import BM25Okapi
import aiofiles


class BM25Indexer:
    def __init__(self, index_file_path: str = "bm25_index.pkl"):
        self.index_file_path = index_file_path
        self.bm25_index = None
        self.corpus = []

    async def load_existing_index(self) -> bool:
        """Load existing BM25 index if it exists"""
        try:
            if os.path.exists(self.index_file_path):
                async with aiofiles.open(self.index_file_path, "rb") as f:
                    content = await f.read()
                    data = pickle.loads(content)
                    self.bm25_index = data["bm25_index"]
                    self.corpus = data["corpus"]
                    print(f"Loaded existing index with {len(self.corpus)} documents")
                    return True
        except Exception as e:
            print(f"Error loading existing index: {e}")
        return False

    async def preprocess_chunks(self, chunks: List[str]) -> List[List[str]]:
        """Preprocess chunks by tokenizing them"""
        tokenized_chunks = []
        for chunk in chunks:
            # Simple tokenization - you can enhance this with more sophisticated methods
            tokens = chunk.lower().split()
            tokenized_chunks.append(tokens)
        return tokenized_chunks

    async def add_chunks_to_index(self, chunks: List[str]):
        """Add new chunks to the existing BM25 index"""
        if not chunks:
            print("No chunks provided")
            return

        print(f"Processing {len(chunks)} chunks...")

        # Preprocess new chunks
        tokenized_chunks = await self.preprocess_chunks(chunks)

        # Add to corpus
        self.corpus.extend(chunks)

        # Create tokenized corpus for BM25
        all_tokenized = []
        for chunk in self.corpus:
            tokens = chunk.lower().split()
            all_tokenized.append(tokens)

        # Rebuild BM25 index with all documents
        self.bm25_index = BM25Okapi(all_tokenized)

        print(f"Updated index now contains {len(self.corpus)} documents")

    async def save_index(self):
        """Save the BM25 index to pickle file"""
        try:
            data = {"bm25_index": self.bm25_index, "corpus": self.corpus}

            # Use async file writing
            async with aiofiles.open(self.index_file_path, "wb") as f:
                pickled_data = pickle.dumps(data)
                await f.write(pickled_data)

            print(f"Index saved to {self.index_file_path}")
        except Exception as e:
            print(f"Error saving index: {e}")
            raise

    async def process_chunks(self, chunks: List[str]):
        """Main method to process chunks and update index"""
        # Load existing index if available
        await self.load_existing_index()

        # Add new chunks to index
        await self.add_chunks_to_index(chunks)

        # Save updated index
        await self.save_index()


async def main():
    """Example usage of the BM25 indexer"""
    # Example chunks to index
    sample_chunks = [
        "Microsoft launched a new AI model to improve search accuracy.",
        "Alphabet is the parent company of Apple, Amazon, and several other tech ventures.",
        "Google Maps now offers real-time traffic updates and route optimization.",
        "The Pixel phone series is known for its camera and clean Android experience.",
        "Google Chrome dominates the web browser market with a large user base.",
    ]

    # Initialize indexer
    indexer = BM25Indexer("bm25_index.pkl")

    # Process chunks
    await indexer.process_chunks(sample_chunks)

    print("Indexing completed successfully!")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
