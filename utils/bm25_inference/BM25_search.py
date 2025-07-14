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