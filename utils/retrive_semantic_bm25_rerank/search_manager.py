"""
Search Manager Module
====================

This module manages both semantic search (using vector embeddings) and BM25 search,
providing unified interface for parallel search operations.

Author: Assistant
Date: 2025
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

# Import the existing modules (assuming they exist)
from utils.emdeddings_inference.voyage_emded import AsyncVoyageEmbeddingsClient
from utils.vector_ingestions.pgvector_ingest import AsyncEmbeddingsDBManager
from utils.bm25_inference.BM25_search import BM25Searcher

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Standardized search result structure."""
    id: str
    chunk: str
    score: float
    source: str  # 'semantic' or 'bm25'
    metadata: Dict[str, Any]
    source_file: Optional[str] = None


class SearchManager:
    """Manages both semantic and BM25 search operations."""
    
    def __init__(self, 
                 voyage_config: Dict[str, Any],
                 db_config: Dict[str, Any],
                 bm25_index_path: str,
                 table_name: str = None
                 ):
        """
        Initialize the search manager.
        
        Args:
            voyage_config: Configuration for Voyage AI services
            db_config: Database configuration for vector search
            bm25_index_path: Path to BM25 index file
        """
        self.voyage_config = voyage_config
        self.db_config = db_config
        self.bm25_index_path = bm25_index_path
        
        # Initialize clients
        self.voyage_client = AsyncVoyageEmbeddingsClient(
            api_key=voyage_config.get("api_key"),
            model=voyage_config.get("embedding_model", "voyage-3-large")
        )
        
        self.vector_db = AsyncEmbeddingsDBManager(db_config, table_name=table_name)
        self.bm25_searcher = BM25Searcher(bm25_index_path)
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize all search components."""
        if self.initialized:
            return
        
        try:
            # Initialize vector database connection pool
            await self.vector_db.create_connection_pool(min_size=5, max_size=15)
            
            # Initialize BM25 searcher
            await self.bm25_searcher.initialize()
            
            self.initialized = True
            logger.info("Search manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize search manager: {e}")
            raise
    
    async def semantic_search_single(self, 
                                   query: str, 
                                   top_k: int = 100) -> List[SearchResult]:
        """
        Perform semantic search for a single query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            # Generate embedding for the query
            query_embedding = await self.voyage_client.get_embedding(
                query, input_type="query"
            )
            
            # Search for similar embeddings
            similar_results = await self.vector_db.search_similar_embeddings(
                query_embedding, limit=top_k
            )
            
            # Convert to standardized format
            search_results = []
            for result in similar_results:
                search_result = SearchResult(
                    id=str(result.get("id", "")),
                    chunk=result.get("chunk", ""),
                    score=float(result.get("similarity_score", 0.0)),
                    source="semantic",
                    metadata={"metadata_text": result.get("metadata_text", "")},
                    source_file=result.get("source_file")
                )
                search_results.append(search_result)
            
            logger.info(f"Semantic search returned {len(search_results)} results for: {query[:50]}...")
            return search_results
            
        except Exception as e:
            logger.error(f"Semantic search failed for query '{query}': {e}")
            return []
    
    async def semantic_search_batch(self, 
                                  queries: List[str], 
                                  top_k: int = 100,
                                  max_concurrent: int = 5) -> List[SearchResult]:
        """
        Perform semantic search for multiple queries and combine results.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            max_concurrent: Maximum concurrent searches
            
        Returns:
            Combined and deduplicated list of search results
        """
        if not queries:
            return []
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def search_single(query: str) -> List[SearchResult]:
            async with semaphore:
                return await self.semantic_search_single(query, top_k)
        
        # Execute searches concurrently
        tasks = [search_single(query) for query in queries]
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine all results
        all_results = []
        for i, results in enumerate(results_lists):
            if isinstance(results, Exception):
                logger.error(f"Query {i} failed: {results}")
            else:
                all_results.extend(results)
        
        # Remove duplicates based on chunk content
        deduplicated = self._deduplicate_results(all_results)
        
        logger.info(f"Semantic batch search: {len(all_results)} total, {len(deduplicated)} after deduplication")
        return deduplicated
    
    async def bm25_search_single(self, 
                               query: str, 
                               top_k: int = 100) -> List[SearchResult]:
        """
        Perform BM25 search for a single query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            # Perform BM25 search
            bm25_results = await self.bm25_searcher.search(query, top_k)
            
            # Convert to standardized format
            search_results = []
            for result in bm25_results:
                search_result = SearchResult(
                    id=f"bm25_{result.get('rank', 0)}_{hash(result.get('chunk', ''))}",
                    chunk=result.get("chunk", ""),
                    score=float(result.get("score", 0.0)),
                    source="bm25",
                    metadata={"rank": result.get("rank", 0)},
                    source_file=None  # BM25 index may not have source file info
                )
                search_results.append(search_result)
            
            logger.info(f"BM25 search returned {len(search_results)} results for: {query[:50]}...")
            return search_results
            
        except Exception as e:
            logger.error(f"BM25 search failed for query '{query}': {e}")
            return []
    
    async def bm25_search_batch(self, 
                              queries: List[str], 
                              top_k: int = 100) -> List[SearchResult]:
        """
        Perform BM25 search for multiple queries and combine results.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            
        Returns:
            Combined and deduplicated list of search results
        """
        if not queries:
            return []
        
        try:
            # Use batch search if available, otherwise fall back to individual searches
            if hasattr(self.bm25_searcher, 'batch_search'):
                batch_results = await self.bm25_searcher.batch_search(queries, top_k)
                
                # Convert batch results to standardized format
                all_results = []
                for query_results in batch_results:
                    for result in query_results:
                        search_result = SearchResult(
                            id=f"bm25_{result.get('rank', 0)}_{hash(result.get('chunk', ''))}",
                            chunk=result.get("chunk", ""),
                            score=float(result.get("score", 0.0)),
                            source="bm25",
                            metadata={"rank": result.get("rank", 0)},
                            source_file=None
                        )
                        all_results.append(search_result)
            else:
                # Fall back to individual searches
                tasks = [self.bm25_search_single(query, top_k) for query in queries]
                results_lists = await asyncio.gather(*tasks, return_exceptions=True)
                
                all_results = []
                for i, results in enumerate(results_lists):
                    if isinstance(results, Exception):
                        logger.error(f"BM25 query {i} failed: {results}")
                    else:
                        all_results.extend(results)
            
            # Remove duplicates
            deduplicated = self._deduplicate_results(all_results)
            
            logger.info(f"BM25 batch search: {len(all_results)} total, {len(deduplicated)} after deduplication")
            return deduplicated
            
        except Exception as e:
            logger.error(f"BM25 batch search failed: {e}")
            return []
    
    async def parallel_search(self, 
                            queries: List[str], 
                            top_k_semantic: int = 100,
                            top_k_bm25: int = 100) -> tuple[List[SearchResult], List[SearchResult]]:
        """
        Perform semantic and BM25 searches in parallel.
        
        Args:
            queries: List of search queries
            top_k_semantic: Number of semantic search results
            top_k_bm25: Number of BM25 search results
            
        Returns:
            Tuple of (semantic_results, bm25_results)
        """
        # Execute both searches concurrently
        semantic_task = self.semantic_search_batch(queries, top_k_semantic)
        bm25_task = self.bm25_search_batch(queries, top_k_bm25)
        
        semantic_results, bm25_results = await asyncio.gather(
            semantic_task, bm25_task, return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(semantic_results, Exception):
            logger.error(f"Semantic search failed: {semantic_results}")
            semantic_results = []
        
        if isinstance(bm25_results, Exception):
            logger.error(f"BM25 search failed: {bm25_results}")
            bm25_results = []
        
        return semantic_results, bm25_results
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove duplicate results based on chunk content.
        
        Args:
            results: List of search results
            
        Returns:
            Deduplicated list of results
        """
        seen_chunks = set()
        deduplicated = []
        
        # Sort by score (descending) to keep highest scoring duplicates
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        for result in sorted_results:
            # Use a normalized version of the chunk for comparison
            chunk_key = result.chunk.strip().lower()
            
            if chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                deduplicated.append(result)
        
        return deduplicated
    
    def _normalize_scores(self, results: List[SearchResult], method: str = "minmax") -> List[SearchResult]:
        """
        Normalize scores across different search methods.
        
        Args:
            results: List of search results
            method: Normalization method ('minmax', 'zscore', 'sigmoid')
            
        Returns:
            Results with normalized scores
        """
        if not results:
            return results
        
        scores = [r.score for r in results]
        
        if method == "minmax":
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
            
            if score_range == 0:
                normalized_scores = [1.0] * len(scores)
            else:
                normalized_scores = [(s - min_score) / score_range for s in scores]
        
        elif method == "zscore":
            import statistics
            mean_score = statistics.mean(scores)
            std_score = statistics.stdev(scores) if len(scores) > 1 else 1.0
            
            if std_score == 0:
                normalized_scores = [0.0] * len(scores)
            else:
                normalized_scores = [(s - mean_score) / std_score for s in scores]
        
        elif method == "sigmoid":
            import math
            normalized_scores = [1 / (1 + math.exp(-s)) for s in scores]
        
        else:
            # Default to original scores
            normalized_scores = scores
        
        # Update results with normalized scores
        normalized_results = []
        for result, norm_score in zip(results, normalized_scores):
            normalized_result = SearchResult(
                id=result.id,
                chunk=result.chunk,
                score=norm_score,
                source=result.source,
                metadata=result.metadata,
                source_file=result.source_file
            )
            normalized_results.append(normalized_result)
        
        return normalized_results
    
    async def get_search_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the search indices.
        
        Returns:
            Dictionary with search statistics
        """
        stats = {}
        
        try:
            # Vector database stats
            vector_stats = await self.vector_db.get_table_stats()
            stats["vector_db"] = vector_stats
        except Exception as e:
            logger.error(f"Failed to get vector DB stats: {e}")
            stats["vector_db"] = {"error": str(e)}
        
        try:
            # BM25 index stats
            bm25_stats = await self.bm25_searcher.get_index_stats()
            stats["bm25"] = bm25_stats
        except Exception as e:
            logger.error(f"Failed to get BM25 stats: {e}")
            stats["bm25"] = {"error": str(e)}
        
        return stats
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Perform health check on all search components.
        
        Returns:
            Dictionary with health status of each component
        """
        health = {}
        
        # Test vector search
        try:
            test_embedding = [0.1] * 1024  # Assuming 1024-dimensional embeddings
            await self.vector_db.search_similar_embeddings(test_embedding, limit=1)
            health["vector_search"] = True
        except Exception as e:
            logger.error(f"Vector search health check failed: {e}")
            health["vector_search"] = False
        
        # Test BM25 search
        try:
            await self.bm25_searcher.search("test query", top_k=1)
            health["bm25_search"] = True
        except Exception as e:
            logger.error(f"BM25 search health check failed: {e}")
            health["bm25_search"] = False
        
        # Test Voyage embeddings
        try:
            await self.voyage_client.get_embedding("test", input_type="query")
            health["voyage_embeddings"] = True
        except Exception as e:
            logger.error(f"Voyage embeddings health check failed: {e}")
            health["voyage_embeddings"] = False
        
        return health
    
    async def close(self):
        """Clean up resources."""
        try:
            await self.vector_db.close_connection_pool()
            await self.voyage_client.close()
            logger.info("Search manager closed successfully")
        except Exception as e:
            logger.error(f"Error closing search manager: {e}")


# Performance monitoring and caching
class CachedSearchManager(SearchManager):
    """Extended search manager with caching capabilities."""
    
    def __init__(self, *args, cache_ttl: int = 3600, max_cache_size: int = 1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self.semantic_cache = {}
        self.bm25_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
    
    def _get_cache_key(self, query: str, top_k: int) -> str:
        """Generate cache key for query and parameters."""
        return f"{hash(query)}_{top_k}"
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - timestamp < self.cache_ttl
    
    async def semantic_search_single(self, query: str, top_k: int = 100) -> List[SearchResult]:
        """Cached version of semantic search."""
        cache_key = self._get_cache_key(query, top_k)
        
        # Check cache
        if cache_key in self.semantic_cache:
            cached_data, timestamp = self.semantic_cache[cache_key]
            if self._is_cache_valid(timestamp):
                self.cache_stats["hits"] += 1
                logger.debug(f"Cache hit for semantic query: {query[:30]}...")
                return cached_data
            else:
                # Remove expired entry
                del self.semantic_cache[cache_key]
        
        # Cache miss - perform search
        self.cache_stats["misses"] += 1
        results = await super().semantic_search_single(query, top_k)
        
        # Cache results
        if len(self.semantic_cache) < self.max_cache_size:
            self.semantic_cache[cache_key] = (results, time.time())
        
        return results
    
    async def bm25_search_single(self, query: str, top_k: int = 100) -> List[SearchResult]:
        """Cached version of BM25 search."""
        cache_key = self._get_cache_key(query, top_k)
        
        # Check cache
        if cache_key in self.bm25_cache:
            cached_data, timestamp = self.bm25_cache[cache_key]
            if self._is_cache_valid(timestamp):
                self.cache_stats["hits"] += 1
                logger.debug(f"Cache hit for BM25 query: {query[:30]}...")
                return cached_data
            else:
                # Remove expired entry
                del self.bm25_cache[cache_key]
        
        # Cache miss - perform search
        self.cache_stats["misses"] += 1
        results = await super().bm25_search_single(query, top_k)
        
        # Cache results
        if len(self.bm25_cache) < self.max_cache_size:
            self.bm25_cache[cache_key] = (results, time.time())
        
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_stats["hits"],
            "cache_misses": self.cache_stats["misses"],
            "hit_rate": hit_rate,
            "semantic_cache_size": len(self.semantic_cache),
            "bm25_cache_size": len(self.bm25_cache)
        }
    
    def clear_cache(self):
        """Clear all cached results."""
        self.semantic_cache.clear()
        self.bm25_cache.clear()
        self.cache_stats = {"hits": 0, "misses": 0}
        logger.info("Search caches cleared")


# Example usage and testing
async def main():
    """Example usage of the search manager."""
    
    voyage_config = {
        "api_key": "your_voyage_api_key",
        "embedding_model": "voyage-3-large"
    }
    
    db_config = {
        "host": "localhost",
        "port": "5432",
        "database": "your_db",
        "user": "your_user",
        "password": "your_password"
    }
    
    # Initialize search manager
    search_manager = SearchManager(
        voyage_config=voyage_config,
        db_config=db_config,
        bm25_index_path="bm25_index.pkl"
    )
    
    try:
        await search_manager.initialize()
        
        # Test single query searches
        query = "machine learning algorithms"
        
        print("=== Single Query Tests ===")
        semantic_results = await search_manager.semantic_search_single(query, top_k=5)
        print(f"Semantic results: {len(semantic_results)}")
        
        bm25_results = await search_manager.bm25_search_single(query, top_k=5)
        print(f"BM25 results: {len(bm25_results)}")
        
        # Test batch searches
        queries = [
            "artificial intelligence",
            "data science",
            "neural networks"
        ]
        
        print("\n=== Batch Search Tests ===")
        semantic_batch = await search_manager.semantic_search_batch(queries, top_k=10)
        print(f"Semantic batch results: {len(semantic_batch)}")
        
        bm25_batch = await search_manager.bm25_search_batch(queries, top_k=10)
        print(f"BM25 batch results: {len(bm25_batch)}")
        
        # Test parallel search
        print("\n=== Parallel Search Test ===")
        semantic_parallel, bm25_parallel = await search_manager.parallel_search(
            queries, top_k_semantic=10, top_k_bm25=10
        )
        print(f"Parallel semantic: {len(semantic_parallel)}")
        print(f"Parallel BM25: {len(bm25_parallel)}")
        
        # Test health check
        health = await search_manager.health_check()
        print(f"\nHealth check: {health}")
        
        # Test statistics
        stats = await search_manager.get_search_statistics()
        print(f"Search statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Search manager test failed: {e}")
    
    finally:
        await search_manager.close()


if __name__ == "__main__":
    asyncio.run(main())