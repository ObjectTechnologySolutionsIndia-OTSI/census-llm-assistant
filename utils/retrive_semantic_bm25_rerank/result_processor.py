"""
Result Processor Module
======================

This module handles combining, deduplicating, and reranking search results
from multiple sources (semantic and BM25 search).

Author: Assistant
Date: 2025
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import hashlib
import time

# Import the existing reranking module
from utils.re_ranking_inference.voyage_reranker import AsyncVoyageRerankingClient, RerankResponse

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Standardized search result structure."""
    id: str
    chunk: str
    score: float
    source: str  # 'semantic', 'bm25', or 'combined'
    metadata: Dict[str, Any]
    source_file: Optional[str] = None


@dataclass
class ProcessingStats:
    """Statistics for result processing operations."""
    total_semantic_results: int
    total_bm25_results: int
    combined_results: int
    deduplicated_results: int
    final_reranked_results: int
    processing_time: float
    reranking_time: float


class ResultProcessor:
    """Handles combination, deduplication, and reranking of search results."""
    
    def __init__(self, voyage_config: Dict[str, Any]):
        """
        Initialize the result processor.
        
        Args:
            voyage_config: Configuration for Voyage AI reranking service
        """
        self.reranker = AsyncVoyageRerankingClient(
            api_key=voyage_config.get("api_key"),
            model=voyage_config.get("rerank_model", "rerank-2")
        )
        
    def _generate_content_hash(self, text: str) -> str:
        """Generate a hash for text content to identify duplicates."""
        # Normalize text for comparison
        normalized = text.strip().lower()
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text chunks."""
        # Simple Jaccard similarity for quick comparison
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    async def combine_and_deduplicate(self, 
                                    semantic_results: List[SearchResult],
                                    bm25_results: List[SearchResult],
                                    similarity_threshold: float = 0.85,
                                    score_normalization: str = "minmax") -> List[SearchResult]:
        """
        Combine semantic and BM25 results and remove duplicates.
        
        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search
            similarity_threshold: Threshold for considering results as duplicates
            score_normalization: Method for normalizing scores ('minmax', 'zscore', 'none')
            
        Returns:
            Combined and deduplicated results
        """
        start_time = time.time()
        
        # Normalize scores within each result set
        normalized_semantic = self._normalize_scores(semantic_results, score_normalization)
        normalized_bm25 = self._normalize_scores(bm25_results, score_normalization)
        
        # Combine all results
        all_results = normalized_semantic + normalized_bm25
        
        if not all_results:
            return []
        
        # Deduplicate based on content similarity
        deduplicated = await self._advanced_deduplication(
            all_results, similarity_threshold
        )
        
        # Sort by combined score (considering source diversity)
        final_results = self._apply_source_diversity_boost(deduplicated)
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Combined {len(semantic_results)} semantic + {len(bm25_results)} BM25 "
            f"= {len(all_results)} total -> {len(final_results)} after deduplication "
            f"in {processing_time:.2f}s"
        )
        
        return final_results
    
    async def _advanced_deduplication(self, 
                                    results: List[SearchResult],
                                    similarity_threshold: float) -> List[SearchResult]:
        """
        Advanced deduplication using content similarity.
        
        Args:
            results: List of search results
            similarity_threshold: Similarity threshold for duplicates
            
        Returns:
            Deduplicated results
        """
        if not results:
            return []
        
        # Sort by score (descending) to prioritize higher-scoring results
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        deduplicated = []
        seen_hashes = set()
        
        for result in sorted_results:
            # First check exact content hash
            content_hash = self._generate_content_hash(result.chunk)
            
            if content_hash in seen_hashes:
                continue
            
            # Check similarity with existing results
            is_duplicate = False
            for existing in deduplicated:
                similarity = self._calculate_similarity(result.chunk, existing.chunk)
                if similarity >= similarity_threshold:
                    # Found a similar result - merge metadata if beneficial
                    existing.metadata.update({
                        "alternative_sources": existing.metadata.get("alternative_sources", []) + [result.source],
                        "combined_score": max(existing.score, result.score)
                    })
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_hashes.add(content_hash)
                # Mark as combined source if it has high scores from both sources
                result.source = "combined"
                deduplicated.append(result)
        
        return deduplicated
    
    def _normalize_scores(self, 
                         results: List[SearchResult], 
                         method: str = "minmax") -> List[SearchResult]:
        """
        Normalize scores within a result set.
        
        Args:
            results: List of search results
            method: Normalization method
            
        Returns:
            Results with normalized scores
        """
        if not results or method == "none":
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
                # Apply sigmoid to z-scores to get 0-1 range
                import math
                z_scores = [(s - mean_score) / std_score for s in scores]
                normalized_scores = [1 / (1 + math.exp(-z)) for z in z_scores]
        
        else:
            # Default to original scores
            return results
        
        # Create new results with normalized scores
        normalized_results = []
        for result, norm_score in zip(results, normalized_scores):
            normalized_result = SearchResult(
                id=result.id,
                chunk=result.chunk,
                score=norm_score,
                source=result.source,
                metadata=result.metadata.copy(),
                source_file=result.source_file
            )
            # Store original score in metadata
            normalized_result.metadata["original_score"] = result.score
            normalized_results.append(normalized_result)
        
        return normalized_results
    
    def _apply_source_diversity_boost(self, 
                                    results: List[SearchResult],
                                    boost_factor: float = 0.1) -> List[SearchResult]:
        """
        Apply a small boost to results that appear in multiple sources.
        
        Args:
            results: List of search results
            boost_factor: Boost factor for multi-source results
            
        Returns:
            Results with diversity boost applied
        """
        boosted_results = []
        
        for result in results:
            boosted_score = result.score
            
            # Boost results that appear in multiple sources
            alt_sources = result.metadata.get("alternative_sources", [])
            if alt_sources:
                unique_sources = set([result.source] + alt_sources)
                if len(unique_sources) > 1:
                    boosted_score = min(1.0, result.score + boost_factor)
            
            boosted_result = SearchResult(
                id=result.id,
                chunk=result.chunk,
                score=boosted_score,
                source=result.source,
                metadata=result.metadata.copy(),
                source_file=result.source_file
            )
            boosted_result.metadata["diversity_boost"] = boosted_score - result.score
            boosted_results.append(boosted_result)
        
        # Sort by final boosted score
        return sorted(boosted_results, key=lambda x: x.score, reverse=True)
    
    async def rerank_results(self, 
                           query: str,
                           results: List[SearchResult],
                           top_k: int = 20) -> List[SearchResult]:
        """
        Rerank combined results using Voyage AI reranker.
        
        Args:
            query: Original user query
            results: Combined search results
            top_k: Number of top results to return
            
        Returns:
            Reranked results
        """
        if not results:
            return []
        
        start_time = time.time()
        
        try:
            # Prepare documents for reranking
            documents = [result.chunk for result in results]
            
            # Perform reranking
            rerank_response = await self.reranker.rerank(
                query=query,
                documents=documents,
                top_k=min(top_k, len(documents))
            )
            
            # Map reranked results back to original SearchResult objects
            reranked_results = []
            for rerank_result in rerank_response.results:
                # Find original result by index or content
                original_result = None
                for result in results:
                    if result.chunk == rerank_result.document:
                        original_result = result
                        break
                
                if original_result:
                    # Create new result with rerank score
                    reranked_result = SearchResult(
                        id=original_result.id,
                        chunk=original_result.chunk,
                        score=rerank_result.relevance_score,
                        source=original_result.source,
                        metadata=original_result.metadata.copy(),
                        source_file=original_result.source_file
                    )
                    # Store original scores in metadata
                    reranked_result.metadata.update({
                        "original_score": original_result.score,
                        "rerank_score": rerank_result.relevance_score,
                        "rerank_index": rerank_result.index
                    })
                    reranked_results.append(reranked_result)
            
            reranking_time = time.time() - start_time
            
            logger.info(
                f"Reranked {len(results)} results -> top {len(reranked_results)} "
                f"in {reranking_time:.2f}s"
            )
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fall back to original results, sorted by score
            return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
    
    async def batch_rerank_results(self, 
                                 queries_and_results: List[tuple[str, List[SearchResult]]],
                                 top_k: int = 20,
                                 max_concurrent: int = 3) -> List[List[SearchResult]]:
        """
        Rerank results for multiple queries concurrently.
        
        Args:
            queries_and_results: List of (query, results) tuples
            top_k: Number of top results per query
            max_concurrent: Maximum concurrent reranking operations
            
        Returns:
            List of reranked result lists
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def rerank_single(query_results: tuple[str, List[SearchResult]]) -> List[SearchResult]:
            async with semaphore:
                query, results = query_results
                return await self.rerank_results(query, results, top_k)
        
        tasks = [rerank_single(qr) for qr in queries_and_results]
        reranked_lists = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(reranked_lists):
            if isinstance(result, Exception):
                logger.error(f"Reranking failed for query {i}: {result}")
                # Fall back to original results for this query
                _, original_results = queries_and_results[i]
                fallback = sorted(original_results, key=lambda x: x.score, reverse=True)[:top_k]
                processed_results.append(fallback)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def analyze_result_distribution(self, results: List[SearchResult]) -> Dict[str, Any]:
        """
        Analyze the distribution of results by source and other metrics.
        
        Args:
            results: List of search results
            
        Returns:
            Analysis statistics
        """
        if not results:
            return {}
        
        # Count by source
        source_counts = {}
        score_by_source = {}
        
        for result in results:
            source = result.source
            source_counts[source] = source_counts.get(source, 0) + 1
            
            if source not in score_by_source:
                score_by_source[source] = []
            score_by_source[source].append(result.score)
        
        # Calculate statistics
        analysis = {
            "total_results": len(results),
            "source_distribution": source_counts,
            "average_scores_by_source": {},
            "score_statistics": {
                "min": min(r.score for r in results),
                "max": max(r.score for r in results),
                "avg": sum(r.score for r in results) / len(results)
            }
        }
        
        # Average scores by source
        for source, scores in score_by_source.items():
            analysis["average_scores_by_source"][source] = sum(scores) / len(scores)
        
        return analysis
    
    def filter_results_by_criteria(self, 
                                 results: List[SearchResult],
                                 min_score: float = 0.0,
                                 max_results: Optional[int] = None,
                                 required_sources: Optional[List[str]] = None,
                                 exclude_sources: Optional[List[str]] = None) -> List[SearchResult]:
        """
        Filter results based on various criteria.
        
        Args:
            results: List of search results
            min_score: Minimum score threshold
            max_results: Maximum number of results to return
            required_sources: Only include results from these sources
            exclude_sources: Exclude results from these sources
            
        Returns:
            Filtered results
        """
        filtered = results
        
        # Filter by score
        if min_score > 0:
            filtered = [r for r in filtered if r.score >= min_score]
        
        # Filter by required sources
        if required_sources:
            filtered = [r for r in filtered if r.source in required_sources]
        
        # Filter by excluded sources
        if exclude_sources:
            filtered = [r for r in filtered if r.source not in exclude_sources]
        
        # Sort by score and limit
        filtered = sorted(filtered, key=lambda x: x.score, reverse=True)
        
        if max_results:
            filtered = filtered[:max_results]
        
        return filtered
    
    async def generate_processing_stats(self, 
                                      semantic_results: List[SearchResult],
                                      bm25_results: List[SearchResult],
                                      combined_results: List[SearchResult],
                                      final_results: List[SearchResult],
                                      processing_time: float,
                                      reranking_time: float) -> ProcessingStats:
        """
        Generate comprehensive processing statistics.
        
        Args:
            semantic_results: Original semantic search results
            bm25_results: Original BM25 search results
            combined_results: Combined results before reranking
            final_results: Final reranked results
            processing_time: Time for combination and deduplication
            reranking_time: Time for reranking
            
        Returns:
            Processing statistics
        """
        return ProcessingStats(
            total_semantic_results=len(semantic_results),
            total_bm25_results=len(bm25_results),
            combined_results=len(combined_results),
            deduplicated_results=len(combined_results),  # After deduplication
            final_reranked_results=len(final_results),
            processing_time=processing_time,
            reranking_time=reranking_time
        )


# Advanced result processing strategies
class AdvancedResultProcessor(ResultProcessor):
    """Extended result processor with advanced strategies."""
    
    def __init__(self, voyage_config: Dict[str, Any], enable_ml_features: bool = False):
        super().__init__(voyage_config)
        self.enable_ml_features = enable_ml_features
        
        if enable_ml_features:
            try:
                # Optional: Import ML libraries for advanced features
                import numpy as np
                import scipy.spatial.distance as distance
                self.np = np
                self.distance = distance
            except ImportError:
                logger.warning("ML libraries not available, disabling ML features")
                self.enable_ml_features = False
    
    async def semantic_clustering_deduplication(self, 
                                              results: List[SearchResult],
                                              similarity_threshold: float = 0.8) -> List[SearchResult]:
        """
        Advanced deduplication using semantic clustering.
        
        Args:
            results: List of search results
            similarity_threshold: Similarity threshold for clustering
            
        Returns:
            Deduplicated results using clustering
        """
        if not self.enable_ml_features or not results:
            return await self._advanced_deduplication(results, similarity_threshold)
        
        try:
            # Generate embeddings for all chunks (simplified - in practice use actual embeddings)
            embeddings = []
            for result in results:
                # This is a placeholder - in practice, you'd use the actual embeddings
                # from the search results or generate them
                embedding = self._generate_simple_embedding(result.chunk)
                embeddings.append(embedding)
            
            # Perform clustering
            clusters = self._cluster_similar_results(embeddings, similarity_threshold)
            
            # Select best result from each cluster
            deduplicated = []
            for cluster in clusters:
                # Get results in this cluster
                cluster_results = [results[i] for i in cluster]
                # Select highest scoring result
                best_result = max(cluster_results, key=lambda x: x.score)
                
                # Merge metadata from cluster
                merged_metadata = best_result.metadata.copy()
                merged_metadata["cluster_size"] = len(cluster)
                merged_metadata["cluster_avg_score"] = sum(r.score for r in cluster_results) / len(cluster_results)
                
                deduplicated_result = SearchResult(
                    id=best_result.id,
                    chunk=best_result.chunk,
                    score=best_result.score,
                    source=best_result.source,
                    metadata=merged_metadata,
                    source_file=best_result.source_file
                )
                deduplicated.append(deduplicated_result)
            
            return sorted(deduplicated, key=lambda x: x.score, reverse=True)
            
        except Exception as e:
            logger.error(f"Semantic clustering failed, falling back to standard deduplication: {e}")
            return await self._advanced_deduplication(results, similarity_threshold)
    
    def _generate_simple_embedding(self, text: str) -> List[float]:
        """Generate a simple embedding for clustering (placeholder)."""
        # This is a very simplified embedding - in practice use proper embeddings
        words = text.lower().split()
        # Create a simple bag-of-words style vector
        vocab_size = 1000
        embedding = [0.0] * vocab_size
        
        for word in words:
            # Simple hash-based position
            pos = hash(word) % vocab_size
            embedding[pos] += 1.0
        
        # Normalize
        total = sum(embedding)
        if total > 0:
            embedding = [x / total for x in embedding]
        
        return embedding
    
    def _cluster_similar_results(self, embeddings: List[List[float]], threshold: float) -> List[List[int]]:
        """Cluster similar results based on embeddings."""
        if not self.enable_ml_features:
            # Fallback to simple grouping
            return [[i] for i in range(len(embeddings))]
        
        try:
            import numpy as np
            from scipy.spatial.distance import pdist, squareform
            from scipy.cluster.hierarchy import linkage, fcluster
            
            # Convert to numpy array
            X = np.array(embeddings)
            
            # Calculate pairwise distances
            distances = pdist(X, metric='cosine')
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(distances, method='average')
            
            # Get clusters
            cluster_labels = fcluster(linkage_matrix, 1 - threshold, criterion='distance')
            
            # Group indices by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(i)
            
            return list(clusters.values())
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            # Fallback to individual clusters
            return [[i] for i in range(len(embeddings))]
    
    async def adaptive_score_fusion(self, 
                                  semantic_results: List[SearchResult],
                                  bm25_results: List[SearchResult],
                                  fusion_strategy: str = "rrf") -> List[SearchResult]:
        """
        Advanced score fusion using multiple strategies.
        
        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search
            fusion_strategy: Fusion strategy ('rrf', 'weighted', 'comb_sum')
            
        Returns:
            Fused results
        """
        if fusion_strategy == "rrf":
            return await self._reciprocal_rank_fusion(semantic_results, bm25_results)
        elif fusion_strategy == "weighted":
            return await self._weighted_score_fusion(semantic_results, bm25_results)
        elif fusion_strategy == "comb_sum":
            return await self._combination_sum_fusion(semantic_results, bm25_results)
        else:
            # Default to simple combination
            return await self.combine_and_deduplicate(semantic_results, bm25_results)
    
    async def _reciprocal_rank_fusion(self, 
                                    semantic_results: List[SearchResult],
                                    bm25_results: List[SearchResult],
                                    k: int = 60) -> List[SearchResult]:
        """Reciprocal Rank Fusion (RRF) algorithm."""
        # Create rank maps
        semantic_ranks = {result.chunk: i + 1 for i, result in enumerate(semantic_results)}
        bm25_ranks = {result.chunk: i + 1 for i, result in enumerate(bm25_results)}
        
        # Get all unique chunks
        all_chunks = set(semantic_ranks.keys()) | set(bm25_ranks.keys())
        
        # Calculate RRF scores
        rrf_results = []
        for chunk in all_chunks:
            rrf_score = 0.0
            
            if chunk in semantic_ranks:
                rrf_score += 1.0 / (k + semantic_ranks[chunk])
            
            if chunk in bm25_ranks:
                rrf_score += 1.0 / (k + bm25_ranks[chunk])
            
            # Find original result to get metadata
            original_result = None
            for result in semantic_results + bm25_results:
                if result.chunk == chunk:
                    original_result = result
                    break
            
            if original_result:
                rrf_result = SearchResult(
                    id=original_result.id,
                    chunk=chunk,
                    score=rrf_score,
                    source="rrf_fusion",
                    metadata=original_result.metadata.copy(),
                    source_file=original_result.source_file
                )
                rrf_result.metadata["rrf_score"] = rrf_score
                rrf_results.append(rrf_result)
        
        return sorted(rrf_results, key=lambda x: x.score, reverse=True)
    
    async def _weighted_score_fusion(self, 
                                   semantic_results: List[SearchResult],
                                   bm25_results: List[SearchResult],
                                   semantic_weight: float = 0.7,
                                   bm25_weight: float = 0.3) -> List[SearchResult]:
        """Weighted score fusion."""
        # Normalize scores within each result set
        norm_semantic = self._normalize_scores(semantic_results, "minmax")
        norm_bm25 = self._normalize_scores(bm25_results, "minmax")
        
        # Create score maps
        semantic_scores = {result.chunk: result.score for result in norm_semantic}
        bm25_scores = {result.chunk: result.score for result in norm_bm25}
        
        # Get all unique chunks
        all_chunks = set(semantic_scores.keys()) | set(bm25_scores.keys())
        
        # Calculate weighted scores
        weighted_results = []
        for chunk in all_chunks:
            weighted_score = 0.0
            
            if chunk in semantic_scores:
                weighted_score += semantic_weight * semantic_scores[chunk]
            
            if chunk in bm25_scores:
                weighted_score += bm25_weight * bm25_scores[chunk]
            
            # Find original result
            original_result = None
            for result in norm_semantic + norm_bm25:
                if result.chunk == chunk:
                    original_result = result
                    break
            
            if original_result:
                weighted_result = SearchResult(
                    id=original_result.id,
                    chunk=chunk,
                    score=weighted_score,
                    source="weighted_fusion",
                    metadata=original_result.metadata.copy(),
                    source_file=original_result.source_file
                )
                weighted_result.metadata["weighted_score"] = weighted_score
                weighted_results.append(weighted_result)
        
        return sorted(weighted_results, key=lambda x: x.score, reverse=True)
    
    async def _combination_sum_fusion(self, 
                                    semantic_results: List[SearchResult],
                                    bm25_results: List[SearchResult]) -> List[SearchResult]:
        """Combination sum fusion (CombSUM)."""
        # Normalize scores
        norm_semantic = self._normalize_scores(semantic_results, "minmax")
        norm_bm25 = self._normalize_scores(bm25_results, "minmax")
        
        # Create score maps
        semantic_scores = {result.chunk: result.score for result in norm_semantic}
        bm25_scores = {result.chunk: result.score for result in norm_bm25}
        
        # Get all unique chunks
        all_chunks = set(semantic_scores.keys()) | set(bm25_scores.keys())
        
        # Calculate sum scores
        sum_results = []
        for chunk in all_chunks:
            sum_score = semantic_scores.get(chunk, 0.0) + bm25_scores.get(chunk, 0.0)
            
            # Find original result
            original_result = None
            for result in norm_semantic + norm_bm25:
                if result.chunk == chunk:
                    original_result = result
                    break
            
            if original_result:
                sum_result = SearchResult(
                    id=original_result.id,
                    chunk=chunk,
                    score=sum_score,
                    source="combsum_fusion",
                    metadata=original_result.metadata.copy(),
                    source_file=original_result.source_file
                )
                sum_result.metadata["combsum_score"] = sum_score
                sum_results.append(sum_result)
        
        return sorted(sum_results, key=lambda x: x.score, reverse=True)


# Example usage and testing
async def main():
    """Example usage of the result processor."""
    
    voyage_config = {
        "api_key": "your_voyage_api_key",
        "rerank_model": "rerank-2"
    }
    
    processor = ResultProcessor(voyage_config)
    
    # Create sample results
    semantic_results = [
        SearchResult("1", "Machine learning is a subset of AI", 0.9, "semantic", {}),
        SearchResult("2", "Deep learning uses neural networks", 0.8, "semantic", {}),
        SearchResult("3", "AI applications in healthcare", 0.7, "semantic", {})
    ]
    
    bm25_results = [
        SearchResult("4", "Machine learning algorithms and AI", 0.85, "bm25", {}),
        SearchResult("5", "Neural networks for deep learning", 0.75, "bm25", {}),
        SearchResult("6", "Data science and analytics", 0.65, "bm25", {})
    ]
    
    print("=== Result Processing Tests ===")
    
    # Test combination and deduplication
    combined = await processor.combine_and_deduplicate(semantic_results, bm25_results)
    print(f"Combined results: {len(combined)}")
    
    # Test reranking
    query = "What is machine learning?"
    reranked = await processor.rerank_results(query, combined, top_k=5)
    print(f"Reranked results: {len(reranked)}")
    
    # Test analysis
    analysis = processor.analyze_result_distribution(reranked)
    print(f"Result analysis: {analysis}")
    
    # Test advanced processor
    advanced_processor = AdvancedResultProcessor(voyage_config, enable_ml_features=False)
    
    # Test RRF fusion
    rrf_results = await advanced_processor.adaptive_score_fusion(
        semantic_results, bm25_results, fusion_strategy="rrf"
    )
    print(f"RRF fusion results: {len(rrf_results)}")
    
    # Test weighted fusion
    weighted_results = await advanced_processor.adaptive_score_fusion(
        semantic_results, bm25_results, fusion_strategy="weighted"
    )
    print(f"Weighted fusion results: {len(weighted_results)}")


if __name__ == "__main__":
    asyncio.run(main())