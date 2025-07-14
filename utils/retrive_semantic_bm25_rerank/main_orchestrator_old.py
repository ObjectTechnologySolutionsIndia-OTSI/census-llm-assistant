"""
Main Orchestrator for Semantic Search Pipeline
=============================================

This module orchestrates the complete semantic search pipeline with query enhancement,
semantic search, BM25 search, result combination, and reranking.

Author: Assistant
Date: 2025
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import time

from .query_enhancer import QueryEnhancer
from .search_manager import SearchManager
from .result_processor import ResultProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SearchOptions(BaseModel):
    """Configuration options for the search pipeline."""
    
    enhance_query: bool = Field(..., description="Whether to enhance user query for optimal semantic search")
    num_variations: int = Field(0, description="Number of variations of user query for semantic search")
    system_prompt: str = Field("", description="System prompt for generating user query variations")
    
    @validator('num_variations')
    def validate_num_variations(cls, v, values):
        if values.get('enhance_query', False) and v < 1:
            raise ValueError("num_variations cannot be less than 1 if enhance_query is True")
        return v
    
    @validator('system_prompt')
    def validate_system_prompt(cls, v, values):
        if values.get('enhance_query', False) and not v.strip():
            raise ValueError("system_prompt cannot be empty if enhance_query is True")
        return v


@dataclass
class SearchResult:
    """Standardized search result structure."""
    id: str
    chunk: str
    score: float
    source: str  # 'semantic', 'bm25', or 'combined'
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_file: Optional[str] = None


@dataclass
class PipelineResult:
    """Complete pipeline result with timing information."""
    query: str
    enhanced_queries: List[str]
    semantic_results: List[SearchResult]
    bm25_results: List[SearchResult]
    combined_results: List[SearchResult]
    reranked_results: List[SearchResult]
    processing_time: float
    step_timings: Dict[str, float]


class SemanticSearchPipeline:
    """Main pipeline orchestrator for semantic search operations."""
    
    def __init__(self, 
                 anthropic_config: Dict[str, Any],
                 voyage_config: Dict[str, Any],
                 db_config: Dict[str, Any],
                 bm25_index_path: str = "bm25_index.pkl"):
        """
        Initialize the semantic search pipeline.
        
        Args:
            anthropic_config: Configuration for Anthropic client
            voyage_config: Configuration for Voyage AI services
            db_config: Database configuration for vector search
            bm25_index_path: Path to BM25 index file
        """
        self.query_enhancer = QueryEnhancer(anthropic_config)
        self.search_manager = SearchManager(voyage_config, db_config, bm25_index_path)
        self.result_processor = ResultProcessor(voyage_config)
        
    async def initialize(self):
        """Initialize all components."""
        await self.search_manager.initialize()
        logger.info("Pipeline initialized successfully")
    
    async def validate_options(self, options: SearchOptions) -> bool:
        """
        Validate search options.
        
        Args:
            options: Search options to validate
            
        Returns:
            bool: True if valid
        """
        try:
            # Pydantic validation happens automatically
            return True
        except Exception as e:
            logger.error(f"Options validation failed: {e}")
            return False
    
    async def process_query(self, 
                           user_query: str, 
                           options: SearchOptions,
                           top_k_semantic: int = 100,
                           top_k_bm25: int = 100,
                           top_k_final: int = 20) -> PipelineResult:
        """
        Process a user query through the complete pipeline.
        
        Args:
            user_query: Original user query
            options: Search configuration options
            top_k_semantic: Number of results from semantic search
            top_k_bm25: Number of results from BM25 search
            top_k_final: Number of final reranked results
            
        Returns:
            PipelineResult: Complete pipeline results
        """
        start_time = time.time()
        step_timings = {}
        
        logger.info(f"Processing query: {user_query}")
        
        # Step 1: Validate options
        step_start = time.time()
        if not await self.validate_options(options):
            raise ValueError("Invalid search options provided")
        step_timings["validation"] = time.time() - step_start
        
        # Step 2: Query enhancement (if enabled)
        step_start = time.time()
        enhanced_queries = []
        if options.enhance_query:
            enhanced_queries = await self.query_enhancer.generate_query_variations(
                user_query, 
                options.system_prompt, 
                options.num_variations
            )
            search_queries = enhanced_queries
        else:
            search_queries = [user_query]
        step_timings["query_enhancement"] = time.time() - step_start
        print(search_queries)
        # Step 3: Parallel semantic and BM25 search
        step_start = time.time()
        semantic_task = self.search_manager.semantic_search_batch(
            search_queries, top_k_semantic
        )
        bm25_task = self.search_manager.bm25_search_batch(
            search_queries, top_k_bm25
        )
        
        semantic_results, bm25_results = await asyncio.gather(
            semantic_task, bm25_task
        )
        step_timings["parallel_search"] = time.time() - step_start
        
        # Step 4: Combine and deduplicate results
        step_start = time.time()
        combined_results = await self.result_processor.combine_and_deduplicate(
            semantic_results, bm25_results
        )
        step_timings["combination"] = time.time() - step_start
        
        # Step 5: Rerank results
        step_start = time.time()
        reranked_results = await self.result_processor.rerank_results(
            user_query, combined_results, top_k_final
        )
        step_timings["reranking"] = time.time() - step_start
        
        total_time = time.time() - start_time
        
        logger.info(f"Pipeline completed in {total_time:.2f}s")
        
        return PipelineResult(
            query=user_query,
            enhanced_queries=enhanced_queries,
            semantic_results=semantic_results,
            bm25_results=bm25_results,
            combined_results=combined_results,
            reranked_results=reranked_results,
            processing_time=total_time,
            step_timings=step_timings
        )
    
    async def process_batch_queries(self, 
                                   queries_with_options: List[tuple[str, SearchOptions]],
                                   max_concurrent: int = 5) -> List[PipelineResult]:
        """
        Process multiple queries concurrently.
        
        Args:
            queries_with_options: List of (query, options) tuples
            max_concurrent: Maximum concurrent processing
            
        Returns:
            List of pipeline results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(query_options: tuple[str, SearchOptions]) -> PipelineResult:
            async with semaphore:
                query, options = query_options
                return await self.process_query(query, options)
        
        tasks = [process_single(qo) for qo in queries_with_options]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Query {i} failed: {result}")
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def close(self):
        """Clean up resources."""
        await self.search_manager.close()
        logger.info("Pipeline closed")


# Example usage and testing
async def main():
    """Example usage of the semantic search pipeline."""
    
    # Configuration
    anthropic_config = {
        "api_key": "your_anthropic_api_key",
        "model": "claude-3-sonnet-20240229"
    }
    
    voyage_config = {
        "api_key": "your_voyage_api_key",
        "embedding_model": "voyage-3-large",
        "rerank_model": "rerank-2"
    }
    
    db_config = {
        "host": "localhost",
        "port": "5432",
        "database": "your_db",
        "user": "your_user",
        "password": "your_password"
    }
    
    # Initialize pipeline
    pipeline = SemanticSearchPipeline(
        anthropic_config=anthropic_config,
        voyage_config=voyage_config,
        db_config=db_config,
        bm25_index_path="bm25_index.pkl"
    )
    
    try:
        await pipeline.initialize()
        
        # Example 1: Single query with enhancement
        options1 = SearchOptions(
            enhance_query=True,
            num_variations=3,
            system_prompt="Generate semantic variations of the user query for comprehensive document search. Focus on different phrasings and related terms."
        )
        
        result1 = await pipeline.process_query(
            "What are the benefits of machine learning?", 
            options1
        )
        
        print("=== Enhanced Query Search ===")
        print(f"Original query: {result1.query}")
        print(f"Enhanced queries: {result1.enhanced_queries}")
        print(f"Final results count: {len(result1.reranked_results)}")
        print(f"Processing time: {result1.processing_time:.2f}s")
        print("Step timings:", result1.step_timings)
        
        # Example 2: Simple query without enhancement
        options2 = SearchOptions(
            enhance_query=False,
            num_variations=0,
            system_prompt=""
        )
        
        result2 = await pipeline.process_query(
            "Python programming tutorial", 
            options2
        )
        
        print("\n=== Simple Query Search ===")
        print(f"Query: {result2.query}")
        print(f"Final results count: {len(result2.reranked_results)}")
        print(f"Processing time: {result2.processing_time:.2f}s")
        
        # Example 3: Batch processing
        batch_queries = [
            ("artificial intelligence applications", options1),
            ("data science tools", options2),
            ("cloud computing benefits", options1)
        ]
        
        batch_results = await pipeline.process_batch_queries(batch_queries)
        
        print(f"\n=== Batch Processing ===")
        print(f"Processed {len(batch_results)} queries")
        for i, result in enumerate(batch_results):
            print(f"Query {i+1}: {result.processing_time:.2f}s")
    
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    
    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
