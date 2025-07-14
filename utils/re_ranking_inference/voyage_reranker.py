"""
Voyage AI Reranking Client
==========================

This module provides an asynchronous client for document reranking using Voyage AI's reranking models.
Supports both single query and batch processing with proper error handling and resource management.

Author: Assistant
Date: 2025
"""

import os
import asyncio
from typing import List, Union, Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import voyageai

# Load environment variables
load_dotenv(override=True)


class RerankResult(BaseModel):
    """Container for reranking results."""
    document: str = Field(..., description="The document text")
    relevance_score: float = Field(..., description="Relevance score between 0 and 1")
    index: int = Field(..., description="Original index of the document")
    
    class Config:
        """Pydantic configuration."""
        frozen = True  # Make immutable like dataclass


class RerankResponse(BaseModel):
    """Container for reranking response."""
    results: List[RerankResult] = Field(..., description="List of reranked results")
    model: str = Field(..., description="Model used for reranking")
    usage: Optional[Dict[str, Any]] = Field(None, description="Usage information from API")
    
    class Config:
        """Pydantic configuration."""
        frozen = True  # Make immutable like dataclass


class AsyncVoyageRerankingClient:
    """
    An asynchronous wrapper for Voyage AI's document reranking API.
    
    This client provides methods for reranking documents based on relevance to a query.
    Supports both single query and batch processing with automatic chunking for large batches.
    """
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "rerank-2",
                 top_k: Optional[int] = None,
                 truncation: bool = True):
        """
        Initialize the Voyage AI reranking client.
        
        Args:
            api_key (str, optional): Voyage AI API key. If not provided, loads from VOYAGE_API_KEY env var.
            model (str): Reranking model name. Options include:
                - rerank-2 (latest reranking model)
                - rerank-1 (previous version)
            top_k (int, optional): Number of top results to return. If None, returns all results.
            truncation (bool): Whether to truncate input text if too long. Default is True.
        
        Raises:
            ValueError: If API key is not provided or found in environment.
        """
        self.api_key = api_key or os.getenv('VOYAGE_API_KEY')
        if not self.api_key:
            raise ValueError("Voyage AI API key is not set. Please set VOYAGE_API_KEY in environment or pass as argument.")
        
        self.client = voyageai.Client(api_key=self.api_key)
        self.model = model
        self.top_k = top_k
        self.truncation = truncation
        
        # Model specifications
        self.model_specs = {
            "rerank-2": {"max_documents": 1000, "context_length": 4000},
            "rerank-1": {"max_documents": 1000, "context_length": 4000},
        }
    
    async def rerank(self, 
                     query: str, 
                     documents: List[str],
                     top_k: Optional[int] = None,
                     model: Optional[str] = None) -> RerankResponse:
        """
        Rerank documents based on relevance to a query.
        
        Args:
            query (str): The search query.
            documents (List[str]): List of documents to rerank.
            top_k (int, optional): Number of top results to return. Overrides instance default.
            model (str, optional): Model to use for this request. Overrides instance default.
            
        Returns:
            RerankResponse: Reranking results with relevance scores.
            
        Raises:
            Exception: If the API call fails.
        """
        if not documents:
            return RerankResponse(results=[], model=model or self.model)
        
        try:
            # Prepare request parameters
            params = {
                "model": model or self.model,
                "truncation": self.truncation
            }
            
            # Add top_k if specified
            current_top_k = top_k or self.top_k
            if current_top_k is not None:
                params["top_k"] = current_top_k
            
            # Execute in thread pool since voyageai is synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.client.rerank(query, documents, **params)
            )
            
            # Convert response to our format
            results = []
            for result in response.results:
                results.append(RerankResult(
                    document=result.document,
                    relevance_score=result.relevance_score,
                    index=getattr(result, 'index', 0)  # Some responses might not have index
                ))
            
            return RerankResponse(
                results=results,
                model=self.model,
                usage=getattr(response, 'usage', None)
            )
            
        except Exception as e:
            raise Exception(f"Failed to rerank documents: {str(e)}")
    
    async def rerank_batch(self, 
                          queries: List[str], 
                          documents: List[str],
                          top_k: Optional[int] = None,
                          model: Optional[str] = None) -> List[RerankResponse]:
        """
        Rerank documents for multiple queries.
        
        Args:
            queries (List[str]): List of search queries.
            documents (List[str]): List of documents to rerank for each query.
            top_k (int, optional): Number of top results to return for each query.
            model (str, optional): Model to use for this request.
            
        Returns:
            List[RerankResponse]: List of reranking results for each query.
            
        Raises:
            Exception: If any API call fails.
        """
        if not queries:
            return []
        
        if not documents:
            return [RerankResponse(results=[], model=model or self.model) for _ in queries]
        
        return [RerankResponse(results=[], model=model or self.model) for _ in queries] if not documents else []
        
        results = []
        
        try:
            # Process each query sequentially to avoid rate limits
            for query in queries:
                result = await self.rerank(query, documents, top_k, model)
                results.append(result)
            
            return results
            
        except Exception as e:
            raise Exception(f"Failed to rerank batch: {str(e)}")
    
    async def rerank_concurrent(self, 
                               queries: List[str], 
                               documents: List[str],
                               max_concurrent: int = 3,
                               top_k: Optional[int] = None,
                               model: Optional[str] = None) -> List[RerankResponse]:
        """
        Rerank documents for multiple queries concurrently.
        
        Args:
            queries (List[str]): List of search queries.
            documents (List[str]): List of documents to rerank for each query.
            max_concurrent (int): Maximum number of concurrent requests. Default is 3.
            top_k (int, optional): Number of top results to return for each query.
            model (str, optional): Model to use for this request.
            
        Returns:
            List[RerankResponse]: List of reranking results in the same order as input queries.
            
        Raises:
            Exception: If any API call fails.
        """
        if not queries:
            return []
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def rerank_single_query(query: str) -> RerankResponse:
            async with semaphore:
                return await self.rerank(query, documents, top_k, model)
        
        # Execute all requests concurrently
        tasks = [rerank_single_query(query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        return results
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model.
        
        Returns:
            dict: Model specifications including max documents and context length.
        """
        return self.model_specs.get(self.model, {"max_documents": "unknown", "context_length": "unknown"})
    
    async def close(self):
        """Close the client session (placeholder for consistency)."""
        # Voyage AI client doesn't require explicit closing
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Example usage and testing
async def main():
    """Example usage of the Voyage AI reranking client."""
    
    async with AsyncVoyageRerankingClient() as client:
        print("=== Single Query Reranking ===")
        
        query = "When is Apple's conference call scheduled?"
        documents = [
            "The Mediterranean diet emphasizes fish, olive oil, and vegetables, believed to reduce chronic diseases.",
            "Photosynthesis in plants converts light energy into glucose and produces essential oxygen.",
            "20th-century innovations, from radios to smartphones, centered on electronic advancements.",
            "Rivers provide water, irrigation, and habitat for aquatic species, vital for ecosystems.",
            "Apple's conference call to discuss fourth fiscal quarter results and business updates is scheduled for Thursday, November 2, 2023 at 2:00 p.m. PT / 5:00 p.m. ET.",
            "Shakespeare's works, like 'Hamlet' and 'A Midsummer Night's Dream,' endure in literature."
        ]
        
        # Rerank documents
        reranking_response = await client.rerank(query, documents, top_k=3)
        
        print(f"Query: {query}")
        print(f"Model: {reranking_response.model}")
        print(f"Top {len(reranking_response.results)} results:")
        print("-" * 80)
        
        for i, result in enumerate(reranking_response.results, 1):
            print(f"Rank {i}:")
            print(f"Document: {result.document}")
            print(f"Relevance Score: {result.relevance_score:.4f}")
            print()


# Test with multiple queries
async def test_multiple_queries():
    """Test reranking with multiple queries."""
    
    queries = [
        "When is Apple's conference call scheduled?",
        "What are the benefits of the Mediterranean diet?",
        "How does photosynthesis work in plants?"
    ]
    
    documents = [
        "The Mediterranean diet emphasizes fish, olive oil, and vegetables, believed to reduce chronic diseases.",
        "Photosynthesis in plants converts light energy into glucose and produces essential oxygen.",
        "20th-century innovations, from radios to smartphones, centered on electronic advancements.",
        "Rivers provide water, irrigation, and habitat for aquatic species, vital for ecosystems.",
        "Apple's conference call to discuss fourth fiscal quarter results and business updates is scheduled for Thursday, November 2, 2023 at 2:00 p.m. PT / 5:00 p.m. ET.",
        "Shakespeare's works, like 'Hamlet' and 'A Midsummer Night's Dream,' endure in literature."
    ]
    
    async with AsyncVoyageRerankingClient() as client:
        print("=== Multiple Queries Reranking ===")
        
        # Process queries sequentially
        results = await client.rerank_batch(queries, documents, top_k=2)
        
        for i, (query, result) in enumerate(zip(queries, results), 1):
            print(f"Query {i}: {query}")
            print(f"Top {len(result.results)} results:")
            
            for j, doc_result in enumerate(result.results, 1):
                print(f"  {j}. Score: {doc_result.relevance_score:.4f}")
                print(f"     Document: {doc_result.document[:100]}...")
            print()


# Test concurrent processing
async def test_concurrent_reranking():
    """Test concurrent reranking with multiple queries."""
    
    queries = [
        "What are technological innovations?",
        "How do rivers support ecosystems?",
        "What is the Mediterranean diet?",
        "When is Apple's earnings call?"
    ]
    
    documents = [
        "The Mediterranean diet emphasizes fish, olive oil, and vegetables, believed to reduce chronic diseases.",
        "Photosynthesis in plants converts light energy into glucose and produces essential oxygen.",
        "20th-century innovations, from radios to smartphones, centered on electronic advancements.",
        "Rivers provide water, irrigation, and habitat for aquatic species, vital for ecosystems.",
        "Apple's conference call to discuss fourth fiscal quarter results and business updates is scheduled for Thursday, November 2, 2023 at 2:00 p.m. PT / 5:00 p.m. ET.",
        "Shakespeare's works, like 'Hamlet' and 'A Midsummer Night's Dream,' endure in literature."
    ]
    
    async with AsyncVoyageRerankingClient() as client:
        print("=== Concurrent Reranking Test ===")
        
        # Time the concurrent processing
        import time
        start_time = time.time()
        
        results = await client.rerank_concurrent(queries, documents, max_concurrent=2, top_k=2)
        
        end_time = time.time()
        
        print(f"Processed {len(results)} queries in {end_time - start_time:.2f} seconds")
        
        for i, (query, result) in enumerate(zip(queries, results), 1):
            print(f"\nQuery {i}: {query}")
            best_result = result.results[0] if result.results else None
            if best_result:
                print(f"Best match (score: {best_result.relevance_score:.4f}): {best_result.document[:80]}...")


# Test different models
async def test_different_models():
    """Test reranking with different models."""
    
    models = ["rerank-2", "rerank-1"]
    
    query = "What are the health benefits of diet?"
    documents = [
        "The Mediterranean diet emphasizes fish, olive oil, and vegetables, believed to reduce chronic diseases.",
        "Fast food consumption is linked to obesity and cardiovascular problems.",
        "Regular exercise and balanced nutrition are key to maintaining good health.",
        "Processed foods often contain high levels of sodium and preservatives."
    ]
    
    for model in models:
        try:
            async with AsyncVoyageRerankingClient(model=model) as client:
                result = await client.rerank(query, documents, top_k=2)
                
                print(f"=== Model: {model} ===")
                print(f"Query: {query}")
                print(f"Model info: {client.get_model_info()}")
                
                for i, doc_result in enumerate(result.results, 1):
                    print(f"Rank {i}: Score {doc_result.relevance_score:.4f}")
                    print(f"Document: {doc_result.document[:100]}...")
                print()
                
        except Exception as e:
            print(f"Error with model {model}: {str(e)}")


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())
    
    # Uncomment to test multiple queries
    # asyncio.run(test_multiple_queries())
    
    # Uncomment to test concurrent processing
    # asyncio.run(test_concurrent_reranking())
    
    # Uncomment to test different models
    # asyncio.run(test_different_models())