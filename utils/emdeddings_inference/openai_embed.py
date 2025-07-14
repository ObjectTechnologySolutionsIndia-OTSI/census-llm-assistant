"""
OpenAI Embeddings Client
========================

This module provides an asynchronous client for generating text embeddings using OpenAI's embedding models.
Supports both single text and batch processing with proper error handling and resource management.

Author: Assistant
Date: 2025
"""

import os
import asyncio
from typing import List, Union, Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI
import numpy as np

# Load environment variables
load_dotenv(override=True)


class AsyncOpenAIEmbeddingsClient:
    """
    An asynchronous wrapper for OpenAI's text embedding API.
    
    This client provides methods for generating embeddings from text using various OpenAI models.
    Supports both single text and batch processing with automatic chunking for large batches.
    """
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "text-embedding-3-large",
                 dimensions: Optional[int] = None):
        """
        Initialize the OpenAI embeddings client.
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided, loads from OPENAI_API_KEY env var.
            model (str): Embedding model name. Options include:
                - text-embedding-3-large (3072 dimensions, best performance)
                - text-embedding-3-small (1536 dimensions, faster/cheaper)
                - text-embedding-ada-002 (1536 dimensions, legacy)
            dimensions (int, optional): Number of dimensions for the embedding vector.
                Only supported by text-embedding-3-* models.
        
        Raises:
            ValueError: If API key is not provided or found in environment.
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is not set. Please set OPENAI_API_KEY in environment or pass as argument.")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self.dimensions = dimensions
        
        # Validate model and dimensions compatibility
        if dimensions and not model.startswith('text-embedding-3'):
            raise ValueError(f"Dimensions parameter is only supported by text-embedding-3-* models, got {model}")
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text (str): Input text to embed.
            
        Returns:
            List[float]: Embedding vector as a list of floats.
            
        Raises:
            Exception: If the API call fails.
        """
        try:
            # Prepare request parameters
            params = {
                "model": self.model,
                "input": text
            }
            
            # Add dimensions if specified and supported
            if self.dimensions:
                params["dimensions"] = self.dimensions
            
            response = await self.client.embeddings.create(**params)
            return response.data[0].embedding
            
        except Exception as e:
            raise Exception(f"Failed to generate embedding for text: {str(e)}")
    
    async def get_embeddings_batch(self, 
                                   texts: List[str],
                                   batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts (List[str]): List of texts to embed.
            batch_size (int): Number of texts to process in each batch. Default is 100.
            
        Returns:
            List[List[float]]: List of embedding vectors.
            
        Raises:
            Exception: If any API call fails.
        """
        if not texts:
            return []
        
        embeddings = []
        
        try:
            # Process texts in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Prepare request parameters
                params = {
                    "model": self.model,
                    "input": batch
                }
                
                # Add dimensions if specified and supported
                if self.dimensions:
                    params["dimensions"] = self.dimensions
                
                response = await self.client.embeddings.create(**params)
                
                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            raise Exception(f"Failed to generate batch embeddings: {str(e)}")
    
    async def get_embeddings_concurrent(self, 
                                        texts: List[str],
                                        max_concurrent: int = 5) -> List[List[float]]:
        """
        Generate embeddings for multiple texts concurrently.
        
        Args:
            texts (List[str]): List of texts to embed.
            max_concurrent (int): Maximum number of concurrent requests. Default is 5.
            
        Returns:
            List[List[float]]: List of embedding vectors in the same order as input texts.
            
        Raises:
            Exception: If any API call fails.
        """
        if not texts:
            return []
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def get_single_embedding(text: str) -> List[float]:
            async with semaphore:
                return await self.get_embedding(text)
        
        # Execute all requests concurrently
        tasks = [get_single_embedding(text) for text in texts]
        embeddings = await asyncio.gather(*tasks)
        
        return embeddings
    
    def cosine_similarity(self, 
                         embedding1: List[float], 
                         embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1 (List[float]): First embedding vector.
            embedding2 (List[float]): Second embedding vector.
            
        Returns:
            float: Cosine similarity score between -1 and 1.
        """
        # Convert to numpy arrays for easier computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def close(self):
        """Close the async client session."""
        await self.client.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Example usage and testing
async def main():
    """Example usage of the OpenAI embeddings client."""
    
    async with AsyncOpenAIEmbeddingsClient() as client:
        # Single text embedding
        print("=== Single Text Embedding ===")
        text = "The quick brown fox jumps over the lazy dog."
        embedding = await client.get_embedding(text)
        print(f"Text: {text}")
        print(f"Embedding dimensions: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        
        # Batch embedding
        print("\n=== Batch Embeddings ===")
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand human language.",
            "Computer vision enables machines to interpret visual information."
        ]
        
        embeddings = await client.get_embeddings_batch(texts)
        print(f"Generated {len(embeddings)} embeddings")
        
        # Calculate similarities
        print("\n=== Similarity Analysis ===")
        for i, text1 in enumerate(texts):
            for j, text2 in enumerate(texts[i+1:], i+1):
                similarity = client.cosine_similarity(embeddings[i], embeddings[j])
                print(f"Similarity between text {i+1} and {j+1}: {similarity:.4f}")


# Test with different models
async def test_different_models():
    """Test embeddings with different OpenAI models."""
    
    models = [
        ("text-embedding-3-large", None),
        ("text-embedding-3-small", None),
        ("text-embedding-3-large", 1024),  # Reduced dimensions
    ]
    
    text = "Artificial intelligence is transforming the world."
    
    for model, dimensions in models:
        try:
            async with AsyncOpenAIEmbeddingsClient(model=model, dimensions=dimensions) as client:
                embedding = await client.get_embedding(text)
                print(f"Model: {model}, Dimensions: {dimensions or 'default'}")
                print(f"Actual embedding size: {len(embedding)}")
                print(f"Sample values: {embedding[:3]}")
                print("-" * 50)
        except Exception as e:
            print(f"Error with model {model}: {str(e)}")


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())
    
    # Uncomment to test different models
    # asyncio.run(test_different_models())