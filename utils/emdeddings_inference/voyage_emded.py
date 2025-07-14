"""
Voyage AI Embeddings Client
===========================

This module provides an asynchronous client for generating text embeddings using Voyage AI's embedding models.
Supports both single text and batch processing with proper error handling and resource management.

Author: Assistant
Date: 2025
"""

import os
import asyncio
from typing import List, Union, Optional
from dotenv import load_dotenv
import voyageai
import numpy as np

# Load environment variables
load_dotenv(override=True)


class AsyncVoyageEmbeddingsClient:
    """
    An asynchronous wrapper for Voyage AI's text embedding API.
    
    This client provides methods for generating embeddings from text using various Voyage AI models.
    Supports both single text and batch processing with automatic chunking for large batches.
    """
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "voyage-3-large",
                 input_type: Optional[str] = None,
                 truncation: bool = True):
        """
        Initialize the Voyage AI embeddings client.
        
        Args:
            api_key (str, optional): Voyage AI API key. If not provided, loads from VOYAGE_API_KEY env var.
            model (str): Embedding model name. Options include:
                - voyage-3-large (1024 dimensions, best performance)
                - voyage-3-lite (512 dimensions, faster/cheaper)
                - voyage-code-2 (1536 dimensions, optimized for code)
                - voyage-finance-2 (1024 dimensions, optimized for finance)
                - voyage-multilingual-2 (1024 dimensions, multilingual support)
                - voyage-law-2 (1024 dimensions, optimized for legal documents)
            input_type (str, optional): Type of input text. Options:
                - "query": For search queries
                - "document": For documents to be searched
                - None: Let the model decide
            truncation (bool): Whether to truncate input text if too long. Default is True.
        
        Raises:
            ValueError: If API key is not provided or found in environment.
        """
        self.api_key = api_key or os.getenv('VOYAGE_API_KEY')
        if not self.api_key:
            raise ValueError("Voyage AI API key is not set. Please set VOYAGE_API_KEY in environment or pass as argument.")
        
        self.client = voyageai.Client(api_key=self.api_key)
        self.model = model
        self.input_type = input_type
        self.truncation = truncation
        
        # Model specifications
        self.model_specs = {
            "voyage-3-large": {"dimensions": 1024, "context_length": 32000},
            "voyage-3-lite": {"dimensions": 512, "context_length": 32000},
            "voyage-code-2": {"dimensions": 1536, "context_length": 16000},
            "voyage-finance-2": {"dimensions": 1024, "context_length": 32000},
            "voyage-multilingual-2": {"dimensions": 1024, "context_length": 32000},
            "voyage-law-2": {"dimensions": 1024, "context_length": 32000},
        }
    
    async def get_embedding(self, text: str, input_type: Optional[str] = None) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text (str): Input text to embed.
            input_type (str, optional): Override default input type for this request.
            
        Returns:
            List[float]: Embedding vector as a list of floats.
            
        Raises:
            Exception: If the API call fails.
        """
        try:
            # Prepare request parameters
            params = {
                "model": self.model,
                "input": text,
                "truncation": self.truncation
            }
            
            # Add input_type if specified
            current_input_type = input_type or self.input_type
            if current_input_type:
                params["input_type"] = current_input_type
            
            # Execute in thread pool since voyageai is synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.client.embed([text], **{k: v for k, v in params.items() if k != 'input'})
            )
            
            return response.embeddings[0]
            
        except Exception as e:
            raise Exception(f"Failed to generate embedding for text: {str(e)}")
    
    async def get_embeddings_batch(self, 
                                   texts: List[str],
                                   batch_size: int = 128,
                                   input_type: Optional[str] = None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts (List[str]): List of texts to embed.
            batch_size (int): Number of texts to process in each batch. Default is 128.
            input_type (str, optional): Override default input type for this request.
            
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
                    "truncation": self.truncation
                }
                
                # Add input_type if specified
                current_input_type = input_type or self.input_type
                if current_input_type:
                    params["input_type"] = current_input_type
                
                # Execute in thread pool since voyageai is synchronous
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda batch=batch: self.client.embed(batch, **params)
                )
                
                # Extract embeddings from response
                batch_embeddings = response.embeddings
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            raise Exception(f"Failed to generate batch embeddings: {str(e)}")
    
    async def get_embeddings_concurrent(self, 
                                        texts: List[str],
                                        max_concurrent: int = 5,
                                        input_type: Optional[str] = None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts concurrently.
        
        Args:
            texts (List[str]): List of texts to embed.
            max_concurrent (int): Maximum number of concurrent requests. Default is 5.
            input_type (str, optional): Override default input type for this request.
            
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
                return await self.get_embedding(text, input_type)
        
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
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model.
        
        Returns:
            dict: Model specifications including dimensions and context length.
        """
        return self.model_specs.get(self.model, {"dimensions": "unknown", "context_length": "unknown"})
    
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
    """Example usage of the Voyage AI embeddings client."""
    
    async with AsyncVoyageEmbeddingsClient() as client:
        # Single text embedding
        print("=== Single Text Embedding ===")
        text = "The quick brown fox jumps over the lazy dog."
        embedding = await client.get_embedding(text)
        print(f"Text: {text}")
        print(f"Embedding dimensions: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        print(f"Model info: {client.get_model_info()}")
        
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


# Test with different models and input types
async def test_different_models():
    """Test embeddings with different Voyage AI models."""
    
    models = [
        "voyage-3-large",
        "voyage-3-lite",
        "voyage-code-2",
        "voyage-finance-2",
        "voyage-multilingual-2",
    ]
    
    text = "Artificial intelligence is transforming the world."
    
    for model in models:
        try:
            async with AsyncVoyageEmbeddingsClient(model=model) as client:
                embedding = await client.get_embedding(text)
                model_info = client.get_model_info()
                print(f"Model: {model}")
                print(f"Expected dimensions: {model_info['dimensions']}")
                print(f"Actual embedding size: {len(embedding)}")
                print(f"Sample values: {embedding[:3]}")
                print("-" * 50)
        except Exception as e:
            print(f"Error with model {model}: {str(e)}")


# Test with different input types
async def test_input_types():
    """Test embeddings with different input types."""
    
    query_text = "What is machine learning?"
    document_text = "Machine learning is a method of data analysis that automates analytical model building."
    
    async with AsyncVoyageEmbeddingsClient() as client:
        # Generate embeddings with different input types
        query_embedding = await client.get_embedding(query_text, input_type="query")
        doc_embedding = await client.get_embedding(document_text, input_type="document")
        
        # Calculate similarity
        similarity = client.cosine_similarity(query_embedding, doc_embedding)
        
        print("=== Input Type Testing ===")
        print(f"Query: {query_text}")
        print(f"Document: {document_text}")
        print(f"Query-Document similarity: {similarity:.4f}")


# Test concurrent processing
async def test_concurrent_processing():
    """Test concurrent embedding generation."""
    
    texts = [
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret and analyze visual information.",
        "Robotics combines AI with mechanical engineering for autonomous systems.",
        "Machine learning algorithms learn patterns from data without explicit programming.",
        "Deep learning uses neural networks with multiple layers for complex pattern recognition."
    ]
    
    async with AsyncVoyageEmbeddingsClient() as client:
        print("=== Concurrent Processing Test ===")
        
        # Time the concurrent processing
        import time
        start_time = time.time()
        
        embeddings = await client.get_embeddings_concurrent(texts, max_concurrent=3)
        
        end_time = time.time()
        
        print(f"Processed {len(embeddings)} embeddings in {end_time - start_time:.2f} seconds")
        print(f"Average embedding dimensions: {len(embeddings[0])}")


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())
    
    # Uncomment to test different models
    # asyncio.run(test_different_models())
    
    # Uncomment to test input types
    # asyncio.run(test_input_types())
    
    # Uncomment to test concurrent processing
    # asyncio.run(test_concurrent_processing())