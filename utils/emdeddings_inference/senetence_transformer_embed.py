"""
Sentence Transformers Embeddings Client
=======================================

This module provides an asynchronous client for generating text embeddings using Sentence Transformers models.
Supports both single text and batch processing with proper error handling and resource management.

Author: Assistant
Date: 2025
"""

import asyncio
import threading
from typing import List, Union, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class AsyncSentenceTransformersClient:
    """
    An asynchronous wrapper for Sentence Transformers embedding models.
    
    This client provides methods for generating embeddings from text using various pre-trained models.
    Supports both single text and batch processing with GPU acceleration when available.
    """
    
    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 device: Optional[str] = None,
                 cache_folder: Optional[str] = None,
                 trust_remote_code: bool = False,
                 max_workers: int = 4):
        """
        Initialize the Sentence Transformers embeddings client.
        
        Args:
            model_name (str): Name of the model to use. Popular options include:
                - all-MiniLM-L6-v2 (384 dimensions, fast and good performance)
                - all-mpnet-base-v2 (768 dimensions, best quality)
                - multi-qa-MiniLM-L6-cos-v1 (384 dimensions, optimized for Q&A)
                - paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions, multilingual)
            device (str, optional): Device to use ('cuda', 'cpu', or None for auto-detection).
            cache_folder (str, optional): Path to cache downloaded models.
            trust_remote_code (bool): Whether to trust remote code in model files.
            max_workers (int): Maximum number of worker threads for async operations.
        
        Raises:
            Exception: If model fails to load.
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_folder = cache_folder
        self.trust_remote_code = trust_remote_code
        self.max_workers = max_workers
        
        # Thread-local storage for model instances
        self._local = threading.local()
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize model in main thread
        self._init_model()
    
    def _init_model(self):
        """Initialize the model for the current thread."""
        if not hasattr(self._local, 'model'):
            try:
                self._local.model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    cache_folder=self.cache_folder,
                    trust_remote_code=self.trust_remote_code
                )
                print(f"Loaded model '{self.model_name}' on device '{self.device}'")
            except Exception as e:
                raise Exception(f"Failed to load model '{self.model_name}': {str(e)}")
    
    def _get_model(self) -> SentenceTransformer:
        """Get model instance for current thread."""
        if not hasattr(self._local, 'model'):
            self._init_model()
        return self._local.model
    
    def _encode_sync(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Synchronous encoding function for thread pool."""
        model = self._get_model()
        return model.encode(texts, **kwargs)
    
    async def get_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text (str): Input text to embed.
            **kwargs: Additional arguments passed to the model's encode method.
            
        Returns:
            List[float]: Embedding vector as a list of floats.
            
        Raises:
            Exception: If the embedding generation fails.
        """
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self._executor, 
                self._encode_sync, 
                text, 
                **kwargs
            )
            return embedding.tolist()
            
        except Exception as e:
            raise Exception(f"Failed to generate embedding for text: {str(e)}")
    
    async def get_embeddings_batch(self, 
                                   texts: List[str],
                                   batch_size: int = 32,
                                   show_progress_bar: bool = False,
                                   **kwargs) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts (List[str]): List of texts to embed.
            batch_size (int): Number of texts to process in each batch. Default is 32.
            show_progress_bar (bool): Whether to show progress bar.
            **kwargs: Additional arguments passed to the model's encode method.
            
        Returns:
            List[List[float]]: List of embedding vectors.
            
        Raises:
            Exception: If embedding generation fails.
        """
        if not texts:
            return []
        
        try:
            # Set default kwargs for batch processing
            encode_kwargs = {
                'batch_size': batch_size,
                'show_progress_bar': show_progress_bar,
                'convert_to_numpy': True,
                **kwargs
            }
            
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                self._executor, 
                self._encode_sync, 
                texts, 
                **encode_kwargs
            )
            
            return embeddings.tolist()
            
        except Exception as e:
            raise Exception(f"Failed to generate batch embeddings: {str(e)}")
    
    async def get_embeddings_concurrent(self, 
                                        texts: List[str],
                                        max_concurrent: int = 4,
                                        chunk_size: int = 10,
                                        **kwargs) -> List[List[float]]:
        """
        Generate embeddings for multiple texts concurrently by chunking.
        
        Args:
            texts (List[str]): List of texts to embed.
            max_concurrent (int): Maximum number of concurrent chunks. Default is 4.
            chunk_size (int): Size of each chunk to process. Default is 10.
            **kwargs: Additional arguments passed to the model's encode method.
            
        Returns:
            List[List[float]]: List of embedding vectors in the same order as input texts.
            
        Raises:
            Exception: If any embedding generation fails.
        """
        if not texts:
            return []
        
        # Create chunks
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_chunk(chunk: List[str]) -> List[List[float]]:
            async with semaphore:
                return await self.get_embeddings_batch(chunk, **kwargs)
        
        # Execute all chunks concurrently
        tasks = [process_chunk(chunk) for chunk in chunks]
        chunk_results = await asyncio.gather(*tasks)
        
        # Flatten results
        embeddings = []
        for chunk_embeddings in chunk_results:
            embeddings.extend(chunk_embeddings)
        
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Dictionary containing model information.
        """
        model = self._get_model()
        return {
            'model_name': self.model_name,
            'device': self.device,
            'max_seq_length': getattr(model, 'max_seq_length', 'Unknown'),
            'embedding_dimension': model.get_sentence_embedding_dimension(),
            'tokenizer_type': type(model.tokenizer).__name__ if hasattr(model, 'tokenizer') else 'Unknown'
        }
    
    async def close(self):
        """Close the thread pool executor."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Example usage and testing
async def main():
    """Example usage of the Sentence Transformers embeddings client."""
    
    async with AsyncSentenceTransformersClient() as client:
        # Display model information
        print("=== Model Information ===")
        info = client.get_model_info()
        for key, value in info.items():
            print(f"{key}: {value}")
        
        # Single text embedding
        print("\n=== Single Text Embedding ===")
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
            "Computer vision enables machines to interpret visual information.",
            "Reinforcement learning trains agents through trial and error.",
            "Supervised learning uses labeled data for training models."
        ]
        
        embeddings = await client.get_embeddings_batch(texts, show_progress_bar=True)
        print(f"Generated {len(embeddings)} embeddings")
        
        # Calculate similarities
        print("\n=== Similarity Analysis ===")
        for i, text1 in enumerate(texts[:3]):  # Limit to first 3 for brevity
            for j, text2 in enumerate(texts[i+1:4], i+1):
                similarity = client.cosine_similarity(embeddings[i], embeddings[j])
                print(f"Similarity between text {i+1} and {j+1}: {similarity:.4f}")


# Test with different models
async def test_different_models():
    """Test embeddings with different Sentence Transformers models."""
    
    models = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "multi-qa-MiniLM-L6-cos-v1",
    ]
    
    text = "Artificial intelligence is transforming the world."
    
    for model_name in models:
        try:
            async with AsyncSentenceTransformersClient(model_name=model_name) as client:
                embedding = await client.get_embedding(text)
                info = client.get_model_info()
                print(f"Model: {model_name}")
                print(f"Embedding dimensions: {len(embedding)}")
                print(f"Model info: {info}")
                print(f"Sample values: {embedding[:3]}")
                print("-" * 50)
        except Exception as e:
            print(f"Error with model {model_name}: {str(e)}")


# Performance benchmark
async def benchmark_performance():
    """Benchmark different processing methods."""
    
    import time
    
    # Create test data
    texts = [f"This is test sentence number {i} for benchmarking." for i in range(100)]
    
    async with AsyncSentenceTransformersClient() as client:
        print("=== Performance Benchmark ===")
        
        # Test batch processing
        start_time = time.time()
        batch_embeddings = await client.get_embeddings_batch(texts)
        batch_time = time.time() - start_time
        print(f"Batch processing: {batch_time:.2f} seconds for {len(texts)} texts")
        
        # Test concurrent processing
        start_time = time.time()
        concurrent_embeddings = await client.get_embeddings_concurrent(texts, chunk_size=20)
        concurrent_time = time.time() - start_time
        print(f"Concurrent processing: {concurrent_time:.2f} seconds for {len(texts)} texts")
        
        # Verify results are similar
        similarity = client.cosine_similarity(batch_embeddings[0], concurrent_embeddings[0])
        print(f"First embedding similarity: {similarity:.6f} (should be ~1.0)")


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())
    
    # Uncomment to test different models
    # asyncio.run(test_different_models())
    
    # Uncomment to run performance benchmark
    # asyncio.run(benchmark_performance())