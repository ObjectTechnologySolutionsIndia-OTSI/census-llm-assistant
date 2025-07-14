"""
Gemini Embeddings Client
========================

This module provides an asynchronous client for generating text embeddings using Google's Gemini embedding models.
Supports both single text and batch processing with proper error handling and resource management.

Author: Assistant
Date: 2025
"""

import os
import asyncio
from typing import List, Union, Optional, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import embed_content_async
import numpy as np

# Load environment variables
load_dotenv(override=True)


class AsyncGeminiEmbeddingsClient:
    """
    An asynchronous wrapper for Google Gemini's text embedding API.
    
    This client provides methods for generating embeddings from text using Gemini models.
    Supports both single text and batch processing with automatic rate limiting.
    """
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "models/text-embedding-004"):
        """
        Initialize the Gemini embeddings client.
        
        Args:
            api_key (str, optional): Google API key. If not provided, loads from GOOGLE_API_KEY env var.
            model (str): Embedding model name. Options include:
                - models/text-embedding-004 (768 dimensions, latest)
                - models/embedding-001 (768 dimensions, legacy)
        
        Raises:
            ValueError: If API key is not provided or found in environment.
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key is not set. Please set GOOGLE_API_KEY in environment or pass as argument.")
        
        # Configure the Gemini client
        genai.configure(api_key=self.api_key)
        self.model = model
        
        # Validate model name
        if not model.startswith('models/'):
            self.model = f"models/{model}"
    
    async def get_embedding(self, 
                           text: str, 
                           task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text (str): Input text to embed.
            task_type (str): Task type for the embedding. Options:
                - retrieval_document: For documents in a retrieval system
                - retrieval_query: For queries in a retrieval system
                - semantic_similarity: For semantic similarity tasks
                - classification: For classification tasks
                - clustering: For clustering tasks
                
        Returns:
            List[float]: Embedding vector as a list of floats.
            
        Raises:
            Exception: If the API call fails.
        """
        try:
            response = await embed_content_async(
                model=self.model,
                content=text,
                task_type=task_type
            )
            
            return response['embedding']
            
        except Exception as e:
            raise Exception(f"Failed to generate embedding for text: {str(e)}")
    
    async def get_embeddings_batch(self, 
                                   texts: List[str],
                                   task_type: str = "RETRIEVAL_DOCUMENT",
                                   batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts (List[str]): List of texts to embed.
            task_type (str): Task type for the embeddings.
                Options: task_type	Description
                    RETRIEVAL_QUERY	Specifies the given text is a query in a search or retrieval setting. Use RETRIEVAL_DOCUMENT for the document side.
                    RETRIEVAL_DOCUMENT	Specifies the given text is a document in a search or retrieval setting.
                    SEMANTIC_SIMILARITY	Specifies the given text is used for Semantic Textual Similarity (STS).
                    CLASSIFICATION	Specifies that the embedding is used for classification.
                    CLUSTERING	Specifies that the embedding is used for clustering.
                    QUESTION_ANSWERING	Specifies that the query embedding is used for answering questions. Use RETRIEVAL_DOCUMENT for the document side.
                    FACT_VERIFICATION	Specifies that the query embedding is used for fact verification. Use RETRIEVAL_DOCUMENT for the document side.
                    CODE_RETRIEVAL_QUERY	Specifies that the query embedding is used for code retrieval for Java and Python. Use RETRIEVAL_DOCUMENT for the document side.
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
                
                # Generate embeddings for the batch
                batch_embeddings = []
                for text in batch:
                    embedding = await self.get_embedding(text, task_type)
                    batch_embeddings.append(embedding)
                    
                    # Small delay to respect rate limits
                    await asyncio.sleep(0.1)
                
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            raise Exception(f"Failed to generate batch embeddings: {str(e)}")
    
    async def get_embeddings_concurrent(self, 
                                        texts: List[str],
                                        task_type: str = "retrieval_document",
                                        max_concurrent: int = 3) -> List[List[float]]:
        """
        Generate embeddings for multiple texts concurrently.
        
        Args:
            texts (List[str]): List of texts to embed.
            task_type (str): Task type for the embeddings.
            max_concurrent (int): Maximum number of concurrent requests. Default is 3.
            
        Returns:
            List[List[float]]: List of embedding vectors in the same order as input texts.
            
        Raises:
            Exception: If any API call fails.
        """
        if not texts:
            return []
        
        # Create semaphore to limit concurrent requests (Gemini has stricter rate limits)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def get_single_embedding(text: str) -> List[float]:
            async with semaphore:
                # Add delay to respect rate limits
                await asyncio.sleep(0.2)
                return await self.get_embedding(text, task_type)
        
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
    
    async def get_embedding_with_metadata(self, 
                                         text: str, 
                                         task_type: str = "retrieval_document") -> Dict[str, Any]:
        """
        Generate embedding with additional metadata.
        
        Args:
            text (str): Input text to embed.
            task_type (str): Task type for the embedding.
            
        Returns:
            Dict[str, Any]: Dictionary containing embedding and metadata.
        """
        try:
            embedding = await self.get_embedding(text, task_type)
            
            return {
                "text": text,
                "embedding": embedding,
                "model": self.model,
                "task_type": task_type,
                "dimensions": len(embedding),
                "text_length": len(text)
            }
            
        except Exception as e:
            raise Exception(f"Failed to generate embedding with metadata: {str(e)}")
    
    async def close(self):
        """Close the client session (cleanup if needed)."""
        # Gemini client doesn't require explicit closing
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Example usage and testing
async def main():
    """Example usage of the Gemini embeddings client."""
    
    async with AsyncGeminiEmbeddingsClient() as client:
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


# Test with different task types
async def test_different_task_types():
    """Test embeddings with different task types."""
    
    task_types = [
        "retrieval_document",
        "retrieval_query", 
        "semantic_similarity",
        "classification",
        "clustering"
    ]
    
    document = "Artificial intelligence is transforming the world."
    query = "What is AI?"
    
    async with AsyncGeminiEmbeddingsClient() as client:
        print("=== Document Embeddings by Task Type ===")
        for task_type in task_types:
            try:
                embedding = await client.get_embedding(document, task_type)
                print(f"Task: {task_type}")
                print(f"Dimensions: {len(embedding)}")
                print(f"Sample values: {embedding[:3]}")
                print("-" * 40)
            except Exception as e:
                print(f"Error with task type {task_type}: {str(e)}")
        
        print("\n=== Query vs Document Similarity ===")
        try:
            doc_embedding = await client.get_embedding(document, "retrieval_document")
            query_embedding = await client.get_embedding(query, "retrieval_query")
            
            similarity = client.cosine_similarity(doc_embedding, query_embedding)
            print(f"Document: {document}")
            print(f"Query: {query}")
            print(f"Similarity: {similarity:.4f}")
            
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")


# Test with metadata
async def test_with_metadata():
    """Test embedding generation with metadata."""
    
    async with AsyncGeminiEmbeddingsClient() as client:
        texts = [
            "Short text",
            "This is a medium-length text that contains more information about the topic.",
            "This is a very long text that contains extensive information about artificial intelligence, machine learning, deep learning, natural language processing, and computer vision technologies that are transforming our world today."
        ]
        
        print("=== Embeddings with Metadata ===")
        for text in texts:
            try:
                result = await client.get_embedding_with_metadata(text)
                print(f"Text length: {result['text_length']}")
                print(f"Dimensions: {result['dimensions']}")
                print(f"Task type: {result['task_type']}")
                print(f"Model: {result['model']}")
                print(f"First 3 embedding values: {result['embedding'][:3]}")
                print("-" * 50)
            except Exception as e:
                print(f"Error processing text: {str(e)}")


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())
    
    # Uncomment to test different task types
    # asyncio.run(test_different_task_types())
    
    # Uncomment to test with metadata
    # asyncio.run(test_with_metadata())