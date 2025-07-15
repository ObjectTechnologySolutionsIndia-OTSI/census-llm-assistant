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
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import time


from .query_enhancer import QueryEnhancer
from .search_manager import SearchManager
from .result_processor import ResultProcessor
from utils.llm_api_inference.anthropic_apis import AsyncAnthropicClient
from utils.llm_api_inference.openai_apis import AsyncOpenAIClient
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
    context_for_chat: str = ""  # Added context string for chat


@dataclass
class ChatStreamResponse:
    """Response container for streaming chat."""
    content: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    model: str = ""
    context_for_chat: str = ""  # Added context string for chat


@dataclass
class ChatStreamResponse:
    """Response container for streaming chat."""
    content: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    model: str = ""


class SemanticSearchPipeline:
    """Main pipeline orchestrator for semantic search operations."""
    
    def __init__(self, 
                 anthropic_config: Dict[str, Any],
                 openai_config: Dict[str, Any],
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
        self.anthropic_config = anthropic_config
        self.query_enhancer = QueryEnhancer(anthropic_config)
        self.search_manager = SearchManager(voyage_config, db_config, bm25_index_path)
        self.result_processor = ResultProcessor(voyage_config)
        
        # Initialize Anthropic client for chat streaming
        self.anthropic_client = AsyncAnthropicClient(
            api_key=anthropic_config.get("api_key"),
            model=anthropic_config.get("model", "claude-3-sonnet-20240229")
        )
        self.openai_client = AsyncOpenAIClient(
            api_key=openai_config.get("api_key"),
            model=openai_config.get("model", "gpt-3.5-turbo")
        )
        self.chat_model = anthropic_config.get("model", "claude-3-sonnet-20240229")
        self.chat_max_tokens = anthropic_config.get("max_tokens", 2000)
        self.chat_temperature = anthropic_config.get("temperature", 0.7)
        
    def _prepare_context_from_results(self, results: List[SearchResult], max_results: int = 3) -> str:
        """
        Prepare context string from search results in ranked order.
        
        Args:
            results: List of search results (already ranked)
            max_results: Maximum number of results to include (default: 3)
            
        Returns:
            Formatted context string with sentence numbering
        """
        if not results:
            return ""
        
        # Take only top N results (default top 3)
        top_results = results[:max_results]
        context_sentences = []
        
        for i, result in enumerate(top_results, 1):
            # Clean and format the chunk
            cleaned_chunk = result.chunk.strip()
            
            # Remove extra whitespace and normalize
            cleaned_chunk = ' '.join(cleaned_chunk.split())
            
            # Add sentence number
            # sentence = f"sentence {i}: {cleaned_chunk}"
            sentence = f"{cleaned_chunk}"
            context_sentences.append(sentence)
        
        return '\n'.join(context_sentences)
    
    def _create_default_system_prompt(self) -> str:
        """Create the default system prompt for chat responses."""
        return """You are an helpful AI assistant. You are tasked to answer user <Question> based on the <Context> provided by following below Instructions:
                    1. Understand the User Intent based on the question
                    2. Analyze the context based on the User Intent
                    3. while answering Pretend as if you answering based on your knowledge
                    4. Output Formatting Guidelines:
                        a. Headings & Subheadings: Use clear hierarchical headings (e.g., ## Main Topic, ### Subtopic).
                        b. Numbered List / Bullet Points/Lists: Break down complex information into scannable lists.
                        c. Tables: Use Markdown tables for comparisons, stats, or structured data.
                        d. Bold Key Terms: Highlight important terms/conclusions.
                        e. Citations: Link or attribute sources if available (e.g., "(Source: XYZ Study, 2023)").
                        f. Conciseness: Prioritize factual accuracy over fluff."""
    
    def _format_user_message(self, user_question: str, context: str) -> str:
        """
        Format the user message with question and context.
        
        Args:
            user_question: The user's question
            context: Formatted context from search results
            
        Returns:
            Formatted message string
        """
        return f"""<Question>{user_question}</Question>

                    <Context>
                    {context}
                    </Context>"""
    
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
                           top_k_final: int = 20,
                           max_results: int = 1
                           ) -> PipelineResult:
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
        
        # Step 3: Parallel semantic and BM25 search
        step_start = time.time()
        if options.enhance_query and enhanced_queries:
            # For enhanced queries, search each query individually
            semantic_results = []
            bm25_results = []
            
            for query in search_queries:
                # Get results for each individual query
                sem_results = await self.search_manager.semantic_search_single(query, top_k_semantic)
                bm25_results_single = await self.search_manager.bm25_search_single(query, top_k_bm25)
                
                semantic_results.extend(sem_results)
                bm25_results.extend(bm25_results_single)
            
        else:
            # Single query - use batch search
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
        
        # Step 6: Prepare context for chat - ALWAYS use top 3 from reranked results
        step_start = time.time()
        context_for_chat = self._prepare_context_from_results(reranked_results, max_results=max_results)

        if options.enhance_query and enhanced_queries:
            logger.info(f"Prepared context from top {max_results} reranked results (after processing {len(enhanced_queries)} enhanced queries)")
        else:
            logger.info(f"Prepared context from top {max_results} reranked results")
        step_timings["context_preparation"] = time.time() - step_start
        
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
            step_timings=step_timings,
            context_for_chat=context_for_chat
        )
    
    async def search_and_chat_stream(self, 
                                    user_question: str,
                                    options: SearchOptions,
                                    custom_system_prompt: Optional[str] = None,
                                    top_k_semantic: int = 10,
                                    top_k_bm25: int = 10,
                                    top_k_final: int = 5,
                                    max_results: int = 1
                                    ) -> AsyncGenerator[str, None]:
        """
        Perform search and stream chat response.
        
        Context Logic:
        - ALWAYS uses top 3 results from final deduplicated and reranked results as context
        - This ensures consistent context quality regardless of query enhancement
        
        Args:
            user_question: User's question
            options: Search configuration options
            custom_system_prompt: Optional custom system prompt
            top_k_semantic: Number of semantic search results per query
            top_k_bm25: Number of BM25 search results per query
            top_k_final: Number of final results (top 3 will be used for context)
            
        Yields:
            Streaming chat response chunks
        """
        try:
            # Perform the complete search pipeline
            logger.info(f"Starting search and chat stream for: {user_question}")
            
            search_result = await self.process_query(
                user_question, 
                options,
                top_k_semantic=top_k_semantic,
                top_k_bm25=top_k_bm25,
                top_k_final=top_k_final,
                max_results=max_results
            )
            
            # Use the prepared context (always top 3 from reranked results)
            context = search_result.context_for_chat
            
            logger.info(f"Using top {max_results} results from {len(search_result.reranked_results)} reranked results as context")
            
            # Create system prompt
            system_prompt = custom_system_prompt or self._create_default_system_prompt()
            
            # Format user message with context
            user_message = self._format_user_message(user_question, context)
            
            # Prepare messages for Anthropic
            messages = [
                {
                    "role": "user",
                    "content": user_message
                }
            ]
            
            # Stream the chat response
            logger.info("Starting chat stream generation")
            async for chunk in self.openai_client.chat_stream(system_prompt, messages):
                yield chunk
            
            logger.info("Chat stream completed successfully")
                
        except Exception as e:
            logger.error(f"Error in search_and_chat_stream: {e}")
            yield f"Error: Unable to process request - {str(e)}"
    
    async def search_and_chat_complete(self, 
                                     user_question: str,
                                     options: SearchOptions,
                                     custom_system_prompt: Optional[str] = None,
                                     top_k_semantic: int = 100,
                                     top_k_bm25: int = 100,
                                     top_k_final: int = 20,
                                     max_results: int = 1
                                     ) -> tuple[PipelineResult, str]:
        """
        Perform search and get complete chat response using TOP 3 reranked results as context.
        
        Args:
            user_question: User's question
            options: Search configuration options
            custom_system_prompt: Optional custom system prompt
            top_k_semantic: Number of semantic search results
            top_k_bm25: Number of BM25 search results
            top_k_final: Number of final results (top 3 will be used for context)
            
        Returns:
            Tuple of (search_result, chat_response)
        """
        try:
            # Perform the complete search pipeline
            logger.info(f"Starting search and chat for: {user_question}")
            
            search_result = await self.process_query(
                user_question, 
                options,
                top_k_semantic=top_k_semantic,
                top_k_bm25=top_k_bm25,
                top_k_final=top_k_final,
                max_results=max_results
            )
            
            # Use ONLY TOP 3 results as context (already prepared in process_query)
            context = search_result.context_for_chat
            
            logger.info(f"Search completed, generating chat response with TOP {max_results} results as context from {len(search_result.reranked_results)} total results")
            
            # Create system prompt
            system_prompt = custom_system_prompt or self._create_default_system_prompt()
            
            # Format user message with context
            user_message = self._format_user_message(user_question, context)
            
            # Prepare messages for Anthropic
            messages = [
                {
                    "role": "user",
                    "content": user_message
                }
            ]
            
            # Get complete chat response
            chat_response = await self.anthropic_client.chat(system_prompt, messages)
            
            logger.info("Chat response generated successfully using top {max_results} results")
            
            return search_result, chat_response
                
        except Exception as e:
            logger.error(f"Error in search_and_chat_complete: {e}")
            return search_result if 'search_result' in locals() else None, f"Error: Unable to generate response - {str(e)}"
    
    def get_context_preview(self, results: List[SearchResult], max_sentences: int = 5) -> str:
        """
        Get a preview of the context that would be created from results.
        
        Args:
            results: Search results to preview
            max_sentences: Maximum number of sentences to show in preview
            
        Returns:
            Context preview string
        """
        preview_results = results[:max_sentences]
        context = self._prepare_context_from_results(preview_results)
        
        if len(results) > max_sentences:
            context += f"\n... (and {len(results) - max_sentences} more sentences)"
        
        return context
    
    def get_context_stats(self, results: List[SearchResult]) -> Dict[str, Any]:
        """
        Get statistics about the context that would be created.
        
        Args:
            results: Search results to analyze
            
        Returns:
            Context statistics
        """
        if not results:
            return {
                "total_results": 0,
                "total_characters": 0,
                "estimated_tokens": 0,
                "avg_result_length": 0,
                "sources": []
            }
        
        context = self._prepare_context_from_results(results)
        
        total_chars = len(context)
        avg_length = sum(len(r.chunk) for r in results) / len(results)
        
        # Rough token estimation (4 characters ≈ 1 token)
        estimated_tokens = total_chars // 4
        
        return {
            "total_results": len(results),
            "total_characters": total_chars,
            "estimated_tokens": estimated_tokens,
            "avg_result_length": avg_length,
            "sources": list(set(r.source for r in results))
        } 
    
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
        await self.anthropic_client.close()
        logger.info("Pipeline closed")


# Enhanced example usage with chat streaming
async def example_with_chat_streaming():
    """Example demonstrating search with chat streaming."""
    
    # Configuration
    anthropic_config = {
        "api_key": "your_anthropic_api_key",
        "model": "claude-3-5-sonnet-latest"
    }
    
    openai_config = {
        "api_key": "your_openai_api_key",
        "model": "gpt-3.5-turbo"
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
        openai_config=openai_config,
        voyage_config=voyage_config,
        db_config=db_config,
        bm25_index_path="/utils/bm25_inference/bm25_index.pkl"
    )
    
    try:
        await pipeline.initialize()
        
        # Example 1: Search and stream chat response
        options = SearchOptions(
            enhance_query=True,
            num_variations=3,
            system_prompt="Generate semantic variations for comprehensive search"
        )
        
        user_question = "What are the benefits of machine learning in healthcare?"
        
        print(f"Question: {user_question}")
        print("Streaming Response:")
        print("-" * 50)
        
        full_response = ""
        async for chunk in pipeline.search_and_chat_stream(user_question, options):
            print(chunk, end="", flush=True)
            full_response += chunk
        
        print(f"\n{'-' * 50}")
        print("Stream completed!")
        
        # Example 2: Get complete response (non-streaming)
        search_result, chat_response = await pipeline.search_and_chat_complete(
            "How does deep learning work?", 
            options
        )
        
        print(f"\n=== Complete Response Example ===")
        print(f"Question: How does deep learning work?")
        print(f"Search Results: {len(search_result.reranked_results)}")
        print(f"Response: {chat_response}")
        
        # Example 3: Custom system prompt
        custom_prompt = """You are a technical expert. Provide detailed, technical explanations based on the context. 
        Include specific technical terms and concepts. Reference the sentence numbers when citing information."""
        
        print(f"\n=== Custom System Prompt Example ===")
        async for chunk in pipeline.search_and_chat_stream(
            "Explain neural network architectures", 
            options, 
            custom_system_prompt=custom_prompt
        ):
            print(chunk, end="", flush=True)
        
        print("\n\nCustom prompt response completed!")
        
        # Example with context preview
        preview_search_result = await pipeline.process_query(
            "What is artificial intelligence?", 
            options,
            top_k_final=10
        )
        
        print(f"\n=== Context Preview Example ===")
        context_preview = pipeline.get_context_preview(preview_search_result.reranked_results, max_sentences=3)
        print(f"Context Preview (first 3 sentences):")
        print(context_preview)
        
        context_stats = pipeline.get_context_stats(preview_search_result.reranked_results)
        print(f"\nContext Statistics: {context_stats}")
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    
    finally:
        await pipeline.close()


# Example usage and testing
async def main():
    """Example usage of the semantic search pipeline with chat streaming."""
    
    # Run the enhanced example with chat streaming
    await example_with_chat_streaming()


if __name__ == "__main__":
    asyncio.run(main())