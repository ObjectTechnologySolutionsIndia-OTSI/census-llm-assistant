from utils.retrive_semantic_bm25_rerank.main_orchestrator_v1 import (
    logger,
    SearchOptions,
    SemanticSearchPipeline,
)
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()


# Enhanced example usage with chat streaming
async def example_with_chat_streaming():
    """Example demonstrating search with chat streaming."""
    
    # Configuration
    # Configuration
    anthropic_config = {"model": "claude-3-sonnet-20240229"}

    voyage_config = {"embedding_model": "voyage-3-large", "rerank_model": "rerank-2"}

    db_config = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "user": os.getenv("POSTGRES_USERNAME"),
        "password": os.getenv("POSTGRES_PASSWORD"),
        "database": os.getenv("POSTGRES_DBNAME"),
    }
    
    # Initialize pipeline
        # Initialize pipeline
    pipeline = SemanticSearchPipeline(
        anthropic_config=anthropic_config,
        voyage_config=voyage_config,
        db_config=db_config,
        bm25_index_path="./utils/bm25_inference/bm25_index.pkl"
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
        
        # # Example 2: Get complete response (non-streaming)
        # search_result, chat_response = await pipeline.search_and_chat_complete(
        #     "How does deep learning work?", 
        #     options
        # )
        
        # print(f"\n=== Complete Response Example ===")
        # print(f"Question: How does deep learning work?")
        # print(f"Search Results: {len(search_result.reranked_results)}")
        # print(f"Response: {chat_response}")
        
        
        
        # # Example 3: Custom system prompt
        # custom_prompt = """You are a technical expert. Provide detailed, technical explanations based on the context. 
        # Include specific technical terms and concepts. Reference the sentence numbers when citing information."""
        
        # print(f"\n=== Custom System Prompt Example ===")
        # async for chunk in pipeline.search_and_chat_stream(
        #     "Explain neural network architectures", 
        #     options, 
        #     custom_system_prompt=custom_prompt
        # ):
        #     print(chunk, end="", flush=True)
        
        # print("\n\nCustom prompt response completed!")
        
        # # Example with context preview
        # preview_search_result = await pipeline.process_query(
        #     "What is artificial intelligence?", 
        #     options,
        #     top_k_final=10
        # )
        
        # print(f"\n=== Context Preview Example ===")
        # context_preview = pipeline.get_context_preview(preview_search_result.reranked_results, max_sentences=3)
        # print(f"Context Preview (first 3 sentences):")
        # print(context_preview)
        
        # context_stats = pipeline.get_context_stats(preview_search_result.reranked_results)
        # print(f"\nContext Statistics: {context_stats}")
        
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
