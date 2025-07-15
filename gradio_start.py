import openai
import gradio as gr
import asyncio
import os
from utils.retrive_semantic_bm25_rerank.main_orchestrator_v1 import (
    logger,
    SearchOptions,
    SemanticSearchPipeline,
)
from dotenv import load_dotenv

load_dotenv()


# Configuration
anthropic_config = {"model": "claude-3-sonnet-20240229"}

openai_config = {"model": "gpt-4o", "max_tokens": 500, "temperature": 0.1}

voyage_config = {"embedding_model": "voyage-3-large", "rerank_model": "rerank-2"}

db_config = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "user": os.getenv("POSTGRES_USERNAME"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "database": os.getenv("POSTGRES_DBNAME"),
}


# Enhanced example usage with chat streaming
async def chat_stream(user_input, history):
    """Example demonstrating search with chat streaming."""
    history = history or []    
    try:
        # Initialize pipeline
        pipeline = SemanticSearchPipeline(
            anthropic_config=anthropic_config,
            openai_config=openai_config,
            voyage_config=voyage_config,
            db_config=db_config,
            bm25_index_path="./utils/bm25_inference/bm25_index.pkl"
        )

        await pipeline.initialize()
        
        # Example 1: Search and stream chat response
        options = SearchOptions(
            enhance_query=False,
            num_variations=3,
            system_prompt="Generate semantic variations for comprehensive search"
        )
        
        user_question = user_input
        
        print(f"Question: {user_question}")
        print("Streaming Response:")
        print("-" * 50)
        
        full_response = ""
        async for chunk in pipeline.search_and_chat_stream(user_question, options):
            print(chunk, end="", flush=True)
            full_response += chunk
            yield full_response
        
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


# Async generator function to stream responses
# async def chat_stream(user_input, history):
#     history = history or []


# Define async chatbot function
async def respond(message, history):
    print("Received message:", message, history)
    async for updated_chat in chat_stream(message, history):
        yield history+ [[message, updated_chat]]
        
        
        
# Gradio interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    with gr.Row():
        msg = gr.Textbox(
            label="Message",
            placeholder="Type a message and press Enter...",
            scale=10,
            interactive=True,
            show_label=False,
        )
        send_btn = gr.Button("Send", scale=1)
    clear_btn = gr.Button("Clear Chat")

    # Submit with Enter or Send button
    msg.submit(respond, [msg, chatbot], chatbot).then(lambda: "", None, msg)
    send_btn.click(respond, [msg, chatbot], chatbot).then(lambda: "", None, msg)

    # Clear resets chat
    clear_btn.click(lambda: [], None, chatbot)

demo.queue().launch()
