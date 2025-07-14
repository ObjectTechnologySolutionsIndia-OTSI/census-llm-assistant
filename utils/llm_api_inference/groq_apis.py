import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from groq import AsyncGroq

# Load environment variables (overrides existing values)
load_dotenv(override=True)

class AsyncGROQClient:
    """
    An asynchronous wrapper around the GROQ API providing async chat completions using the native async Groq client.
    """
    def __init__(self,
                 api_key: str = None,
                 model: str = "llama-3.3-70b-versatile",
                 temperature: float = 0.7):
        """
        Initialize the async GROQ client with API key, model, and temperature.
        Args:
            api_key (str, optional): Your GROQ API key. If not provided, will be loaded from GROQ_API_KEY env var.
            model (str): Model name to be used for completions.
            temperature (float): Sampling temperature for the model.
        """
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ API key is not set. Please set GROQ_API_KEY in environment or pass as argument.")
        self.client = AsyncGroq(api_key=self.api_key)
        self.model = model
        self.temperature = temperature

    def _prepare_messages(self, system_prompt: str, messages: list) -> list:
        """
        Prepend a system prompt message to the conversation history.
        Args:
            system_prompt (str): The system-level instruction.
            messages (list): Conversation history excluding system message.
        Returns:
            list: New list of messages ready for the API call.
        """
        return [{"role": "system", "content": system_prompt}] + messages

    async def chat(self, system_prompt: str, messages: list) -> str:
        """
        Perform an asynchronous chat completion with GROQ.
        Args:
            system_prompt (str): The system-level instruction.
            messages (list): List of user/assistant messages.
        Returns:
            str: The assistant's response content.
        """
        payload = self._prepare_messages(system_prompt, messages)
        chat_completion = await self.client.chat.completions.create(
            messages=payload,
            model=self.model,
            temperature=self.temperature
        )
        return chat_completion.choices[0].message.content.strip()

    async def chat_stream(self, system_prompt: str, messages: list):
        """
        Perform an asynchronous streaming chat completion with GROQ, yielding cumulative chunks.
        Args:
            system_prompt (str): The system-level instruction.
            messages (list): List of user/assistant messages.
        Yields:
            str: Cumulative response chunks as they arrive.
        """
        payload = self._prepare_messages(system_prompt, messages)
        stream = await self.client.chat.completions.create(
            messages=payload,
            model=self.model,
            temperature=self.temperature,
            stream=True
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ''
            yield delta

    async def close(self):
        """
        Close the async client session.
        """
        await self.client.close()

    async def __aenter__(self):
        """
        Async context manager entry.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit.
        """
        await self.close()


# Example usage:
async def main():
    # Using the async client with context manager
    async with AsyncGROQClient() as client:
        # Example conversation
        messages = [
            {"role": "user", "content": "Hello, can you help me understand machine learning?"}
        ]
        
        # Regular async chat
        response = await client.chat(
            system_prompt="You are a knowledgeable AI expert who explains concepts clearly and concisely.",
            messages=messages
        )
        print("Regular response:", response)
        
        # Streaming chat
        print("\nStreaming response:")
        async for chunk in client.chat_stream(
            system_prompt="You are a helpful coding assistant.",
            messages=[{"role": "user", "content": "Explain Python list comprehensions with examples"}]
        ):
            print(f"\rCurrent response: {chunk}", end="")
        print()  # New line after streaming


# Alternative usage without context manager:
async def example_without_context_manager():
    client = AsyncGROQClient()
    try:
        messages = [{"role": "user", "content": "What is deep learning?"}]
        response = await client.chat(
            system_prompt="You are an AI researcher who provides detailed technical explanations.",
            messages=messages
        )
        print(response)
    finally:
        await client.close()


# Example with multiple concurrent requests:
async def concurrent_requests_example():
    async with AsyncGROQClient() as client:
        # Multiple requests that can run concurrently
        tasks = []
        questions = [
            "What is natural language processing?",
            "Explain computer vision",
            "What is reinforcement learning?"
        ]
        
        for question in questions:
            task = client.chat(
                system_prompt="You are a helpful AI tutor specializing in machine learning.",
                messages=[{"role": "user", "content": question}]
            )
            tasks.append(task)
        
        # Wait for all responses concurrently
        responses = await asyncio.gather(*tasks)
        
        for i, response in enumerate(responses):
            print(f"Question {i+1}: {questions[i]}")
            print(f"Answer: {response}\n")


# Example with conversation history:
async def conversation_example():
    async with AsyncGROQClient() as client:
        # Build a conversation
        messages = [
            {"role": "user", "content": "Hi, I'm learning to code in Python."},
            {"role": "assistant", "content": "Hello! That's great. Python is an excellent language to learn. What would you like to know about?"},
            {"role": "user", "content": "Can you explain what functions are and how to use them?"}
        ]
        
        response = await client.chat(
            system_prompt="You are a patient Python instructor who provides clear examples and explanations.",
            messages=messages
        )
        print("Conversation response:", response)


# Example with streaming and real-time display:
async def streaming_example():
    async with AsyncGROQClient() as client:
        messages = [
            {"role": "user", "content": "Write a Python function that calculates the factorial of a number using recursion."}
        ]
        
        print("Streaming code response:")
        full_response = ""
        async for chunk in client.chat_stream(
            system_prompt="You are an expert Python developer who writes clean, well-documented code.",
            messages=messages
        ):
            # Only print the new part (delta)
            new_text = chunk[len(full_response):]
            print(new_text, end="", flush=True)
            full_response = chunk
        print(f"\n\nFull response length: {len(full_response)} characters")


# Example with different models:
async def model_comparison_example():
    models = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"]
    question = "Explain the concept of recursion in programming."
    
    for model in models:
        async with AsyncGROQClient(model=model) as client:
            response = await client.chat(
                system_prompt="You are a computer science educator.",
                messages=[{"role": "user", "content": question}]
            )
            print(f"\n{model} response:")
            print(response)
            print("-" * 50)


if __name__ == "__main__":
    # Run the async example
    asyncio.run(main())
    
    # Uncomment to run other examples:
    # asyncio.run(example_without_context_manager())
    # asyncio.run(concurrent_requests_example())
    # asyncio.run(conversation_example())
    # asyncio.run(streaming_example())
    # asyncio.run(model_comparison_example())