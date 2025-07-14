import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
import google.generativeai

load_dotenv(override=True)

class AsyncGeminiClient:
    """
    An asynchronous wrapper around Google Gemini via the OpenAI-compatible API endpoint, providing async chat interfaces.
    """
    def __init__(self,
                 api_key: str = None,
                 model: str = "gemini-2.0-flash-exp",
                 base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
                 temperature: float = 0.7):
        """
        Initialize the async Gemini client with API key, model, base URL, and temperature.
        Args:
            api_key (str, optional): Your Google API key. If not provided, will be loaded from GOOGLE_API_KEY env var.
            model (str): Model name to be used for completions.
            base_url (str): Base URL for the Gemini endpoint.
            temperature (float): Sampling temperature for the model.
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key is not set. Please set GOOGLE_API_KEY in environment or pass as argument.")
        # Instantiate the async OpenAI-compatible client for Gemini
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=base_url)
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
        Perform an asynchronous chat completion with Gemini.
        Args:
            system_prompt (str): The system-level instruction.
            messages (list): List of user/assistant messages.
        Returns:
            str: The assistant's response content.
        """
        payload = self._prepare_messages(system_prompt, messages)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=self.temperature
        )
        return response.choices[0].message.content.strip()

    async def chat_stream(self, system_prompt: str, messages: list):
        """
        Perform an asynchronous streaming chat completion with Gemini, yielding cumulative chunks.
        Args:
            system_prompt (str): The system-level instruction.
            messages (list): List of user/assistant messages.
        Yields:
            str: Cumulative response chunks as they arrive.
        """
        payload = self._prepare_messages(system_prompt, messages)
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=payload,
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
    async with AsyncGeminiClient() as client:
        # Example conversation
        messages = [
            {"role": "user", "content": "Hello, can you help me understand quantum computing?"}
        ]
        
        # Regular async chat
        response = await client.chat(
            system_prompt="You are a knowledgeable physics professor who explains complex concepts simply.",
            messages=messages
        )
        print("Regular response:", response)
        
        # Streaming chat
        print("\nStreaming response:")
        async for chunk in client.chat_stream(
            system_prompt="You are a helpful science tutor.",
            messages=[{"role": "user", "content": "Explain photosynthesis in simple terms"}]
        ):
            print(f"\rCurrent response: {chunk}", end="")
        print()  # New line after streaming


# Alternative usage without context manager:
async def example_without_context_manager():
    client = AsyncGeminiClient()
    try:
        messages = [{"role": "user", "content": "What is artificial intelligence?"}]
        response = await client.chat(
            system_prompt="You are an AI expert who provides clear, accessible explanations.",
            messages=messages
        )
        print(response)
    finally:
        await client.close()


# Example with multiple concurrent requests:
async def concurrent_requests_example():
    async with AsyncGeminiClient() as client:
        # Multiple requests that can run concurrently
        tasks = []
        questions = [
            "What is machine learning?",
            "Explain neural networks",
            "What is natural language processing?"
        ]
        
        for question in questions:
            task = client.chat(
                system_prompt="You are a helpful AI assistant specializing in technology.",
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
    async with AsyncGeminiClient() as client:
        # Build a conversation
        messages = [
            {"role": "user", "content": "Hi, I'm interested in learning about space exploration."},
            {"role": "assistant", "content": "Hello! Space exploration is fascinating. What specific aspect interests you most?"},
            {"role": "user", "content": "Tell me about Mars missions and what we've discovered."}
        ]
        
        response = await client.chat(
            system_prompt="You are an enthusiastic space science educator with deep knowledge of planetary exploration.",
            messages=messages
        )
        print("Conversation response:", response)


# Example with streaming and real-time display:
async def streaming_example():
    async with AsyncGeminiClient() as client:
        messages = [
            {"role": "user", "content": "Write a short story about a robot discovering emotions."}
        ]
        
        print("Streaming creative response:")
        full_response = ""
        async for chunk in client.chat_stream(
            system_prompt="You are a creative writer who crafts engaging, thoughtful stories.",
            messages=messages
        ):
            # Only print the new part (delta)
            new_text = chunk[len(full_response):]
            print(new_text, end="", flush=True)
            full_response = chunk
        print(f"\n\nFull response length: {len(full_response)} characters")


if __name__ == "__main__":
    # Run the async example
    asyncio.run(main())
    
    # Uncomment to run other examples:
    # asyncio.run(example_without_context_manager())
    # asyncio.run(concurrent_requests_example())
    # asyncio.run(conversation_example())
    # asyncio.run(streaming_example())