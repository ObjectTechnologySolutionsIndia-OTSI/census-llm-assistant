import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables (overrides existing values)
load_dotenv(override=True)

class AsyncDeepSeekClient:
    """
    An asynchronous wrapper around the DeepSeek Chat API via OpenAI-compatible endpoint, providing async chat interfaces.
    """
    def __init__(self,
                 api_key: str = None,
                 model: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com",
                 temperature: float = 0.7):
        """
        Initialize the async DeepSeek client with API key, model, base URL, and temperature.
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("DeepSeek API key is not set. Please set DEEPSEEK_API_KEY in environment or pass as argument.")
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
        Perform an asynchronous chat completion with DeepSeek.
        Args:
            system_prompt (str): The system-level instruction.
            messages (list): List of user/assistant messages.
        Returns:
            str: The assistant's full response.
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
        Perform an asynchronous streaming chat completion with DeepSeek, yielding cumulative chunks.
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
    async with AsyncDeepSeekClient() as client:
        # Example conversation
        messages = [
            {"role": "user", "content": "Hello, can you help me with Python programming?"}
        ]
        
        # Regular async chat
        response = await client.chat(
            system_prompt="You are a helpful programming assistant specializing in Python.",
            messages=messages
        )
        print("Regular response:", response)
        
        # Streaming chat
        print("\nStreaming response:")
        async for chunk in client.chat_stream(
            system_prompt="You are a helpful programming assistant.",
            messages=[{"role": "user", "content": "Explain async/await in Python"}]
        ):
            print(f"\rCurrent response: {chunk}", end="")
        print()  # New line after streaming


# Alternative usage without context manager:
async def example_without_context_manager():
    client = AsyncDeepSeekClient()
    try:
        messages = [{"role": "user", "content": "What is machine learning?"}]
        response = await client.chat(
            system_prompt="You are an AI expert who explains concepts clearly.",
            messages=messages
        )
        print(response)
    finally:
        await client.close()


# Example with multiple concurrent requests:
async def concurrent_requests_example():
    async with AsyncDeepSeekClient() as client:
        # Multiple requests that can run concurrently
        tasks = []
        questions = [
            "What is Python?",
            "Explain neural networks",
            "What is deep learning?"
        ]
        
        for question in questions:
            task = client.chat(
                system_prompt="You are a helpful technical assistant.",
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
    async with AsyncDeepSeekClient() as client:
        # Build a conversation
        messages = [
            {"role": "user", "content": "Hello, I'm learning Python."},
            {"role": "assistant", "content": "Hello! I'd be happy to help you with Python. What would you like to learn about?"},
            {"role": "user", "content": "Can you explain list comprehensions?"}
        ]
        
        response = await client.chat(
            system_prompt="You are a patient Python tutor who gives clear examples.",
            messages=messages
        )
        print("Conversation response:", response)


if __name__ == "__main__":
    # Run the async example
    asyncio.run(main())
    
    # Uncomment to run other examples:
    # asyncio.run(example_without_context_manager())
    # asyncio.run(concurrent_requests_example())
    # asyncio.run(conversation_example())