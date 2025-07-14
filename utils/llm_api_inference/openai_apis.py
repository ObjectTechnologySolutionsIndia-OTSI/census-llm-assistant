import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

class AsyncOpenAIClient:
    """
    An asynchronous wrapper around the OpenAI Chat API providing async chat interfaces.
    """
    def __init__(self,
                 api_key: str = None,
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.7):
        """
        Initialize the async OpenAI client with API key, model, and temperature.
        Args:
            api_key (str, optional): Your OpenAI API key. If not provided, it will be loaded from environment.
            model (str): Model name to be used for chat completions.
            temperature (float): Sampling temperature for the model.
        """
        # Load environment variables, override existing
        load_dotenv(override=True)
        # Get API key from argument or environment
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is not set. Please set OPENAI_API_KEY in env or pass as argument.")
        # Instantiate the async OpenAI client
        self.client = AsyncOpenAI(api_key=self.api_key)
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
        Perform an asynchronous chat completion.
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
        # Extract and return content of the first choice
        return response.choices[0].message.content.strip()

    async def chat_stream(self, system_prompt: str, messages: list):
        """
        Perform an asynchronous streaming chat completion.
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
            # Append new delta and yield the updated reply
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
    # Using the async client
    async with AsyncOpenAIClient() as client:
        # Example conversation
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        # Regular async chat
        response = await client.chat(
            system_prompt="You are a helpful assistant.",
            messages=messages
        )
        print("Regular response:", response)
        
        # Streaming chat
        print("\nStreaming response:")
        async for chunk in client.chat_stream(
            system_prompt="You are a helpful assistant.",
            messages=messages
        ):
            print(f"\rCurrent response: {chunk}", end="")
        print()  # New line after streaming


# Alternative usage without context manager:
async def example_without_context_manager():
    client = AsyncOpenAIClient()
    try:
        messages = [{"role": "user", "content": "What is Python?"}]
        response = await client.chat(
            system_prompt="You are a helpful programming assistant.",
            messages=messages
        )
        print(response)
    finally:
        await client.close()


if __name__ == "__main__":
    # Run the async example
    asyncio.run(main())