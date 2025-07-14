import os
import asyncio
from dotenv import load_dotenv
import anthropic

load_dotenv(override=True)

class AsyncAnthropicClient:
    """
    An asynchronous wrapper around the Anthropic Claude API providing async chat interfaces.
    """
    def __init__(self,
                 api_key: str = None,
                 model: str = "claude-3-5-sonnet-latest",
                 max_tokens: int = 200,
                 temperature: float = 0.7):
        """
        Initialize the async Anthropic client with API key, model, token limit, and temperature.
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API key is not set. Please set ANTHROPIC_API_KEY in environment or pass as argument.")
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def chat(self, system_prompt: str, messages: list) -> str:
        """
        Perform an asynchronous chat completion with Claude using the messages API.
        """
        # Claude v3 usage: embed system and user messages
        message = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=messages
        )
        # The response content is in message.content[0].text
        return message.content[0].text.strip()

    async def chat_stream(self, system_prompt: str, messages: list):
        """
        Perform an asynchronous streaming chat completion with Claude, yielding each text chunk as it arrives.
        """
        # Initiate streaming request
        result = self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=messages
        )
        
        # Use async context manager to iterate over text_stream
        async with result as stream:
            async for text in stream.text_stream:
                yield text

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
    async with AsyncAnthropicClient() as client:
        # Example conversation
        messages = [
            {"role": "user", "content": "Hello, how are you doing today?"}
        ]
        
        # Regular async chat
        response = await client.chat(
            system_prompt="You are a helpful assistant.",
            messages=messages
        )
        print("Regular response:", response)
        
        # Streaming chat
        print("\nStreaming response:")
        full_response = ""
        async for chunk in client.chat_stream(
            system_prompt="You are a helpful assistant.",
            messages=messages
        ):
            full_response += chunk
            print(chunk, end="", flush=True)
        print(f"\n\nFull response: {full_response}")


# Alternative usage without context manager:
async def example_without_context_manager():
    client = AsyncAnthropicClient()
    try:
        messages = [{"role": "user", "content": "What is machine learning?"}]
        response = await client.chat(
            system_prompt="You are a helpful AI assistant specializing in technology.",
            messages=messages
        )
        print(response)
    finally:
        await client.close()


# Example with multiple concurrent requests:
async def concurrent_requests_example():
    async with AsyncAnthropicClient() as client:
        # Multiple requests that can run concurrently
        tasks = []
        questions = [
            "What is Python?",
            "Explain machine learning",
            "What is cloud computing?"
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


if __name__ == "__main__":
    # Run the async example
    asyncio.run(main())
    
    # Uncomment to run other examples:
    # asyncio.run(example_without_context_manager())
    # asyncio.run(concurrent_requests_example())