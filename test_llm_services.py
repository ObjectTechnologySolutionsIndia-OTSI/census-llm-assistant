from utils.llm_api_inference.service import AsyncDeepSeekClient


# Example usage:
# Example usage:
async def main():
    # Using the async client with context manager
    async with AsyncDeepSeekClient() as client:
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



if __name__ == "__main__":
    
    import asyncio
    asyncio.run(main()) 