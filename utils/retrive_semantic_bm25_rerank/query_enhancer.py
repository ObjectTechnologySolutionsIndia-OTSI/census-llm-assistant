"""
Query Enhancer Module
====================

This module handles query enhancement using Anthropic's LLM to generate
semantic variations of user queries for improved search performance.

Author: Assistant
Date: 2025
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
import aiohttp
from dataclasses import dataclass

# Assuming anthropic_apis exists - we'll implement a simplified version
try:
    from utils.llm_api_inference.anthropic_apis import AsyncAnthropicClient
except ImportError:
    # Fallback implementation
    import anthropic
    
    class AsyncAnthropicClient:
        def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model
        
        async def chat(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
            """Async wrapper for Anthropic chat completion."""
            loop = asyncio.get_event_loop()
            
            def sync_chat():
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    system=system_prompt,
                    messages=messages
                )
                return response.content[0].text
            
            return await loop.run_in_executor(None, sync_chat)

logger = logging.getLogger(__name__)


@dataclass
class QueryVariation:
    """Container for query variation result."""
    original_query: str
    variations: List[str]
    system_prompt: str
    processing_time: float


class QueryEnhancer:
    """Handles query enhancement using Anthropic's LLM."""
    
    def __init__(self, anthropic_config: Dict[str, Any]):
        """
        Initialize the query enhancer.
        
        Args:
            anthropic_config: Configuration containing api_key and model
        """
        self.anthropic_client = AsyncAnthropicClient(
            api_key=anthropic_config.get("api_key"),
            model=anthropic_config.get("model", "claude-3-sonnet-20240229")
        )
        
    async def generate_query_variations(self, 
                                      user_query: str, 
                                      system_prompt: str, 
                                      num_variations: int) -> List[str]:
        """
        Generate semantic variations of the user query.
        
        Args:
            user_query: Original user query
            system_prompt: System prompt for LLM
            num_variations: Number of variations to generate
            
        Returns:
            List of query variations
        """
        if num_variations <= 0:
            return [user_query]
        
        try:
            # Enhance the system prompt with specific instructions
            enhanced_system_prompt = f"""
            {system_prompt}
            
            You must return a JSON response with exactly {num_variations} search query variations.
            The response must be in the following format:
            {{"search_strings": ["variation1", "variation2", "variation3"]}}
            
            Guidelines:
            - Create semantically diverse variations of the original query
            - Maintain the original intent and meaning
            - Use different phrasings, synonyms, and related terms
            - Each variation should be a complete, searchable query
            - Return exactly {num_variations} variations
            """
            
            messages = [{"role": "user", "content": user_query}]
            
            response = await self.anthropic_client.chat(enhanced_system_prompt, messages)
            
            # Parse the JSON response
            parsed_response = json.loads(response.strip())
            variations = parsed_response.get("search_strings", [])
            
            # Ensure we have the right number of variations
            if len(variations) < num_variations:
                # Pad with the original query if needed
                variations.extend([user_query] * (num_variations - len(variations)))
            elif len(variations) > num_variations:
                # Trim to the requested number
                variations = variations[:num_variations]
            
            logger.info(f"Generated {len(variations)} query variations for: {user_query}")
            return variations
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            # Fallback to original query
            return [user_query] * num_variations
            
        except Exception as e:
            logger.error(f"Error generating query variations: {e}")
            # Fallback to original query
            return [user_query] * num_variations
    
    async def generate_batch_variations(self, 
                                       queries: List[str], 
                                       system_prompt: str, 
                                       num_variations: int,
                                       max_concurrent: int = 3) -> Dict[str, List[str]]:
        """
        Generate variations for multiple queries concurrently.
        
        Args:
            queries: List of user queries
            system_prompt: System prompt for LLM
            num_variations: Number of variations per query
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Dictionary mapping original queries to their variations
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_query(query: str) -> tuple[str, List[str]]:
            async with semaphore:
                variations = await self.generate_query_variations(
                    query, system_prompt, num_variations
                )
                return query, variations
        
        tasks = [process_single_query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        variations_map = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Query variation generation failed: {result}")
            else:
                query, variations = result
                variations_map[query] = variations
        
        return variations_map
    
    async def enhance_query_with_context(self, 
                                       user_query: str, 
                                       context: str, 
                                       num_variations: int = 3) -> List[str]:
        """
        Generate query variations using additional context.
        
        Args:
            user_query: Original user query
            context: Additional context to inform variations
            num_variations: Number of variations to generate
            
        Returns:
            List of enhanced query variations
        """
        system_prompt = f"""
        Given the following context information, generate {num_variations} semantic variations 
        of the user query that would be effective for document search.
        
        Context: {context}
        
        Create variations that:
        1. Maintain the original search intent
        2. Incorporate relevant context terms
        3. Use different phrasings and synonyms
        4. Are optimized for semantic search
        
        Return as JSON: {{"search_strings": ["variation1", "variation2", "variation3"]}}
        """
        
        return await self.generate_query_variations(user_query, system_prompt, num_variations)
    
    def create_domain_specific_prompt(self, domain: str) -> str:
        """
        Create a domain-specific system prompt for query enhancement.
        
        Args:
            domain: The domain/field (e.g., 'medical', 'legal', 'technical')
            
        Returns:
            Domain-specific system prompt
        """
        domain_prompts = {
            "medical": """
            Generate query variations optimized for medical document search.
            Include medical terminology, synonyms, and related clinical terms.
            Consider different ways medical professionals might phrase the same concept.
            """,
            "legal": """
            Generate query variations optimized for legal document search.
            Include legal terminology, statute references, and case law concepts.
            Consider different ways legal professionals might phrase the same issue.
            """,
            "technical": """
            Generate query variations optimized for technical documentation search.
            Include technical terms, API names, programming concepts, and implementation details.
            Consider different ways developers might phrase the same technical question.
            """,
            "academic": """
            Generate query variations optimized for academic paper search.
            Include scholarly terminology, research concepts, and field-specific jargon.
            Consider different ways researchers might phrase the same research question.
            """,
            "business": """
            Generate query variations optimized for business document search.
            Include business terminology, industry concepts, and corporate language.
            Consider different ways business professionals might phrase the same concept.
            """
        }
        
        base_prompt = """
        Generate semantic variations of the user query for comprehensive document search.
        Focus on different phrasings, synonyms, and related terms while maintaining the original intent.
        """
        
        domain_specific = domain_prompts.get(domain.lower(), "")
        
        return f"{base_prompt}\n\n{domain_specific}".strip()


# Advanced query enhancement strategies
class AdvancedQueryEnhancer(QueryEnhancer):
    """Extended query enhancer with advanced strategies."""
    
    async def generate_hierarchical_variations(self, 
                                             user_query: str, 
                                             num_variations: int = 5) -> Dict[str, List[str]]:
        """
        Generate variations at different specificity levels.
        
        Args:
            user_query: Original user query
            num_variations: Number of variations per level
            
        Returns:
            Dictionary with variations by specificity level
        """
        levels = {
            "broader": "Generate broader, more general variations of the query",
            "same": "Generate variations at the same specificity level",
            "narrower": "Generate more specific, detailed variations of the query"
        }
        
        results = {}
        
        for level, instruction in levels.items():
            system_prompt = f"""
            {instruction}
            
            Return exactly {num_variations} variations as JSON:
            {{"search_strings": ["var1", "var2", "var3"]}}
            """
            
            variations = await self.generate_query_variations(
                user_query, system_prompt, num_variations
            )
            results[level] = variations
        
        return results
    
    async def generate_perspective_variations(self, 
                                           user_query: str, 
                                           perspectives: List[str],
                                           num_variations: int = 3) -> Dict[str, List[str]]:
        """
        Generate variations from different user perspectives.
        
        Args:
            user_query: Original user query
            perspectives: List of perspectives (e.g., ['beginner', 'expert', 'practitioner'])
            num_variations: Number of variations per perspective
            
        Returns:
            Dictionary mapping perspectives to their query variations
        """
        results = {}
        
        for perspective in perspectives:
            system_prompt = f"""
            Rephrase the query from the perspective of a {perspective}.
            Consider how a {perspective} would phrase this question or search request.
            
            Return exactly {num_variations} variations as JSON:
            {{"search_strings": ["var1", "var2", "var3"]}}
            """
            
            variations = await self.generate_query_variations(
                user_query, system_prompt, num_variations
            )
            results[perspective] = variations
        
        return results


# Example usage and testing
async def main():
    """Example usage of the query enhancer."""
    
    config = {
        "api_key": "your_anthropic_api_key",
        "model": "claude-3-sonnet-20240229"
    }
    
    enhancer = QueryEnhancer(config)
    
    # Test basic query enhancement
    user_query = "machine learning benefits"
    system_prompt = "Generate semantic variations for document search"
    
    variations = await enhancer.generate_query_variations(
        user_query, system_prompt, 3
    )
    
    print("=== Basic Query Enhancement ===")
    print(f"Original: {user_query}")
    print(f"Variations: {variations}")
    
    # Test batch processing
    queries = [
        "artificial intelligence applications",
        "data science tools",
        "cloud computing benefits"
    ]
    
    batch_results = await enhancer.generate_batch_variations(
        queries, system_prompt, 2
    )
    
    print("\n=== Batch Query Enhancement ===")
    for query, variations in batch_results.items():
        print(f"Original: {query}")
        print(f"Variations: {variations}")
        print()
    
    # Test domain-specific enhancement
    domain_prompt = enhancer.create_domain_specific_prompt("technical")
    tech_variations = await enhancer.generate_query_variations(
        "API authentication methods", domain_prompt, 3
    )
    
    print("=== Domain-Specific Enhancement ===")
    print(f"Technical variations: {tech_variations}")


if __name__ == "__main__":
    asyncio.run(main())
