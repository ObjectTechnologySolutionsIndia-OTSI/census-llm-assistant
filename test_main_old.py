from utils.retrive_semantic_bm25_rerank.main_orchestrator_old import (
    logger,
    SearchOptions,
    SemanticSearchPipeline,
)
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()



async def main():
    """Example usage of the semantic search pipeline."""

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
    pipeline = SemanticSearchPipeline(
        anthropic_config=anthropic_config,
        voyage_config=voyage_config,
        db_config=db_config,
        bm25_index_path="./utils/bm25_inference/bm25_index.pkl",
    )

    try:
        await pipeline.initialize()

        # Example 1: Single query with enhancement
        options1 = SearchOptions(
            enhance_query=True,
            num_variations=3,
            system_prompt="Generate semantic variations of the user query for comprehensive document search. Focus on different phrasings and related terms.",
        )

        result1 = await pipeline.process_query(
            "What are the benefits of machine learning?", options1
        )

        print("=== Enhanced Query Search ===")
        print(f"Original query: {result1.query}")
        print(f"Enhanced queries: {result1.enhanced_queries}")
        print(f"Final results count: {len(result1.reranked_results)}")
        print(f"Processing time: {result1.processing_time:.2f}s")
        print("Step timings:", result1.step_timings)
        print(f"results: {result1.combined_results}")

        # # Example 2: Simple query without enhancement
        # options2 = SearchOptions(
        #     enhance_query=False, num_variations=0, system_prompt=""
        # )

        # result2 = await pipeline.process_query("Python programming tutorial", options2)

        # print("\n=== Simple Query Search ===")
        # print(f"Query: {result2.query}")
        # print(f"Final results count: {len(result2.reranked_results)}")
        # print(f"Processing time: {result2.processing_time:.2f}s")

        # Example 3: Batch processing
        # batch_queries = [
        #     ("artificial intelligence applications", options1),
        #     ("data science tools", options2),
        #     ("cloud computing benefits", options1),
        # ]

        # batch_results = await pipeline.process_batch_queries(batch_queries)

        # print(f"\n=== Batch Processing ===")
        # print(f"Processed {len(batch_results)} queries")
        # for i, result in enumerate(batch_results):
        #     print(f"Query {i+1}: {result.processing_time:.2f}s")

    except Exception as e:
        logger.error(f"Pipeline error: {e}")

    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
