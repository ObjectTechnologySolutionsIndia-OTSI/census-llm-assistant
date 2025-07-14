"""
Complete Example Usage of the Semantic Search Pipeline
=====================================================

This example demonstrates how to use the complete semantic search pipeline
with all components working together.

Author: Assistant
Date: 2025
"""

import asyncio
import os
import time
import json
from typing import List, Dict, Any

# Import our modules
from census_bot_v1.utils.retrive_semantic_bm25_rerank.main_orchestrator_old import SemanticSearchPipeline, SearchOptions, PipelineResult
from config_manager import ConfigManager, setup_logging
from search_manager import CachedSearchManager
from result_processor import AdvancedResultProcessor


async def example_basic_usage():
    """Basic usage example of the semantic search pipeline."""
    
    print("=== Basic Usage Example ===")
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Setup logging
    setup_logging(config.logging)
    
    # Initialize pipeline
    pipeline = SemanticSearchPipeline(
        anthropic_config={
            "api_key": config.anthropic.api_key,
            "model": config.anthropic.model
        },
        voyage_config={
            "api_key": config.voyage.api_key,
            "embedding_model": config.voyage.embedding_model,
            "rerank_model": config.voyage.rerank_model
        },
        db_config={
            "host": config.database.host,
            "port": config.database.port,
            "database": config.database.database,
            "user": config.database.user,
            "password": config.database.password
        },
        bm25_index_path=config.search.bm25_index_path
    )
    
    try:
        # Initialize the pipeline
        await pipeline.initialize()
        
        # Example 1: Query with enhancement
        options_enhanced = SearchOptions(
            enhance_query=True,
            num_variations=3,
            system_prompt="Generate semantic variations of the user query for comprehensive document search. Focus on different phrasings and related terms that would help find relevant documents."
        )
        
        result = await pipeline.process_query(
            user_query="What are the benefits of machine learning in healthcare?",
            options=options_enhanced,
            top_k_semantic=config.search.semantic_top_k,
            top_k_bm25=config.search.bm25_top_k,
            top_k_final=config.search.final_top_k
        )
        
        print(f"Enhanced Query Results:")
        print(f"Original query: {result.query}")
        print(f"Enhanced queries: {result.enhanced_queries}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Final results: {len(result.reranked_results)}")
        
        # Display top results
        for i, search_result in enumerate(result.reranked_results[:3], 1):
            print(f"\nResult {i}:")
            print(f"Score: {search_result.score:.4f}")
            print(f"Source: {search_result.source}")
            print(f"Chunk: {search_result.chunk[:200]}...")
        
        # Example 2: Simple query without enhancement
        options_simple = SearchOptions(
            enhance_query=False,
            num_variations=0,
            system_prompt=""
        )
        
        result_simple = await pipeline.process_query(
            user_query="Python programming tutorial",
            options=options_simple
        )
        
        print(f"\nSimple Query Results:")
        print(f"Query: {result_simple.query}")
        print(f"Processing time: {result_simple.processing_time:.2f}s")
        print(f"Final results: {len(result_simple.reranked_results)}")
        
    finally:
        await pipeline.close()


async def example_batch_processing():
    """Example of batch processing multiple queries."""
    
    print("\n=== Batch Processing Example ===")
    
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    pipeline = SemanticSearchPipeline(
        anthropic_config={
            "api_key": config.anthropic.api_key,
            "model": config.anthropic.model
        },
        voyage_config={
            "api_key": config.voyage.api_key,
            "embedding_model": config.voyage.embedding_model,
            "rerank_model": config.voyage.rerank_model
        },
        db_config={
            "host": config.database.host,
            "port": config.database.port,
            "database": config.database.database,
            "user": config.database.user,
            "password": config.database.password
        }
    )
    
    try:
        await pipeline.initialize()
        
        # Prepare batch queries
        enhanced_options = SearchOptions(
            enhance_query=True,
            num_variations=2,
            system_prompt="Create search variations for technical documentation"
        )
        
        simple_options = SearchOptions(
            enhance_query=False,
            num_variations=0,
            system_prompt=""
        )
        
        batch_queries = [
            ("artificial intelligence applications", enhanced_options),
            ("data science best practices", simple_options),
            ("cloud computing security", enhanced_options),
            ("database optimization techniques", simple_options),
            ("machine learning algorithms comparison", enhanced_options)
        ]
        
        # Process batch
        start_time = time.time()
        batch_results = await pipeline.process_batch_queries(
            batch_queries, max_concurrent=3
        )
        total_time = time.time() - start_time
        
        print(f"Processed {len(batch_results)} queries in {total_time:.2f}s")
        print(f"Average time per query: {total_time/len(batch_results):.2f}s")
        
        # Display summary
        for i, result in enumerate(batch_results):
            query, _ = batch_queries[i]
            print(f"\nQuery {i+1}: {query}")
            print(f"  Processing time: {result.processing_time:.2f}s")
            print(f"  Results found: {len(result.reranked_results)}")
            print(f"  Step timings: {result.step_timings}")
    
    finally:
        await pipeline.close()


async def example_advanced_features():
    """Example demonstrating advanced features and customization."""
    
    print("\n=== Advanced Features Example ===")
    
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Use cached search manager for better performance
    cached_search_manager = CachedSearchManager(
        voyage_config={
            "api_key": config.voyage.api_key,
            "embedding_model": config.voyage.embedding_model
        },
        db_config={
            "host": config.database.host,
            "port": config.database.port,
            "database": config.database.database,
            "user": config.database.user,
            "password": config.database.password
        },
        bm25_index_path=config.search.bm25_index_path,
        cache_ttl=1800,  # 30 minutes
        max_cache_size=500
    )
    
    # Use advanced result processor
    advanced_processor = AdvancedResultProcessor(
        voyage_config={
            "api_key": config.voyage.api_key,
            "rerank_model": config.voyage.rerank_model
        },
        enable_ml_features=config.processing.enable_ml_features
    )
    
    try:
        await cached_search_manager.initialize()
        
        # Test different fusion strategies
        queries = [
            "neural network architectures",
            "deep learning frameworks",
            "computer vision applications"
        ]
        
        print("Testing different fusion strategies:")
        
        # Test each fusion strategy
        fusion_strategies = ["combine", "rrf", "weighted", "combsum"]
        
        for strategy in fusion_strategies:
            print(f"\n--- Fusion Strategy: {strategy} ---")
            
            start_time = time.time()
            
            # Get search results
            semantic_results, bm25_results = await cached_search_manager.parallel_search(
                queries, top_k_semantic=50, top_k_bm25=50
            )
            
            # Apply fusion strategy
            if strategy == "combine":
                combined_results = await advanced_processor.combine_and_deduplicate(
                    semantic_results, bm25_results
                )
            else:
                combined_results = await advanced_processor.adaptive_score_fusion(
                    semantic_results, bm25_results, fusion_strategy=strategy
                )
            
            # Rerank results
            reranked_results = await advanced_processor.rerank_results(
                "neural networks and deep learning", combined_results, top_k=10
            )
            
            processing_time = time.time() - start_time
            
            print(f"  Results: {len(reranked_results)}")
            print(f"  Processing time: {processing_time:.2f}s")
            
            # Show top result
            if reranked_results:
                top_result = reranked_results[0]
                print(f"  Top result score: {top_result.score:.4f}")
                print(f"  Top result: {top_result.chunk[:100]}...")
        
        # Show cache statistics
        cache_stats = cached_search_manager.get_cache_stats()
        print(f"\nCache Statistics: {cache_stats}")
        
        # Show search statistics
        search_stats = await cached_search_manager.get_search_statistics()
        print(f"Search Statistics: {search_stats}")
        
        # Health check
        health = await cached_search_manager.health_check()
        print(f"Health Check: {health}")
    
    finally:
        await cached_search_manager.close()


async def example_error_handling():
    """Example demonstrating error handling and fallback mechanisms."""
    
    print("\n=== Error Handling Example ===")
    
    # Create configuration with intentional issues for testing
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Test with invalid API keys (for demonstration)
    invalid_config = {
        "api_key": "invalid_key_for_testing",
        "embedding_model": config.voyage.embedding_model
    }
    
    pipeline = SemanticSearchPipeline(
        anthropic_config={
            "api_key": "invalid_anthropic_key",
            "model": config.anthropic.model
        },
        voyage_config=invalid_config,
        db_config={
            "host": "invalid_host",
            "port": config.database.port,
            "database": config.database.database,
            "user": config.database.user,
            "password": config.database.password
        }
    )
    
    try:
        # Try to initialize (this should handle errors gracefully)
        await pipeline.initialize()
        
        # Try to process a query (should use fallback mechanisms)
        options = SearchOptions(
            enhance_query=True,
            num_variations=2,
            system_prompt="Generate query variations"
        )
        
        result = await pipeline.process_query(
            "test query for error handling",
            options
        )
        
        print("Pipeline handled errors gracefully")
        print(f"Results returned: {len(result.reranked_results)}")
        
    except Exception as e:
        print(f"Expected error occurred: {e}")
        print("Error handling worked as expected")
    
    finally:
        try:
            await pipeline.close()
        except:
            pass


async def example_performance_monitoring():
    """Example of performance monitoring and optimization."""
    
    print("\n=== Performance Monitoring Example ===")
    
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    pipeline = SemanticSearchPipeline(
        anthropic_config={
            "api_key": config.anthropic.api_key,
            "model": config.anthropic.model
        },
        voyage_config={
            "api_key": config.voyage.api_key,
            "embedding_model": config.voyage.embedding_model,
            "rerank_model": config.voyage.rerank_model
        },
        db_config={
            "host": config.database.host,
            "port": config.database.port,
            "database": config.database.database,
            "user": config.database.user,
            "password": config.database.password
        }
    )
    
    try:
        await pipeline.initialize()
        
        # Test queries with performance monitoring
        test_queries = [
            "machine learning algorithms",
            "data visualization techniques",
            "cloud computing platforms",
            "cybersecurity best practices",
            "software development methodologies"
        ]
        
        performance_data = []
        
        for query in test_queries:
            options = SearchOptions(
                enhance_query=True,
                num_variations=2,
                system_prompt="Generate effective search variations"
            )
            
            start_time = time.time()
            result = await pipeline.process_query(query, options)
            total_time = time.time() - start_time
            
            # Collect performance metrics
            metrics = {
                "query": query,
                "total_time": total_time,
                "processing_time": result.processing_time,
                "step_timings": result.step_timings,
                "results_count": len(result.reranked_results),
                "enhanced_queries": len(result.enhanced_queries)
            }
            
            performance_data.append(metrics)
            
            print(f"Query: {query[:30]}...")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Results: {len(result.reranked_results)}")
        
        # Analyze performance
        avg_total_time = sum(p["total_time"] for p in performance_data) / len(performance_data)
        avg_processing_time = sum(p["processing_time"] for p in performance_data) / len(performance_data)
        avg_results = sum(p["results_count"] for p in performance_data) / len(performance_data)
        
        print(f"\nPerformance Summary:")
        print(f"  Average total time: {avg_total_time:.2f}s")
        print(f"  Average processing time: {avg_processing_time:.2f}s")
        print(f"  Average results per query: {avg_results:.1f}")
        
        # Step timing analysis
        step_totals = {}
        for metrics in performance_data:
            for step, timing in metrics["step_timings"].items():
                if step not in step_totals:
                    step_totals[step] = []
                step_totals[step].append(timing)
        
        print(f"\nStep Timing Analysis:")
        for step, timings in step_totals.items():
            avg_timing = sum(timings) / len(timings)
            max_timing = max(timings)
            print(f"  {step}: avg={avg_timing:.2f}s, max={max_timing:.2f}s")
        
        # Save performance data
        with open("performance_metrics.json", "w") as f:
            json.dump(performance_data, f, indent=2)
        
        print("\nPerformance data saved to performance_metrics.json")
    
    finally:
        await pipeline.close()


async def example_custom_system_prompts():
    """Example demonstrating different system prompts for query enhancement."""
    
    print("\n=== Custom System Prompts Example ===")
    
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    pipeline = SemanticSearchPipeline(
        anthropic_config={
            "api_key": config.anthropic.api_key,
            "model": config.anthropic.model
        },
        voyage_config={
            "api_key": config.voyage.api_key,
            "embedding_model": config.voyage.embedding_model,
            "rerank_model": config.voyage.rerank_model
        },
        db_config={
            "host": config.database.host,
            "port": config.database.port,
            "database": config.database.database,
            "user": config.database.user,
            "password": config.database.password
        }
    )
    
    try:
        await pipeline.initialize()
        
        test_query = "machine learning model optimization"
        
        # Different system prompts for different use cases
        prompt_examples = {
            "technical": """
            Generate technical variations of the search query focusing on:
            - API documentation and code examples
            - Implementation details and algorithms
            - Performance metrics and benchmarks
            - Best practices and design patterns
            Return exactly 3 variations as JSON: {"search_strings": ["var1", "var2", "var3"]}
            """,
            
            "academic": """
            Generate academic variations of the search query focusing on:
            - Research papers and scholarly articles
            - Theoretical foundations and mathematical concepts
            - Experimental methodologies and results
            - Literature reviews and citations
            Return exactly 3 variations as JSON: {"search_strings": ["var1", "var2", "var3"]}
            """,
            
            "business": """
            Generate business-oriented variations of the search query focusing on:
            - ROI and cost-benefit analysis
            - Implementation strategies and roadmaps
            - Case studies and success stories
            - Market trends and competitive analysis
            Return exactly 3 variations as JSON: {"search_strings": ["var1", "var2", "var3"]}
            """,
            
            "beginner": """
            Generate beginner-friendly variations of the search query focusing on:
            - Tutorials and step-by-step guides
            - Basic concepts and fundamentals
            - Simple examples and explanations
            - Getting started resources
            Return exactly 3 variations as JSON: {"search_strings": ["var1", "var2", "var3"]}
            """
        }
        
        for prompt_type, system_prompt in prompt_examples.items():
            print(f"\n--- {prompt_type.upper()} SYSTEM PROMPT ---")
            
            options = SearchOptions(
                enhance_query=True,
                num_variations=3,
                system_prompt=system_prompt
            )
            
            result = await pipeline.process_query(test_query, options, top_k_final=5)
            
            print(f"Original query: {result.query}")
            print(f"Enhanced queries:")
            for i, enhanced in enumerate(result.enhanced_queries, 1):
                print(f"  {i}. {enhanced}")
            
            print(f"Top result: {result.reranked_results[0].chunk[:100]}..." if result.reranked_results else "No results")
    
    finally:
        await pipeline.close()


async def example_result_analysis():
    """Example demonstrating result analysis and filtering."""
    
    print("\n=== Result Analysis Example ===")
    
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Use advanced result processor for analysis features
    from result_processor import AdvancedResultProcessor
    
    processor = AdvancedResultProcessor(
        voyage_config={
            "api_key": config.voyage.api_key,
            "rerank_model": config.voyage.rerank_model
        },
        enable_ml_features=False
    )
    
    pipeline = SemanticSearchPipeline(
        anthropic_config={
            "api_key": config.anthropic.api_key,
            "model": config.anthropic.model
        },
        voyage_config={
            "api_key": config.voyage.api_key,
            "embedding_model": config.voyage.embedding_model,
            "rerank_model": config.voyage.rerank_model
        },
        db_config={
            "host": config.database.host,
            "port": config.database.port,
            "database": config.database.database,
            "user": config.database.user,
            "password": config.database.password
        }
    )
    
    try:
        await pipeline.initialize()
        
        # Process a complex query
        options = SearchOptions(
            enhance_query=True,
            num_variations=3,
            system_prompt="Generate comprehensive search variations for technical documentation"
        )
        
        result = await pipeline.process_query(
            "deep learning neural network architectures comparison",
            options,
            top_k_final=50  # Get more results for analysis
        )
        
        # Analyze result distribution
        analysis = processor.analyze_result_distribution(result.reranked_results)
        print("Result Distribution Analysis:")
        print(f"  Total results: {analysis['total_results']}")
        print(f"  Source distribution: {analysis['source_distribution']}")
        print(f"  Score statistics: {analysis['score_statistics']}")
        print(f"  Average scores by source: {analysis['average_scores_by_source']}")
        
        # Filter results by different criteria
        print("\nFiltering Examples:")
        
        # High-quality results only
        high_quality = processor.filter_results_by_criteria(
            result.reranked_results,
            min_score=0.7,
            max_results=10
        )
        print(f"  High-quality results (score >= 0.7): {len(high_quality)}")
        
        # Results from specific sources
        semantic_only = processor.filter_results_by_criteria(
            result.reranked_results,
            required_sources=["semantic"],
            max_results=15
        )
        print(f"  Semantic-only results: {len(semantic_only)}")
        
        # Combined source results
        combined_only = processor.filter_results_by_criteria(
            result.reranked_results,
            required_sources=["combined"],
            max_results=10
        )
        print(f"  Combined-source results: {len(combined_only)}")
        
        # Show detailed analysis of top results
        print(f"\nDetailed Analysis of Top 5 Results:")
        for i, search_result in enumerate(result.reranked_results[:5], 1):
            print(f"\nResult {i}:")
            print(f"  Score: {search_result.score:.4f}")
            print(f"  Source: {search_result.source}")
            print(f"  Metadata: {search_result.metadata}")
            print(f"  Content preview: {search_result.chunk[:150]}...")
    
    finally:
        await pipeline.close()


async def example_configuration_management():
    """Example demonstrating configuration management features."""
    
    print("\n=== Configuration Management Example ===")
    
    # Load configuration with validation
    config_manager = ConfigManager()
    
    try:
        config = config_manager.load_config()
        
        # Display configuration summary
        summary = config_manager.get_config_summary(config)
        print("Configuration Summary:")
        print(summary)
        
        # Validate configuration
        from config_manager import ConfigValidator
        validator = ConfigValidator()
        
        # Check file paths
        file_validation = validator.validate_file_paths(config)
        print(f"\nFile Path Validation: {file_validation}")
        
        # Save configuration for reference
        config_manager.save_config(config, "current_config.yaml", format="yaml")
        print("Current configuration saved to current_config.yaml")
        
        # Create different environment configurations
        from config_manager import ConfigTemplates
        
        environments = ["development", "production", "testing"]
        for env in environments:
            if env == "development":
                template = ConfigTemplates.development_config()
            elif env == "production":
                template = ConfigTemplates.production_config()
            else:
                template = ConfigTemplates.testing_config()
            
            filename = f"config_{env}.yaml"
            with open(filename, 'w') as f:
                import yaml
                yaml.dump(template, f, default_flow_style=False, indent=2)
            
            print(f"Template configuration saved to {filename}")
    
    except Exception as e:
        print(f"Configuration error: {e}")


def create_sample_config_file():
    """Create a sample configuration file for users."""
    
    sample_config = {
        "anthropic": {
            "api_key": "your_anthropic_api_key_here",
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1000,
            "timeout": 30
        },
        "voyage": {
            "api_key": "your_voyage_api_key_here",
            "embedding_model": "voyage-3-large",
            "rerank_model": "rerank-2",
            "input_type": None,
            "truncation": True,
            "timeout": 30
        },
        "database": {
            "host": "localhost",
            "port": "5432",
            "database": "your_database_name",
            "user": "your_username",
            "password": "your_password",
            "min_pool_size": 5,
            "max_pool_size": 20,
            "ssl_mode": "prefer"
        },
        "search": {
            "bm25_index_path": "bm25_index.pkl",
            "semantic_top_k": 100,
            "bm25_top_k": 100,
            "final_top_k": 20,
            "similarity_threshold": 0.85,
            "score_normalization": "minmax",
            "max_concurrent_searches": 5,
            "enable_caching": True,
            "cache_ttl": 3600,
            "max_cache_size": 1000
        },
        "processing": {
            "deduplication_method": "advanced",
            "fusion_strategy": "combine",
            "semantic_weight": 0.7,
            "bm25_weight": 0.3,
            "diversity_boost_factor": 0.1,
            "enable_ml_features": False,
            "max_concurrent_reranking": 3
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            "file_path": "logs/pipeline.log",
            "max_file_size": 10485760,
            "backup_count": 5
        }
    }
    
    import yaml
    with open("config_sample.yaml", 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)
    
    print("Sample configuration file created: config_sample.yaml")
    print("Please update with your actual API keys and database credentials.")


async def main():
    """Run all examples."""
    
    print("Semantic Search Pipeline - Complete Examples")
    print("=" * 50)
    
    # Create sample configuration file
    create_sample_config_file()
    
    # Check if configuration is available
    config_manager = ConfigManager()
    try:
        config = config_manager.load_config()
        
        # Only run examples if we have valid configuration
        if config.anthropic.api_key and config.voyage.api_key:
            await example_basic_usage()
            await example_batch_processing()
            await example_advanced_features()
            await example_custom_system_prompts()
            await example_result_analysis()
            await example_performance_monitoring()
        else:
            print("\nSkipping pipeline examples - API keys not configured")
            print("Please update config_sample.yaml with your API keys")
        
        # Configuration examples can run without API keys
        await example_configuration_management()
        
        # Error handling example (will demonstrate graceful error handling)
        await example_error_handling()
        
    except Exception as e:
        print(f"Error in examples: {e}")
        print("Please check your configuration and ensure all dependencies are installed")
    
    print("\nExamples completed!")


if __name__ == "__main__":
    asyncio.run(main())