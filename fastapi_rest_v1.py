from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager
import asyncio
import logging
from dataclasses import asdict
import uvicorn
import os  
from dotenv import load_dotenv
from utils.retrive_semantic_bm25_rerank.main_orchestrator_v1 import  SemanticSearchPipeline

load_dotenv(override=True)

# Import your existing classes (assuming they're in separate modules)
# from your_module import SearchOptions, SearchResult, PipelineResult, SemanticSearchPipeline

# Global pipeline instance
pipeline: Optional[SemanticSearchPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    global pipeline
    try:
        logging.info("Initializing semantic search pipeline...")
        pipeline = SemanticSearchPipeline(
            anthropic_config=anthropic_config,
            openai_config=openai_config,
            voyage_config=voyage_config,
            db_config=db_config,
            table_name=embeddings_table_name,
            bm25_index_path=bm25_index_path
        )
        await pipeline.initialize()
        logging.info("Pipeline initialized successfully on startup")
    except Exception as e:
        logging.error(f"Failed to initialize pipeline: {e}")
        raise
    
    yield  # Application runs here
    
    # Shutdown
    logging.info("Shutting down semantic search pipeline...")
    # Add any cleanup logic here if needed
    pipeline = None
    logging.info("Pipeline shutdown complete")

app = FastAPI(
    title="Semantic Search API", 
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response models
class SearchOptions(BaseModel):
    """Configuration options for the search pipeline."""
    
    enhance_query: bool = Field(..., description="Whether to enhance user query for optimal semantic search")
    num_variations: int = Field(0, description="Number of variations of user query for semantic search")
    system_prompt: str = Field("", description="System prompt for generating user query variations")
    
    @validator('num_variations')
    def validate_num_variations(cls, v, values):
        if values.get('enhance_query', False) and v < 1:
            raise ValueError("num_variations cannot be less than 1 if enhance_query is True")
        return v
    
    @validator('system_prompt')
    def validate_system_prompt(cls, v, values):
        if values.get('enhance_query', False) and not v.strip():
            raise ValueError("system_prompt cannot be empty if enhance_query is True")
        return v

class ProcessQueryRequest(BaseModel):
    """Request model for process_query endpoint."""
    user_query: str = Field(..., description="User's search query")
    options: SearchOptions = Field(..., description="Search configuration options")
    top_k_semantic: int = Field(100, description="Number of top semantic results to retrieve")
    top_k_bm25: int = Field(100, description="Number of top BM25 results to retrieve")
    top_k_final: int = Field(20, description="Number of final results after combination")
    max_results: int = Field(1, description="Maximum number of results to return")
    
class SearchResultResponse(BaseModel):
    """Response model for search results."""
    id: str
    chunk: str
    score: float
    source: str
    metadata: Dict[str, Any] = {}
    source_file: Optional[str] = None

class PipelineResultResponse(BaseModel):
    """Response model for pipeline results."""
    query: str
    enhanced_queries: List[str]
    semantic_results: List[SearchResultResponse]
    bm25_results: List[SearchResultResponse]
    combined_results: List[SearchResultResponse]
    reranked_results: List[SearchResultResponse]
    processing_time: float
    step_timings: Dict[str, float]
    context_for_chat: str = ""

# Configuration (you can move these to environment variables or config files)
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

embeddings_table_name = "census_data_embeddings_v1"
bm25_index_path = f"./utils/bm25_inference/{embeddings_table_name}.pkl"


async def get_pipeline() -> SemanticSearchPipeline:
    """Dependency to get the initialized pipeline."""
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline

@app.post("/process-query/", response_model=PipelineResultResponse)
async def process_query(
    request: ProcessQueryRequest,
    pipeline: SemanticSearchPipeline = Depends(get_pipeline)
):
    """
    Process a search query using the semantic search pipeline.
    
    Args:
        request: The process query request containing query and options
        pipeline: The initialized semantic search pipeline
        
    Returns:
        PipelineResultResponse: Complete pipeline results with timing information
    """
    try:
        # Validate options
        is_valid = await pipeline.validate_options(request.options)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Invalid search options")
        
        # Process the query
        result = await pipeline.process_query(
            user_query=request.user_query,
            options=request.options,
            top_k_semantic=request.top_k_semantic,
            top_k_bm25=request.top_k_bm25,
            top_k_final=request.top_k_final,
            max_results=request.max_results
        )
        
        # Convert dataclass to response model
        def convert_search_result(sr):
            return SearchResultResponse(
                id=sr.id,
                chunk=sr.chunk,
                score=sr.score,
                source=sr.source,
                metadata=sr.metadata,
                source_file=sr.source_file
            )
        
        response = PipelineResultResponse(
            query=result.query,
            enhanced_queries=result.enhanced_queries,
            semantic_results=[convert_search_result(sr) for sr in result.semantic_results],
            bm25_results=[convert_search_result(sr) for sr in result.bm25_results],
            combined_results=[convert_search_result(sr) for sr in result.combined_results],
            reranked_results=[convert_search_result(sr) for sr in result.reranked_results],
            processing_time=result.processing_time,
            step_timings=result.step_timings,
            context_for_chat=result.context_for_chat
        )
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/simple-search/")
async def simple_search(
    query: str,
    enhance_query: bool = False,
    max_results: int = 3,
    pipeline: SemanticSearchPipeline = Depends(get_pipeline)
):
    """
    Simplified search endpoint with minimal configuration.
    
    Args:
        query: Search query string
        enhance_query: Whether to enhance the query
        max_results: Maximum number of results
        pipeline: The initialized semantic search pipeline
    """
    try:
        # Create default options
        options = SearchOptions(
            enhance_query=enhance_query,
            num_variations=3 if enhance_query else 0,
            system_prompt="Generate variations of the user query for better search results." if enhance_query else ""
        )
        
        # Process query
        result = await pipeline.process_query(
            user_query=query,
            options=options,
            max_results=max_results
        )
        
        # Return simplified response
        return {
            "query": result.query,
            "results": [
                {
                    "chunk": sr.chunk,
                    "score": sr.score,
                    "source": sr.source
                }
                for sr in result.reranked_results[:max_results]
            ],
            "processing_time": result.processing_time,
            "context_for_chat": result.context_for_chat
        }
        
    except Exception as e:
        logging.error(f"Error in simple search: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "semantic-search-api"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)