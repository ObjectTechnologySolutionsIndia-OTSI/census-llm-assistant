from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union
from contextlib import asynccontextmanager
import asyncio
import logging
from dataclasses import asdict
import uvicorn
import os  
import json
import uuid
from datetime import datetime
import aiofiles
from dotenv import load_dotenv
from utils.retrive_semantic_bm25_rerank.main_orchestrator_v1 import SemanticSearchPipeline

# Import embedding processing classes
from utils.vector_indexing.indexing_ingestion_batch_async import AsyncEmbeddingProcessor, logger
from utils.bm25_inference.BM25_indexer import BM25Indexer

load_dotenv(override=True)

# Global instances
pipeline: Optional[SemanticSearchPipeline] = None
embedding_processor: Optional[AsyncEmbeddingProcessor] = None
processing_jobs: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    global pipeline, embedding_processor
    try:
        logging.info("Initializing semantic search pipeline...")
        pipeline = SemanticSearchPipeline(
            anthropic_config=anthropic_config,
            openai_config=openai_config,
            voyage_config=voyage_config,
            db_config=db_config,
            bm25_index_path=bm25_index_path
        )
        await pipeline.initialize()
        logging.info("Pipeline initialized successfully on startup")
        
        # Initialize embedding processor
        logging.info("Initializing embedding processor...")
        embedding_processor = AsyncEmbeddingProcessor(
            voyage_model="voyage-3-large",
            voyage_input_type="document",
            batch_size=10,
            max_concurrent_operations=3
        )
        await embedding_processor.initialize()
        logging.info("Embedding processor initialized successfully")
        
    except Exception as e:
        logging.error(f"Failed to initialize services: {e}")
        raise
    
    yield  # Application runs here
    
    # Shutdown
    logging.info("Shutting down services...")
    if embedding_processor:
        await embedding_processor.cleanup()
    pipeline = None
    embedding_processor = None
    logging.info("Services shutdown complete")

app = FastAPI(
    title="Semantic Search & Embedding API", 
    version="1.0.0",
    description="API for semantic search and embedding processing",
    lifespan=lifespan
)

# Existing models for search functionality
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

# New models for embedding functionality
class DocumentItem(BaseModel):
    """Single document item for embedding processing."""
    chunk: str = Field(..., description="Text chunk to process")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    source_file: Optional[str] = Field(None, description="Source file name")
    id: Optional[str] = Field(None, description="Document ID")

class LoadEmbeddingsRequest(BaseModel):
    """Request model for loading embeddings."""
    documents: List[DocumentItem] = Field(..., description="List of documents to process")
    processing_config: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "voyage_model": "voyage-3-large",
            "voyage_input_type": "document",
            "batch_size": 10,
            "max_concurrent_operations": 3
        },
        description="Processing configuration"
    )
    create_bm25_index: bool = Field(True, description="Whether to create BM25 index")
    bm25_index_path: str = Field("./utils/bm25_inference/bm25_index.pkl", description="Path for BM25 index")

class LoadEmbeddingsResponse(BaseModel):
    """Response model for load embeddings endpoint."""
    job_id: str
    status: str
    message: str
    total_documents: int
    estimated_processing_time: Optional[float] = None

class ProcessingStatus(BaseModel):
    """Response model for processing status."""
    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    progress: Dict[str, Any]
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

class ProcessingResult(BaseModel):
    """Final processing result model."""
    success: bool
    total_records: int
    total_batches: int
    processed_batches: int
    failed_batches: int
    successful_records: int
    processing_time: float
    database_stats: Optional[Dict[str, Any]] = None
    bm25_indexed: bool = False
    error: Optional[str] = None

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

bm25_index_path = "./utils/bm25_inference/bm25_index.pkl"

# Dependencies
async def get_pipeline() -> SemanticSearchPipeline:
    """Dependency to get the initialized pipeline."""
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline

async def get_embedding_processor() -> AsyncEmbeddingProcessor:
    """Dependency to get the embedding processor."""
    global embedding_processor
    if embedding_processor is None:
        raise HTTPException(status_code=503, detail="Embedding processor not initialized")
    return embedding_processor

# Background task for processing embeddings
async def process_embeddings_background(
    job_id: str,
    documents: List[Dict[str, Any]],
    config: Dict[str, Any],
    create_bm25: bool,
    bm25_path: str
):
    """Background task to process embeddings."""
    global processing_jobs
    
    try:
        # Update job status
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["progress"]["stage"] = "embedding_processing"
        
        processor = await get_embedding_processor()
        
        # Process documents with async pipeline
        result = await processor.process_documents_async_pipeline(documents)
        
        # Update progress
        processing_jobs[job_id]["progress"].update({
            "embedding_result": result,
            "stage": "bm25_indexing" if create_bm25 else "completed"
        })
        
        bm25_success = False
        if create_bm25 and result["success"]:
            try:
                logger.info("Starting BM25 indexing process...")
                sample_chunks = [item['chunk'] for item in documents]
                
                # Initialize BM25 indexer
                indexer = BM25Indexer(bm25_path)
                await indexer.process_chunks(sample_chunks)
                
                bm25_success = True
                logger.info("BM25 indexing completed successfully!")
                
            except Exception as e:
                logger.error(f"BM25 Indexer error: {e}")
                processing_jobs[job_id]["progress"]["bm25_error"] = str(e)
        
        # Get database statistics
        db_stats = None
        if processor.db_manager:
            db_stats = await processor.db_manager.get_table_stats()
        
        # Final result
        final_result = ProcessingResult(
            success=result["success"],
            total_records=result["total_records"],
            total_batches=result["total_batches"],
            processed_batches=result["processed_batches"],
            failed_batches=result["failed_batches"],
            successful_records=result["successful_records"],
            processing_time=result.get("processing_time", 0),
            database_stats=db_stats,
            bm25_indexed=bm25_success,
            error=result.get("error")
        )
        
        # Update job completion
        processing_jobs[job_id].update({
            "status": "completed" if result["success"] else "failed",
            "completed_at": datetime.now(),
            "final_result": final_result.dict(),
            "progress": {**processing_jobs[job_id]["progress"], "stage": "completed"}
        })
        
    except Exception as e:
        logger.error(f"Background processing error for job {job_id}: {e}")
        processing_jobs[job_id].update({
            "status": "failed",
            "completed_at": datetime.now(),
            "error_message": str(e),
            "progress": {**processing_jobs[job_id]["progress"], "stage": "failed"}
        })

# EXISTING SEARCH ENDPOINTS
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

# NEW EMBEDDING ENDPOINTS
@app.post("/load-embeddings/", response_model=LoadEmbeddingsResponse)
async def load_embeddings(
    request: LoadEmbeddingsRequest,
    background_tasks: BackgroundTasks
):
    """
    Load embeddings for a list of documents.
    
    This endpoint processes documents asynchronously and returns a job ID
    that can be used to track progress.
    """
    try:
        # Validate input
        if not request.documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Convert Pydantic models to dict for processing
        documents_dict = [doc.dict() for doc in request.documents]
        
        # Initialize job tracking
        processing_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": datetime.now(),
            "progress": {
                "total_documents": len(request.documents),
                "stage": "queued"
            }
        }
        
        # Start background processing
        background_tasks.add_task(
            process_embeddings_background,
            job_id,
            documents_dict,
            request.processing_config,
            request.create_bm25_index,
            request.bm25_index_path
        )
        
        # Estimate processing time (rough calculation)
        estimated_time = len(request.documents) * 0.1  # 0.1 seconds per document estimate
        
        return LoadEmbeddingsResponse(
            job_id=job_id,
            status="queued",
            message=f"Processing started for {len(request.documents)} documents",
            total_documents=len(request.documents),
            estimated_processing_time=estimated_time
        )
        
    except Exception as e:
        logger.error(f"Error starting embedding processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")

@app.post("/load-embeddings-from-file/")
async def load_embeddings_from_file(
    file: UploadFile = File(...),
    create_bm25_index: bool = True,
    batch_size: int = 10,
    max_concurrent_operations: int = 3,
    background_tasks: BackgroundTasks = None
):
    """
    Load embeddings from a JSON file.
    
    Expects a JSON file with a list of objects containing at least a 'chunk' field.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="Only JSON files are supported")
        
        # Read file content
        content = await file.read()
        try:
            data = json.loads(content.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
        
        # Validate data structure
        if not isinstance(data, list):
            raise HTTPException(status_code=400, detail="JSON file must contain a list of objects")
        
        # Convert to DocumentItem format
        documents = []
        for i, item in enumerate(data):
            if not isinstance(item, dict) or 'chunk' not in item:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Item {i} must be an object with a 'chunk' field"
                )
            
            documents.append(DocumentItem(
                chunk=item['chunk'],
                metadata=item.get('metadata', {}),
                source_file=item.get('source_file', file.filename),
                id=item.get('id', f"{file.filename}_{i}")
            ))
        
        # Create request object
        request = LoadEmbeddingsRequest(
            documents=documents,
            processing_config={
                "voyage_model": "voyage-3-large",
                "voyage_input_type": "document",
                "batch_size": batch_size,
                "max_concurrent_operations": max_concurrent_operations
            },
            create_bm25_index=create_bm25_index,
            bm25_index_path="./utils/bm25_inference/bm25_index.pkl"
        )
        
        # Process using the main load_embeddings function
        return await load_embeddings(request, background_tasks)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file upload: {e}")
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

@app.get("/job-status/{job_id}", response_model=ProcessingStatus)
async def get_job_status(job_id: str):
    """
    Get the status of a processing job.
    
    Args:
        job_id: The job ID returned from load_embeddings endpoint
        
    Returns:
        ProcessingStatus: Current status and progress of the job
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = processing_jobs[job_id]
    
    return ProcessingStatus(
        job_id=job_id,
        status=job_data["status"],
        progress=job_data["progress"],
        created_at=job_data["created_at"],
        completed_at=job_data.get("completed_at"),
        error_message=job_data.get("error_message")
    )

@app.get("/job-result/{job_id}")
async def get_job_result(job_id: str):
    """
    Get the final result of a completed processing job.
    
    Args:
        job_id: The job ID returned from load_embeddings endpoint
        
    Returns:
        Final processing result with detailed statistics
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = processing_jobs[job_id]
    
    if job_data["status"] not in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Job not yet completed")
    
    return {
        "job_id": job_id,
        "status": job_data["status"],
        "final_result": job_data.get("final_result"),
        "error_message": job_data.get("error_message"),
        "processing_time": (
            job_data.get("completed_at", datetime.now()) - job_data["created_at"]
        ).total_seconds() if job_data.get("completed_at") else None
    }

@app.get("/jobs/")
async def list_jobs(status: Optional[str] = None, limit: int = 50):
    """
    List processing jobs with optional status filtering.
    
    Args:
        status: Filter by job status (queued, processing, completed, failed)
        limit: Maximum number of jobs to return
        
    Returns:
        List of job summaries
    """
    jobs = list(processing_jobs.values())
    
    if status:
        jobs = [job for job in jobs if job["status"] == status]
    
    # Sort by creation time (newest first)
    jobs.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Limit results
    jobs = jobs[:limit]
    
    # Return summary information
    return [
        {
            "job_id": job["job_id"],
            "status": job["status"],
            "created_at": job["created_at"],
            "completed_at": job.get("completed_at"),
            "total_documents": job["progress"].get("total_documents", 0),
            "stage": job["progress"].get("stage", "unknown")
        }
        for job in jobs
    ]

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job from the tracking system.
    
    Args:
        job_id: The job ID to delete
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del processing_jobs[job_id]
    return {"message": f"Job {job_id} deleted successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "service": "semantic-search-embedding-api",
        "pipeline_initialized": pipeline is not None,
        "embedding_processor_initialized": embedding_processor is not None,
        "active_jobs": len([job for job in processing_jobs.values() if job["status"] in ["queued", "processing"]])
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)