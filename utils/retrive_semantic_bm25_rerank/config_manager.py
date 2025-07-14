"""
Configuration Management Module
==============================

This module handles configuration management for the semantic search pipeline,
including environment variables, validation, and default settings.

Author: Assistant
Date: 2025
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
import yaml

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)


@dataclass
class AnthropicConfig:
    """Configuration for Anthropic API."""
    api_key: str
    model: str = "claude-3-sonnet-20240229"
    max_tokens: int = 1000
    timeout: int = 30


@dataclass
class VoyageConfig:
    """Configuration for Voyage AI services."""
    api_key: str
    embedding_model: str = "voyage-3-large"
    rerank_model: str = "rerank-2"
    input_type: Optional[str] = None
    truncation: bool = True
    timeout: int = 30


@dataclass
class DatabaseConfig:
    """Configuration for PostgreSQL database."""
    host: str
    port: str = "5432"
    database: str
    user: str
    password: str
    min_pool_size: int = 5
    max_pool_size: int = 20
    ssl_mode: str = "prefer"


@dataclass
class SearchConfig:
    """Configuration for search operations."""
    bm25_index_path: str = "bm25_index.pkl"
    semantic_top_k: int = 100
    bm25_top_k: int = 100
    final_top_k: int = 20
    similarity_threshold: float = 0.85
    score_normalization: str = "minmax"
    max_concurrent_searches: int = 5
    enable_caching: bool = True
    cache_ttl: int = 3600
    max_cache_size: int = 1000


@dataclass
class ProcessingConfig:
    """Configuration for result processing."""
    deduplication_method: str = "advanced"  # 'simple', 'advanced', 'clustering'
    fusion_strategy: str = "combine"  # 'combine', 'rrf', 'weighted', 'combsum'
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3
    diversity_boost_factor: float = 0.1
    enable_ml_features: bool = False
    max_concurrent_reranking: int = 3


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5


@dataclass
class PipelineConfig:
    """Main configuration container for the entire pipeline."""
    anthropic: AnthropicConfig
    voyage: VoyageConfig
    database: DatabaseConfig
    search: SearchConfig = field(default_factory=SearchConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


class ConfigManager:
    """Manages configuration loading, validation, and defaults."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file (JSON or YAML)
        """
        self.config_file = config_file
        self._config: Optional[PipelineConfig] = None
    
    def load_config(self) -> PipelineConfig:
        """
        Load configuration from multiple sources with priority:
        1. Configuration file (if provided)
        2. Environment variables
        3. Default values
        
        Returns:
            Complete pipeline configuration
        """
        if self._config is not None:
            return self._config
        
        # Start with defaults
        config_dict = self._get_default_config()
        
        # Override with config file if provided
        if self.config_file and os.path.exists(self.config_file):
            file_config = self._load_config_file(self.config_file)
            config_dict = self._deep_merge(config_dict, file_config)
        
        # Override with environment variables
        env_config = self._load_env_config()
        config_dict = self._deep_merge(config_dict, env_config)
        
        # Validate and create config objects
        self._config = self._create_config_objects(config_dict)
        
        # Validate configuration
        self._validate_config(self._config)
        
        logger.info("Configuration loaded successfully")
        return self._config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "anthropic": {
                "api_key": "",
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 1000,
                "timeout": 30
            },
            "voyage": {
                "api_key": "",
                "embedding_model": "voyage-3-large",
                "rerank_model": "rerank-2",
                "input_type": None,
                "truncation": True,
                "timeout": 30
            },
            "database": {
                "host": "localhost",
                "port": "5432",
                "database": "",
                "user": "",
                "password": "",
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
                "file_path": None,
                "max_file_size": 10485760,
                "backup_count": 5
            }
        }
    
    def _load_config_file(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from file (JSON or YAML)."""
        try:
            file_path = Path(config_file)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {file_path.suffix}")
            
            logger.info(f"Loaded configuration from {config_file}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config file {config_file}: {e}")
            return {}
    
    def _load_env_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {
            "anthropic": {},
            "voyage": {},
            "database": {},
            "search": {},
            "processing": {},
            "logging": {}
        }
        
        # Anthropic configuration
        if os.getenv("ANTHROPIC_API_KEY"):
            env_config["anthropic"]["api_key"] = os.getenv("ANTHROPIC_API_KEY")
        if os.getenv("ANTHROPIC_MODEL"):
            env_config["anthropic"]["model"] = os.getenv("ANTHROPIC_MODEL")
        
        # Voyage configuration
        if os.getenv("VOYAGE_API_KEY"):
            env_config["voyage"]["api_key"] = os.getenv("VOYAGE_API_KEY")
        if os.getenv("VOYAGE_EMBEDDING_MODEL"):
            env_config["voyage"]["embedding_model"] = os.getenv("VOYAGE_EMBEDDING_MODEL")
        if os.getenv("VOYAGE_RERANK_MODEL"):
            env_config["voyage"]["rerank_model"] = os.getenv("VOYAGE_RERANK_MODEL")
        
        # Database configuration
        db_env_mapping = {
            "host": "POSTGRES_HOST",
            "port": "POSTGRES_PORT",
            "database": "POSTGRES_DBNAME",
            "user": "POSTGRES_USERNAME",
            "password": "POSTGRES_PASSWORD"
        }
        
        for key, env_var in db_env_mapping.items():
            if os.getenv(env_var):
                env_config["database"][key] = os.getenv(env_var)
        
        # Search configuration
        if os.getenv("BM25_INDEX_PATH"):
            env_config["search"]["bm25_index_path"] = os.getenv("BM25_INDEX_PATH")
        if os.getenv("SEMANTIC_TOP_K"):
            env_config["search"]["semantic_top_k"] = int(os.getenv("SEMANTIC_TOP_K"))
        if os.getenv("BM25_TOP_K"):
            env_config["search"]["bm25_top_k"] = int(os.getenv("BM25_TOP_K"))
        if os.getenv("FINAL_TOP_K"):
            env_config["search"]["final_top_k"] = int(os.getenv("FINAL_TOP_K"))
        
        # Processing configuration
        if os.getenv("DEDUPLICATION_METHOD"):
            env_config["processing"]["deduplication_method"] = os.getenv("DEDUPLICATION_METHOD")
        if os.getenv("FUSION_STRATEGY"):
            env_config["processing"]["fusion_strategy"] = os.getenv("FUSION_STRATEGY")
        if os.getenv("ENABLE_ML_FEATURES"):
            env_config["processing"]["enable_ml_features"] = os.getenv("ENABLE_ML_FEATURES").lower() == "true"
        
        # Logging configuration
        if os.getenv("LOG_LEVEL"):
            env_config["logging"]["level"] = os.getenv("LOG_LEVEL")
        if os.getenv("LOG_FILE"):
            env_config["logging"]["file_path"] = os.getenv("LOG_FILE")
        
        return env_config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _create_config_objects(self, config_dict: Dict[str, Any]) -> PipelineConfig:
        """Create configuration objects from dictionary."""
        try:
            anthropic_config = AnthropicConfig(**config_dict["anthropic"])
            voyage_config = VoyageConfig(**config_dict["voyage"])
            database_config = DatabaseConfig(**config_dict["database"])
            search_config = SearchConfig(**config_dict["search"])
            processing_config = ProcessingConfig(**config_dict["processing"])
            logging_config = LoggingConfig(**config_dict["logging"])
            
            return PipelineConfig(
                anthropic=anthropic_config,
                voyage=voyage_config,
                database=database_config,
                search=search_config,
                processing=processing_config,
                logging=logging_config
            )
            
        except Exception as e:
            logger.error(f"Failed to create configuration objects: {e}")
            raise
    
    def _validate_config(self, config: PipelineConfig) -> None:
        """Validate configuration values."""
        errors = []
        
        # Validate Anthropic configuration
        if not config.anthropic.api_key:
            errors.append("Anthropic API key is required")
        
        # Validate Voyage configuration
        if not config.voyage.api_key:
            errors.append("Voyage API key is required")
        
        # Validate database configuration
        required_db_fields = ["host", "database", "user", "password"]
        for field in required_db_fields:
            if not getattr(config.database, field):
                errors.append(f"Database {field} is required")
        
        # Validate search configuration
        if config.search.semantic_top_k <= 0:
            errors.append("semantic_top_k must be positive")
        if config.search.bm25_top_k <= 0:
            errors.append("bm25_top_k must be positive")
        if config.search.final_top_k <= 0:
            errors.append("final_top_k must be positive")
        if not 0 <= config.search.similarity_threshold <= 1:
            errors.append("similarity_threshold must be between 0 and 1")
        
        # Validate processing configuration
        valid_dedup_methods = ["simple", "advanced", "clustering"]
        if config.processing.deduplication_method not in valid_dedup_methods:
            errors.append(f"deduplication_method must be one of {valid_dedup_methods}")
        
        valid_fusion_strategies = ["combine", "rrf", "weighted", "combsum"]
        if config.processing.fusion_strategy not in valid_fusion_strategies:
            errors.append(f"fusion_strategy must be one of {valid_fusion_strategies}")
        
        if not 0 <= config.processing.semantic_weight <= 1:
            errors.append("semantic_weight must be between 0 and 1")
        if not 0 <= config.processing.bm25_weight <= 1:
            errors.append("bm25_weight must be between 0 and 1")
        
        # Validate logging configuration
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config.logging.level.upper() not in valid_log_levels:
            errors.append(f"logging level must be one of {valid_log_levels}")
        
        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
            logger.error(error_message)
            raise ValueError(error_message)
        
        logger.info("Configuration validation passed")
    
    def save_config(self, config: PipelineConfig, file_path: str, format: str = "yaml") -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            file_path: Output file path
            format: File format ('yaml' or 'json')
        """
        try:
            config_dict = self._config_to_dict(config)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'yaml':
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif format.lower() == 'json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def _config_to_dict(self, config: PipelineConfig) -> Dict[str, Any]:
        """Convert configuration objects to dictionary."""
        return {
            "anthropic": {
                "api_key": config.anthropic.api_key,
                "model": config.anthropic.model,
                "max_tokens": config.anthropic.max_tokens,
                "timeout": config.anthropic.timeout
            },
            "voyage": {
                "api_key": config.voyage.api_key,
                "embedding_model": config.voyage.embedding_model,
                "rerank_model": config.voyage.rerank_model,
                "input_type": config.voyage.input_type,
                "truncation": config.voyage.truncation,
                "timeout": config.voyage.timeout
            },
            "database": {
                "host": config.database.host,
                "port": config.database.port,
                "database": config.database.database,
                "user": config.database.user,
                "password": config.database.password,
                "min_pool_size": config.database.min_pool_size,
                "max_pool_size": config.database.max_pool_size,
                "ssl_mode": config.database.ssl_mode
            },
            "search": {
                "bm25_index_path": config.search.bm25_index_path,
                "semantic_top_k": config.search.semantic_top_k,
                "bm25_top_k": config.search.bm25_top_k,
                "final_top_k": config.search.final_top_k,
                "similarity_threshold": config.search.similarity_threshold,
                "score_normalization": config.search.score_normalization,
                "max_concurrent_searches": config.search.max_concurrent_searches,
                "enable_caching": config.search.enable_caching,
                "cache_ttl": config.search.cache_ttl,
                "max_cache_size": config.search.max_cache_size
            },
            "processing": {
                "deduplication_method": config.processing.deduplication_method,
                "fusion_strategy": config.processing.fusion_strategy,
                "semantic_weight": config.processing.semantic_weight,
                "bm25_weight": config.processing.bm25_weight,
                "diversity_boost_factor": config.processing.diversity_boost_factor,
                "enable_ml_features": config.processing.enable_ml_features,
                "max_concurrent_reranking": config.processing.max_concurrent_reranking
            },
            "logging": {
                "level": config.logging.level,
                "format": config.logging.format,
                "file_path": config.logging.file_path,
                "max_file_size": config.logging.max_file_size,
                "backup_count": config.logging.backup_count
            }
        }
    
    def get_config_summary(self, config: PipelineConfig) -> str:
        """Get a human-readable summary of the configuration."""
        summary = []
        summary.append("=== Pipeline Configuration Summary ===")
        summary.append(f"Anthropic Model: {config.anthropic.model}")
        summary.append(f"Voyage Embedding Model: {config.voyage.embedding_model}")
        summary.append(f"Voyage Rerank Model: {config.voyage.rerank_model}")
        summary.append(f"Database: {config.database.user}@{config.database.host}:{config.database.port}/{config.database.database}")
        summary.append(f"BM25 Index: {config.search.bm25_index_path}")
        summary.append(f"Search Limits: Semantic={config.search.semantic_top_k}, BM25={config.search.bm25_top_k}, Final={config.search.final_top_k}")
        summary.append(f"Processing Strategy: {config.processing.deduplication_method} + {config.processing.fusion_strategy}")
        summary.append(f"Caching: {'Enabled' if config.search.enable_caching else 'Disabled'}")
        summary.append(f"ML Features: {'Enabled' if config.processing.enable_ml_features else 'Disabled'}")
        summary.append(f"Log Level: {config.logging.level}")
        
        return "\n".join(summary)


class ConfigValidator:
    """Advanced configuration validation utilities."""
    
    @staticmethod
    def validate_api_keys(config: PipelineConfig) -> Dict[str, bool]:
        """
        Validate API keys by making test requests.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Dictionary with validation results for each service
        """
        validation_results = {}
        
        # Test Anthropic API key (placeholder - implement actual test)
        try:
            # This would make a test request to Anthropic
            validation_results["anthropic"] = bool(config.anthropic.api_key)
        except Exception as e:
            logger.error(f"Anthropic API key validation failed: {e}")
            validation_results["anthropic"] = False
        
        # Test Voyage API key (placeholder - implement actual test)
        try:
            # This would make a test request to Voyage AI
            validation_results["voyage"] = bool(config.voyage.api_key)
        except Exception as e:
            logger.error(f"Voyage API key validation failed: {e}")
            validation_results["voyage"] = False
        
        return validation_results
    
    @staticmethod
    def validate_database_connection(config: DatabaseConfig) -> bool:
        """
        Validate database connection.
        
        Args:
            config: Database configuration
            
        Returns:
            True if connection successful
        """
        try:
            import asyncio
            import asyncpg
            
            async def test_connection():
                conn = await asyncpg.connect(
                    host=config.host,
                    port=config.port,
                    database=config.database,
                    user=config.user,
                    password=config.password
                )
                await conn.close()
                return True
            
            return asyncio.run(test_connection())
            
        except Exception as e:
            logger.error(f"Database connection validation failed: {e}")
            return False
    
    @staticmethod
    def validate_file_paths(config: PipelineConfig) -> Dict[str, bool]:
        """
        Validate file paths in configuration.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Dictionary with validation results for each file path
        """
        validation_results = {}
        
        # Check BM25 index file
        bm25_path = config.search.bm25_index_path
        validation_results["bm25_index"] = os.path.exists(bm25_path)
        
        # Check log file path (if specified)
        if config.logging.file_path:
            log_dir = os.path.dirname(config.logging.file_path)
            validation_results["log_directory"] = os.path.exists(log_dir) if log_dir else True
        else:
            validation_results["log_directory"] = True
        
        return validation_results


# Configuration templates for different environments
class ConfigTemplates:
    """Predefined configuration templates for different deployment scenarios."""
    
    @staticmethod
    def development_config() -> Dict[str, Any]:
        """Configuration template for development environment."""
        return {
            "anthropic": {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 1000,
                "timeout": 30
            },
            "voyage": {
                "embedding_model": "voyage-3-lite",  # Faster for development
                "rerank_model": "rerank-2",
                "truncation": True,
                "timeout": 30
            },
            "database": {
                "host": "localhost",
                "port": "5432",
                "min_pool_size": 2,
                "max_pool_size": 5,
                "ssl_mode": "disable"
            },
            "search": {
                "semantic_top_k": 50,
                "bm25_top_k": 50,
                "final_top_k": 10,
                "enable_caching": True,
                "cache_ttl": 1800,  # 30 minutes
                "max_concurrent_searches": 3
            },
            "processing": {
                "deduplication_method": "simple",
                "fusion_strategy": "combine",
                "enable_ml_features": False,
                "max_concurrent_reranking": 2
            },
            "logging": {
                "level": "DEBUG",
                "file_path": "logs/development.log"
            }
        }
    
    @staticmethod
    def production_config() -> Dict[str, Any]:
        """Configuration template for production environment."""
        return {
            "anthropic": {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 1000,
                "timeout": 60
            },
            "voyage": {
                "embedding_model": "voyage-3-large",  # Best quality for production
                "rerank_model": "rerank-2",
                "truncation": True,
                "timeout": 60
            },
            "database": {
                "host": "production-db-host",
                "port": "5432",
                "min_pool_size": 10,
                "max_pool_size": 50,
                "ssl_mode": "require"
            },
            "search": {
                "semantic_top_k": 100,
                "bm25_top_k": 100,
                "final_top_k": 20,
                "enable_caching": True,
                "cache_ttl": 3600,  # 1 hour
                "max_concurrent_searches": 10
            },
            "processing": {
                "deduplication_method": "advanced",
                "fusion_strategy": "rrf",
                "enable_ml_features": True,
                "max_concurrent_reranking": 5
            },
            "logging": {
                "level": "INFO",
                "file_path": "logs/production.log",
                "max_file_size": 52428800,  # 50MB
                "backup_count": 10
            }
        }
    
    @staticmethod
    def testing_config() -> Dict[str, Any]:
        """Configuration template for testing environment."""
        return {
            "anthropic": {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 500,
                "timeout": 15
            },
            "voyage": {
                "embedding_model": "voyage-3-lite",
                "rerank_model": "rerank-2",
                "truncation": True,
                "timeout": 15
            },
            "database": {
                "host": "localhost",
                "port": "5432",
                "database": "test_db",
                "min_pool_size": 1,
                "max_pool_size": 3,
                "ssl_mode": "disable"
            },
            "search": {
                "semantic_top_k": 20,
                "bm25_top_k": 20,
                "final_top_k": 5,
                "enable_caching": False,
                "max_concurrent_searches": 2
            },
            "processing": {
                "deduplication_method": "simple",
                "fusion_strategy": "combine",
                "enable_ml_features": False,
                "max_concurrent_reranking": 1
            },
            "logging": {
                "level": "WARNING",
                "file_path": None  # Log to console only
            }
        }


# Example usage and configuration setup
def setup_logging(config: LoggingConfig) -> None:
    """Setup logging based on configuration."""
    import logging.handlers
    
    # Set logging level
    numeric_level = getattr(logging, config.level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(config.format)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if config.file_path:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(config.file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            config.file_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def main():
    """Example usage of configuration management."""
    
    # Initialize config manager
    config_manager = ConfigManager()
    
    # Load configuration
    config = config_manager.load_config()
    
    # Setup logging
    setup_logging(config.logging)
    
    # Print configuration summary
    print(config_manager.get_config_summary(config))
    
    # Validate configuration
    validator = ConfigValidator()
    
    # Check file paths
    file_validation = validator.validate_file_paths(config)
    print(f"File validation: {file_validation}")
    
    # Save configuration template
    template_config = ConfigTemplates.development_config()
    with open("config_template.yaml", 'w') as f:
        yaml.dump(template_config, f, default_flow_style=False, indent=2)
    
    print("Configuration management example completed")


if __name__ == "__main__":
    main()