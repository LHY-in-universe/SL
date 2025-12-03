"""
Production Setup Example

This example demonstrates a production-ready Split Learning setup with:
- Proper error handling
- Logging configuration
- Resource monitoring
- Configuration management
- Health checks

Use this as a template for deploying Split Learning in production.
"""

import os
import logging
import asyncio
from pathlib import Path
from typing import Optional

import torch

from splitlearn_manager.server import AsyncManagedServer
from splitlearn_manager.config import ServerConfig, ModelConfig


# Configure logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[console_handler]
    )

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
        logging.info(f"Logging to file: {log_file}")


# Production server configuration
class ProductionConfig:
    """Production server configuration"""

    def __init__(self):
        # Server settings
        self.host = os.getenv("SERVER_HOST", "0.0.0.0")
        self.port = int(os.getenv("SERVER_PORT", "50051"))
        self.max_models = int(os.getenv("MAX_MODELS", "10"))

        # Model settings
        self.model_type = os.getenv("MODEL_TYPE", "gpt2")
        self.model_path = os.getenv("MODEL_PATH", "gpt2")
        self.component = os.getenv("MODEL_COMPONENT", "trunk")
        self.device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

        # Model split points
        self.split_point_1 = int(os.getenv("SPLIT_POINT_1", "2"))
        self.split_point_2 = int(os.getenv("SPLIT_POINT_2", "10"))

        # Resource limits
        self.memory_limit_mb = int(os.getenv("MEMORY_LIMIT_MB", "8192"))
        self.gpu_memory_limit_mb = int(os.getenv("GPU_MEMORY_LIMIT_MB", "8192"))

        # Monitoring
        self.enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"
        self.health_check_interval = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))

        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_file = os.getenv("LOG_FILE", None)

        # Cache directory
        self.cache_dir = os.getenv("CACHE_DIR", "./models")
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def validate(self):
        """Validate configuration"""
        errors = []

        if self.port < 1024 or self.port > 65535:
            errors.append(f"Invalid port: {self.port}")

        if self.max_models < 1:
            errors.append(f"max_models must be >= 1: {self.max_models}")

        if self.component not in ["bottom", "trunk", "top"]:
            errors.append(f"Invalid component: {self.component}")

        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")

        logging.info("Configuration validated successfully")


async def run_production_server():
    """
    Run production server with full configuration and monitoring.
    """
    # Load configuration
    config = ProductionConfig()
    config.validate()

    # Setup logging
    setup_logging(log_level=config.log_level, log_file=config.log_file)

    logger = logging.getLogger(__name__)
    logger.info("=== Production Split Learning Server ===")
    logger.info(f"Host: {config.host}")
    logger.info(f"Port: {config.port}")
    logger.info(f"Model: {config.model_type} ({config.component})")
    logger.info(f"Device: {config.device}")
    logger.info(f"Max models: {config.max_models}")
    logger.info(f"Metrics enabled: {config.enable_metrics}")

    # Create server configuration
    server_config = ServerConfig(
        host=config.host,
        port=config.port,
        max_models=config.max_models,
        health_check_interval=config.health_check_interval,
        enable_metrics=config.enable_metrics
    )

    # Create server
    server = AsyncManagedServer(config=server_config)

    try:
        # Start server
        logger.info("Starting server...")
        await server.start()
        logger.info("✓ Server started successfully")

        # Load model
        logger.info(f"Loading model: {config.model_type}...")
        model_config = ModelConfig(
            model_id=f"{config.model_type}_{config.component}",
            model_type=config.model_type,
            component=config.component,
            model_name_or_path=config.model_path,
            device=config.device,
            start_layer=config.split_point_1 if config.component == "trunk" else None,
            end_layer=config.split_point_2 if config.component == "trunk" else None,
        )

        await server.load_model(model_config)
        logger.info("✓ Model loaded successfully")

        # Log server status
        status = await server.get_status()
        logger.info("Server status:")
        logger.info(f"  Running: {status['running']}")
        logger.info(f"  Loaded models: {len(status['models'])}")
        logger.info(f"  Health: {status['health']['status']}")

        # Wait for termination
        logger.info("Server is running. Press Ctrl+C to stop.")
        logger.info("-" * 60)
        await server.wait_for_termination()

    except KeyboardInterrupt:
        logger.info("Server interrupted by user")

    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise

    finally:
        # Graceful shutdown
        logger.info("Shutting down server...")
        await server.stop(grace=10.0)
        logger.info("✓ Server stopped")


def main():
    """Main entry point"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║       Split Learning Production Server                      ║
║                                                              ║
║  Configuration via environment variables:                   ║
║    SERVER_HOST        - Server host (default: 0.0.0.0)      ║
║    SERVER_PORT        - Server port (default: 50051)        ║
║    MODEL_TYPE         - Model type (default: gpt2)          ║
║    MODEL_PATH         - Model path (default: gpt2)          ║
║    MODEL_COMPONENT    - Component (default: trunk)          ║
║    DEVICE             - Device (default: auto)              ║
║    MAX_MODELS         - Max models (default: 10)            ║
║    LOG_LEVEL          - Log level (default: INFO)           ║
║    LOG_FILE           - Log file (default: None)            ║
║    ENABLE_METRICS     - Enable metrics (default: true)      ║
║                                                              ║
║  Example:                                                    ║
║    export MODEL_TYPE=qwen2                                  ║
║    export DEVICE=cuda                                        ║
║    python production_setup.py                                ║
╚══════════════════════════════════════════════════════════════╝
    """)

    try:
        # Run async server
        asyncio.run(run_production_server())

    except KeyboardInterrupt:
        print("\n\nServer stopped by user")

    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
