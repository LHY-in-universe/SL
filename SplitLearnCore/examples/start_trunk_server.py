#!/usr/bin/env python3
"""
Start Trunk Server for Distributed Training

This script starts a gRPC server hosting the Trunk model for distributed training.

Usage:
    python examples/start_trunk_server.py --model-type qwen2 --split-point-1 0 --split-point-2 14

    # With LoRA
    python examples/start_trunk_server.py --model-type qwen2 --use-lora --lora-rank 8

    # With GPU
    python examples/start_trunk_server.py --device cuda --port 50051
"""

import argparse
import logging
import signal
import sys
import time

import torch

from splitlearn_core import ModelFactory
from splitlearn_core.training import SplitLoraConfig, OptimizerConfig
from splitlearn_comm import ComputeServicer, start_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Start Trunk server for distributed training")

    # Model configuration
    parser.add_argument("--model-type", type=str, default="qwen2",
                       choices=["qwen2", "qwen3_vl", "gpt2", "gemma"],
                       help="Model architecture type")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-0.5B",
                       help="HuggingFace model name or path")
    parser.add_argument("--split-point-1", type=int, default=0,
                       help="Split point between Bottom and Trunk")
    parser.add_argument("--split-point-2", type=int, default=14,
                       help="Split point between Trunk and Top")

    # Server configuration
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Server host (0.0.0.0 for all interfaces)")
    parser.add_argument("--port", type=int, default=50051,
                       help="Server port")
    parser.add_argument("--max-workers", type=int, default=10,
                       help="Maximum number of worker threads")

    # LoRA configuration
    parser.add_argument("--use-lora", action="store_true",
                       help="Enable LoRA fine-tuning")
    parser.add_argument("--lora-rank", type=int, default=8,
                       help="LoRA rank (if --use-lora)")
    parser.add_argument("--lora-alpha", type=int, default=16,
                       help="LoRA alpha scaling factor")

    # Training configuration
    parser.add_argument("--enable-training", action="store_true", default=True,
                       help="Enable training mode (activation caching, backward pass)")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate for Trunk optimizer")
    parser.add_argument("--cache-size", type=int, default=100,
                       help="Activation cache size")
    parser.add_argument("--cache-ttl", type=float, default=60.0,
                       help="Activation cache TTL in seconds")

    # Device configuration
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device for Trunk model")

    return parser.parse_args()


def create_trunk_optimizer(trunk_model, learning_rate: float):
    """Create optimizer for Trunk model"""
    from torch.optim import AdamW

    trainable_params = [p for p in trunk_model.parameters() if p.requires_grad]

    if len(trainable_params) == 0:
        logger.warning("No trainable parameters in Trunk model!")
        return None

    optimizer = AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    logger.info(f"Created Trunk optimizer: lr={learning_rate}, trainable_params={len(trainable_params)}")
    return optimizer


def main():
    args = parse_args()

    logger.info("="*80)
    logger.info("SplitLearn Trunk Server")
    logger.info("="*80)
    logger.info(f"Model: {args.model_type} ({args.model_name})")
    logger.info(f"Server: {args.host}:{args.port}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Training mode: {'Enabled' if args.enable_training else 'Disabled'}")
    if args.use_lora:
        logger.info(f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
    logger.info("="*80)

    # 1. Create split models (only need Trunk)
    logger.info("\n[Step 1/3] Loading Trunk model...")

    lora_config = None
    if args.use_lora:
        # Create LoRA configuration
        if args.model_type == "qwen2":
            lora_config = SplitLoraConfig.for_qwen2(
                rank=args.lora_rank,
                alpha=args.lora_alpha
            )
        elif args.model_type == "qwen3_vl":
            lora_config = SplitLoraConfig.for_qwen3_vl(
                rank=args.lora_rank,
                alpha=args.lora_alpha
            )
        elif args.model_type == "gpt2":
            lora_config = SplitLoraConfig.for_gpt2(
                rank=args.lora_rank,
                alpha=args.lora_alpha
            )
        else:
            lora_config = SplitLoraConfig(
                rank=args.lora_rank,
                alpha=args.lora_alpha
            )

        lora_config.learning_rate = args.learning_rate

    # Create split models
    _, trunk_model, _ = ModelFactory.create_split_models(
        model_type=args.model_type,
        model_name_or_path=args.model_name,
        split_point_1=args.split_point_1,
        split_point_2=args.split_point_2,
        device=args.device,
        use_lora=args.use_lora,
        lora_config=lora_config,
        verbose=True
    )

    logger.info(f"✓ Trunk model loaded on {args.device}")

    # 2. Create optimizer if training is enabled
    trunk_optimizer = None
    if args.enable_training:
        logger.info("\n[Step 2/3] Creating Trunk optimizer...")
        trunk_optimizer = create_trunk_optimizer(trunk_model, args.learning_rate)

        if trunk_optimizer is None:
            logger.warning("Training mode enabled but no trainable parameters found")
            logger.warning("Continuing in inference-only mode")
            args.enable_training = False

    # 3. Start server
    logger.info("\n[Step 3/3] Starting gRPC server...")

    # Create compute function from trunk model
    def compute_fn(x):
        """Trunk model forward pass"""
        return trunk_model(x)

    # Create servicer
    servicer = ComputeServicer(
        compute_fn=compute_fn,
        device=args.device,
        enable_training=args.enable_training,
        trunk_optimizer=trunk_optimizer,
        activation_cache_size=args.cache_size,
        activation_cache_ttl=args.cache_ttl
    )

    # Start server
    logger.info(f"Starting server on {args.host}:{args.port}...")
    server = start_server(
        servicer=servicer,
        host=args.host,
        port=args.port,
        max_workers=args.max_workers
    )

    logger.info("="*80)
    logger.info("✓ Server started successfully!")
    logger.info(f"  Listening on: {args.host}:{args.port}")
    logger.info(f"  Training mode: {'Enabled' if args.enable_training else 'Disabled'}")
    logger.info(f"  Cache size: {args.cache_size} entries")
    logger.info(f"  Cache TTL: {args.cache_ttl} seconds")
    logger.info("\nServer is ready to accept connections.")
    logger.info("Press Ctrl+C to stop the server.")
    logger.info("="*80)

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("\nShutdown signal received, stopping server...")
        server.stop(grace=5)
        logger.info("Server stopped")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Keep server running
    try:
        while True:
            time.sleep(3600)  # Sleep for 1 hour
    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
        server.stop(grace=5)
        logger.info("Server stopped")


if __name__ == "__main__":
    main()
