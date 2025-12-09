#!/usr/bin/env python3
"""
Distributed LoRA Training Example for SplitLearn

This example demonstrates how to fine-tune a language model using LoRA
in a distributed setting with SplitLearn architecture:
- Client side: Bottom model + Top model
- Server side: Trunk model (via gRPC)

Requirements:
    pip install torch transformers datasets peft splitlearn-core splitlearn-comm

Usage:
    # 1. Start Trunk server first (on server machine or localhost):
    python examples/start_trunk_server.py --model-type qwen2 --split-point-1 0 --split-point-2 14

    # 2. Run training (on client machine):
    python examples/train_lora_distributed.py --server-host localhost --server-port 50051
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from splitlearn_core import ModelFactory
from splitlearn_core.training import (
    SplitLoraConfig,
    SplitOptimizer,
    OptimizerConfig,
    SplitTrainer,
    TrainingConfig,
    create_scheduler
)
from splitlearn_comm import GRPCComputeClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Distributed LoRA training with SplitLearn")

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
    parser.add_argument("--server-host", type=str, default="localhost",
                       help="Trunk server hostname")
    parser.add_argument("--server-port", type=int, default=50051,
                       help="Trunk server port")

    # LoRA configuration
    parser.add_argument("--lora-rank", type=int, default=8,
                       help="LoRA rank (typically 4-16)")
    parser.add_argument("--lora-alpha", type=int, default=16,
                       help="LoRA alpha scaling factor")
    parser.add_argument("--lora-dropout", type=float, default=0.1,
                       help="LoRA dropout probability")

    # Training configuration
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--max-seq-length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                       help="Max gradient norm for clipping")
    parser.add_argument("--warmup-steps", type=int, default=100,
                       help="Number of warmup steps for scheduler")

    # Dataset configuration
    parser.add_argument("--dataset-name", type=str, default="wikitext",
                       help="Dataset name from HuggingFace")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1",
                       help="Dataset configuration")
    parser.add_argument("--max-train-samples", type=int, default=10000,
                       help="Maximum training samples (for quick testing)")

    # Output configuration
    parser.add_argument("--output-dir", type=str, default="./checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--save-interval", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--log-interval", type=int, default=10,
                       help="Log metrics every N steps")
    parser.add_argument("--eval-interval", type=int, default=100,
                       help="Evaluate every N steps")

    # Device configuration
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device for Bottom and Top models")
    parser.add_argument("--mixed-precision", action="store_true",
                       help="Enable mixed precision training (FP16/BF16)")

    return parser.parse_args()


def load_and_prepare_dataset(args, tokenizer):
    """Load and tokenize dataset"""
    logger.info(f"Loading dataset: {args.dataset_name}/{args.dataset_config}")

    # Load dataset
    dataset = load_dataset(args.dataset_name, args.dataset_config)

    # Take subset for quick testing
    if args.max_train_samples:
        train_size = min(args.max_train_samples, len(dataset['train']))
        dataset['train'] = dataset['train'].select(range(train_size))
        logger.info(f"Using {train_size} training samples")

    # Tokenization function
    def tokenize_function(examples):
        """Tokenize text and create labels"""
        # Tokenize with truncation and padding
        outputs = tokenizer(
            examples['text'],
            truncation=True,
            max_length=args.max_seq_length,
            padding='max_length',
            return_tensors='pt'
        )

        # For causal language modeling, labels = input_ids
        outputs['labels'] = outputs['input_ids'].clone()

        return outputs

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing"
    )

    # Set format to PyTorch
    tokenized_dataset.set_format('torch')

    return tokenized_dataset


def create_dataloaders(tokenized_dataset, batch_size: int):
    """Create training and validation dataloaders"""
    train_dataloader = DataLoader(
        tokenized_dataset['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    eval_dataloader = None
    if 'validation' in tokenized_dataset:
        eval_dataloader = DataLoader(
            tokenized_dataset['validation'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

    logger.info(f"Created dataloaders: train={len(train_dataloader)} batches")
    if eval_dataloader:
        logger.info(f"  validation={len(eval_dataloader)} batches")

    return train_dataloader, eval_dataloader


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("SplitLearn Distributed LoRA Training")
    logger.info("="*80)
    logger.info(f"Model: {args.model_type} ({args.model_name})")
    logger.info(f"Server: {args.server_host}:{args.server_port}")
    logger.info(f"Device: {args.device}")
    logger.info(f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
    logger.info(f"Training: {args.num_epochs} epochs, lr={args.learning_rate}")
    logger.info("="*80)

    # 1. Load tokenizer
    logger.info("\n[Step 1/7] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Set padding token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token = eos_token = '{tokenizer.eos_token}'")

    # 2. Load and prepare dataset
    logger.info("\n[Step 2/7] Loading and tokenizing dataset...")
    tokenized_dataset = load_and_prepare_dataset(args, tokenizer)
    train_dataloader, eval_dataloader = create_dataloaders(
        tokenized_dataset,
        args.batch_size
    )

    # 3. Create split models with LoRA
    logger.info("\n[Step 3/7] Creating split models with LoRA...")

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
            alpha=args.lora_alpha,
            dropout=args.lora_dropout
        )

    lora_config.learning_rate = args.learning_rate

    # Create split models (only Bottom and Top on client side)
    bottom_model, _, top_model = ModelFactory.create_split_models(
        model_type=args.model_type,
        model_name_or_path=args.model_name,
        split_point_1=args.split_point_1,
        split_point_2=args.split_point_2,
        device=args.device,
        use_lora=True,
        lora_config=lora_config,
        verbose=True
    )

    logger.info(f"✓ Bottom model created with LoRA on {args.device}")
    logger.info(f"✓ Top model created with LoRA on {args.device}")
    logger.info(f"✓ Trunk model runs on server at {args.server_host}:{args.server_port}")

    # 4. Connect to Trunk server
    logger.info("\n[Step 4/7] Connecting to Trunk server...")
    trunk_client = GRPCComputeClient(
        host=args.server_host,
        port=args.server_port,
        max_retries=3,
        timeout=30.0
    )

    try:
        trunk_client.connect()
        logger.info("✓ Connected to Trunk server successfully")

        # Get server info
        info = trunk_client.get_service_info()
        logger.info(f"  Server version: {info.get('version', 'unknown')}")
        logger.info(f"  Server device: {info.get('device', 'unknown')}")
    except Exception as e:
        logger.error(f"Failed to connect to Trunk server: {e}")
        logger.error("Please ensure the server is running:")
        logger.error(f"  python examples/start_trunk_server.py --model-type {args.model_type} --split-point-1 {args.split_point_1} --split-point-2 {args.split_point_2}")
        return

    # 5. Create optimizer
    logger.info("\n[Step 5/7] Creating optimizer and scheduler...")
    optimizer_config = OptimizerConfig(
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        grad_clip_norm=args.max_grad_norm
    )

    optimizer = SplitOptimizer(
        bottom_model=bottom_model,
        top_model=top_model,
        config=optimizer_config
    )

    logger.info(f"✓ Created optimizer: lr={args.learning_rate}")

    # Create learning rate scheduler
    total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    schedulers = create_scheduler(
        optimizer=optimizer,
        scheduler_type='cosine',
        num_training_steps=total_steps,
        num_warmup_steps=args.warmup_steps
    )

    logger.info(f"✓ Created scheduler: warmup={args.warmup_steps}, total={total_steps}")

    # 6. Create trainer
    logger.info("\n[Step 6/7] Creating trainer...")
    training_config = TrainingConfig(
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        output_dir=str(output_dir),
        mixed_precision=args.mixed_precision
    )

    trainer = SplitTrainer(
        bottom_model=bottom_model,
        top_model=top_model,
        trunk_client=trunk_client,
        optimizer=optimizer,
        config=training_config,
        device=args.device,
        schedulers=schedulers
    )

    logger.info("✓ Trainer created successfully")

    # 7. Start training
    logger.info("\n[Step 7/7] Starting training...")
    logger.info("="*80)

    try:
        trainer.train(
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader
        )

        logger.info("="*80)
        logger.info("Training completed successfully!")
        logger.info(f"Best evaluation loss: {trainer.best_eval_loss:.4f}")
        logger.info(f"Total steps: {trainer.global_step}")
        logger.info(f"Checkpoints saved to: {output_dir}")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint(output_dir / f"checkpoint_step_{trainer.global_step}_interrupted")
        logger.info("✓ Checkpoint saved")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise

    finally:
        # Disconnect from server
        trunk_client.disconnect()
        logger.info("Disconnected from Trunk server")


if __name__ == "__main__":
    main()
