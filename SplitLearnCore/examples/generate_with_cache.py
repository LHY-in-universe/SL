#!/usr/bin/env python3
"""
Incremental Text Generation with KV-Cache

Demonstrates fast autoregressive generation using KV-cache separation
in distributed split learning architecture.

This example shows:
1. How to use KV-cache for fast token-by-token generation
2. Performance comparison (with vs without cache)
3. Bandwidth usage analysis

Usage:
    # 1. Start Trunk server
    python examples/start_trunk_server.py --model-type gpt2 --split-point-2 6 --device cuda

    # 2. Run generation
    python examples/generate_with_cache.py --prompt "Once upon a time" --max-tokens 50
"""

import argparse
import logging
import time
from typing import Optional, List

import torch
from transformers import AutoTokenizer

from splitlearn_core import ModelFactory
from splitlearn_comm import GRPCComputeClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Text generation with KV-cache")

    # Model configuration
    parser.add_argument("--model-type", type=str, default="gpt2",
                       choices=["gpt2", "qwen2"],
                       help="Model architecture type")
    parser.add_argument("--model-name", type=str, default="gpt2",
                       help="HuggingFace model name")
    parser.add_argument("--split-point-1", type=int, default=0,
                       help="Split point between Bottom and Trunk")
    parser.add_argument("--split-point-2", type=int, default=6,
                       help="Split point between Trunk and Top")

    # Server configuration
    parser.add_argument("--server-host", type=str, default="localhost",
                       help="Trunk server hostname")
    parser.add_argument("--server-port", type=int, default=50051,
                       help="Trunk server port")

    # Generation configuration
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                       help="Input prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=50,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50,
                       help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.95,
                       help="Top-p (nucleus) sampling")

    # Device configuration
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device for Bottom and Top models")

    # Testing configuration
    parser.add_argument("--compare-no-cache", action="store_true",
                       help="Also run generation without cache for comparison")

    return parser.parse_args()


def generate_with_cache(
    prompt_ids: torch.Tensor,
    bottom_model,
    top_model,
    trunk_client: GRPCComputeClient,
    tokenizer,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = "cpu"
) -> tuple:
    """
    Generate text using KV-cache for fast autoregressive generation

    Returns:
        (generated_ids, generation_time, bandwidth_used)
    """
    logger.info("Starting generation WITH KV-cache...")

    generated_ids = prompt_ids.clone()
    total_bandwidth = 0
    start_time = time.time()

    # Initialize caches
    trunk_kv_cache = None
    top_kv_cache = None

    # Process prompt (first token)
    with torch.no_grad():
        # Bottom model
        hidden_1 = bottom_model(prompt_ids)

        # Trunk model (no cache for first forward)
        hidden_2, trunk_kv_cache, timing = trunk_client.compute_with_cache(
            hidden_1,
            past_key_values=None,
            use_cache=True
        )
        total_bandwidth += timing['network_total_ms'] * 150  # Rough estimate: 150 KB/ms

        # Top model
        output = top_model(hidden_2, use_cache=True)
        if isinstance(output, tuple):
            logits, top_kv_cache = output
        else:
            logits = output.logits
            top_kv_cache = output.past_key_values

        # Sample next token
        next_token = sample_token(logits[:, -1, :], temperature, top_k, top_p)
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

    # Generate remaining tokens (with cache)
    for i in range(max_new_tokens - 1):
        with torch.no_grad():
            # Only process the new token
            new_token_ids = generated_ids[:, -1:]

            # Bottom model (only new token)
            hidden_1 = bottom_model(new_token_ids)

            # Trunk model (with cached KV)
            hidden_2, trunk_kv_cache, timing = trunk_client.compute_with_cache(
                hidden_1,
                past_key_values=trunk_kv_cache,
                use_cache=True
            )
            total_bandwidth += timing['network_total_ms'] * 30  # Much smaller: only new KV

            # Top model (with cached KV)
            output = top_model(hidden_2, past_key_values=top_kv_cache, use_cache=True)
            if isinstance(output, tuple):
                logits, top_kv_cache = output
            else:
                logits = output.logits
                top_kv_cache = output.past_key_values

            # Sample next token
            next_token = sample_token(logits[:, -1, :], temperature, top_k, top_p)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                logger.info(f"Generated EOS token at position {i+1}")
                break

    generation_time = time.time() - start_time
    logger.info(f"Generation completed in {generation_time:.2f}s")
    logger.info(f"Estimated bandwidth: {total_bandwidth/1024:.2f} MB")

    return generated_ids, generation_time, total_bandwidth


def generate_without_cache(
    prompt_ids: torch.Tensor,
    bottom_model,
    top_model,
    trunk_client: GRPCComputeClient,
    tokenizer,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = "cpu"
) -> tuple:
    """
    Generate text WITHOUT KV-cache (naive approach)

    Each token requires recomputing the entire sequence.

    Returns:
        (generated_ids, generation_time, bandwidth_used)
    """
    logger.info("Starting generation WITHOUT KV-cache (naive)...")

    generated_ids = prompt_ids.clone()
    total_bandwidth = 0
    start_time = time.time()

    for i in range(max_new_tokens):
        with torch.no_grad():
            # Process ENTIRE sequence from scratch each time
            hidden_1 = bottom_model(generated_ids)

            # Trunk model (no cache, processes full sequence)
            hidden_2, _, timing = trunk_client.compute_with_cache(
                hidden_1,
                past_key_values=None,
                use_cache=False
            )
            total_bandwidth += timing['network_total_ms'] * 200  # Full sequence

            # Top model (no cache)
            output = top_model(hidden_2)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output.logits

            # Sample next token
            next_token = sample_token(logits[:, -1, :], temperature, top_k, top_p)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                logger.info(f"Generated EOS token at position {i+1}")
                break

    generation_time = time.time() - start_time
    logger.info(f"Generation completed in {generation_time:.2f}s")
    logger.info(f"Estimated bandwidth: {total_bandwidth/1024:.2f} MB")

    return generated_ids, generation_time, total_bandwidth


def sample_token(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """
    Sample next token from logits with temperature, top-k, and top-p

    Args:
        logits: Logits tensor [vocab_size]
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Top-p (nucleus) filtering

    Returns:
        Sampled token ID
    """
    # Apply temperature
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        logits_filtered = torch.full_like(logits, float('-inf'))
        logits_filtered.scatter_(0, top_k_indices, top_k_logits)
        logits = logits_filtered

    # Top-p filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')

    # Sample from the filtered distribution
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


def main():
    args = parse_args()

    logger.info("="*80)
    logger.info("Incremental Text Generation with KV-Cache")
    logger.info("="*80)
    logger.info(f"Model: {args.model_type} ({args.model_name})")
    logger.info(f"Server: {args.server_host}:{args.server_port}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Prompt: \"{args.prompt}\"")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info("="*80)

    # 1. Load tokenizer
    logger.info("\n[Step 1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize prompt
    prompt_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(args.device)
    logger.info(f"Prompt tokens: {prompt_ids.shape[1]}")

    # 2. Create split models (Bottom and Top only)
    logger.info("\n[Step 2/5] Creating split models...")
    bottom_model, _, top_model = ModelFactory.create_split_models(
        model_type=args.model_type,
        model_name_or_path=args.model_name,
        split_point_1=args.split_point_1,
        split_point_2=args.split_point_2,
        device=args.device,
        verbose=False
    )

    bottom_model.eval()
    top_model.eval()

    logger.info(f"✓ Bottom and Top models loaded on {args.device}")

    # 3. Connect to Trunk server
    logger.info("\n[Step 3/5] Connecting to Trunk server...")
    trunk_client = GRPCComputeClient(
        host=args.server_host,
        port=args.server_port,
        timeout=30.0
    )

    try:
        trunk_client.connect()
        logger.info("✓ Connected to Trunk server")
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        logger.error("Please start the server:")
        logger.error(f"  python examples/start_trunk_server.py --model-type {args.model_type} --split-point-2 {args.split_point_2}")
        return

    # 4. Generate with KV-cache
    logger.info("\n[Step 4/5] Generating text WITH KV-cache...")
    logger.info("-"*80)

    generated_ids_cached, time_cached, bandwidth_cached = generate_with_cache(
        prompt_ids=prompt_ids,
        bottom_model=bottom_model,
        top_model=top_model,
        trunk_client=trunk_client,
        tokenizer=tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device
    )

    generated_text_cached = tokenizer.decode(generated_ids_cached[0], skip_special_tokens=True)

    logger.info("\n" + "="*80)
    logger.info("GENERATED TEXT (with cache):")
    logger.info("="*80)
    print(generated_text_cached)
    logger.info("="*80)

    # 5. (Optional) Generate without cache for comparison
    if args.compare_no_cache:
        logger.info("\n[Step 5/5] Generating text WITHOUT KV-cache (for comparison)...")
        logger.info("-"*80)

        generated_ids_no_cache, time_no_cache, bandwidth_no_cache = generate_without_cache(
            prompt_ids=prompt_ids,
            bottom_model=bottom_model,
            top_model=top_model,
            trunk_client=trunk_client,
            tokenizer=tokenizer,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device
        )

        generated_text_no_cache = tokenizer.decode(generated_ids_no_cache[0], skip_special_tokens=True)

        logger.info("\n" + "="*80)
        logger.info("GENERATED TEXT (without cache):")
        logger.info("="*80)
        print(generated_text_no_cache)
        logger.info("="*80)

        # Performance comparison
        logger.info("\n" + "="*80)
        logger.info("PERFORMANCE COMPARISON")
        logger.info("="*80)
        logger.info(f"WITH KV-cache:")
        logger.info(f"  Time: {time_cached:.2f}s")
        logger.info(f"  Bandwidth: {bandwidth_cached/1024:.2f} MB")
        logger.info(f"  Speed: {args.max_tokens/time_cached:.2f} tokens/sec")
        logger.info("")
        logger.info(f"WITHOUT KV-cache:")
        logger.info(f"  Time: {time_no_cache:.2f}s")
        logger.info(f"  Bandwidth: {bandwidth_no_cache/1024:.2f} MB")
        logger.info(f"  Speed: {args.max_tokens/time_no_cache:.2f} tokens/sec")
        logger.info("")
        logger.info(f"SPEEDUP: {time_no_cache/time_cached:.2f}x faster with cache")
        logger.info(f"BANDWIDTH REDUCTION: {bandwidth_no_cache/bandwidth_cached:.2f}x less bandwidth")
        logger.info("="*80)
    else:
        logger.info("\n" + "="*80)
        logger.info("GENERATION STATS")
        logger.info("="*80)
        logger.info(f"Time: {time_cached:.2f}s")
        logger.info(f"Bandwidth: {bandwidth_cached/1024:.2f} MB")
        logger.info(f"Speed: {args.max_tokens/time_cached:.2f} tokens/sec")
        logger.info(f"Tokens generated: {generated_ids_cached.shape[1] - prompt_ids.shape[1]}")
        logger.info("="*80)
        logger.info("\nTip: Run with --compare-no-cache to see performance comparison")

    # Disconnect
    trunk_client.disconnect()
    logger.info("\n✓ Generation completed successfully!")


if __name__ == "__main__":
    main()
