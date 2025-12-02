"""
Example of directly creating split models from a pretrained model.

This demonstrates the lower-level API for more control over model splitting.
"""

import torch
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
from splitlearn.models.gpt2 import GPT2BottomModel, GPT2TrunkModel, GPT2TopModel


def main():
    model_name = 'gpt2'

    print(f"Loading full {model_name} model...")

    # Load full model and config
    config = GPT2Config.from_pretrained(model_name)
    full_model = GPT2LMHeadModel.from_pretrained(model_name)
    state_dict = full_model.state_dict()

    print(f"Total layers in model: {config.n_layer}")

    # Define split points
    split_point_1 = 3
    split_point_2 = 9

    print(f"\n=== Creating Split Models ===")
    print(f"Bottom: layers 0-{split_point_1-1}")
    print(f"Trunk: layers {split_point_1}-{split_point_2-1}")
    print(f"Top: layers {split_point_2}-{config.n_layer-1}")

    # Create bottom model
    print("\nCreating bottom model...")
    bottom = GPT2BottomModel.from_pretrained_split(
        full_state_dict=state_dict,
        config=config,
        end_layer=split_point_1
    )

    # Create trunk model
    print("Creating trunk model...")
    trunk = GPT2TrunkModel.from_pretrained_split(
        full_state_dict=state_dict,
        config=config,
        start_layer=split_point_1,
        end_layer=split_point_2
    )

    # Create top model
    print("Creating top model...")
    top = GPT2TopModel.from_pretrained_split(
        full_state_dict=state_dict,
        config=config,
        start_layer=split_point_2
    )

    print("\n=== Testing Split Models ===")

    # Load tokenizer and prepare input
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_text = "Artificial intelligence is"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Test with split models
    print(f"Input: '{input_text}'")

    with torch.no_grad():
        # Split model inference
        h1 = bottom(input_ids)
        h2 = trunk(h1)
        split_output = top(h2)
        split_logits = split_output.logits

        # Full model inference for comparison
        full_output = full_model(input_ids)
        full_logits = full_output.logits

    # Compare outputs
    print("\n=== Verification ===")
    print(f"Split model logits shape: {split_logits.shape}")
    print(f"Full model logits shape: {full_logits.shape}")

    # Calculate difference
    max_diff = torch.max(torch.abs(split_logits - full_logits)).item()
    mean_diff = torch.mean(torch.abs(split_logits - full_logits)).item()

    print(f"\nLogits difference:")
    print(f"  Max absolute diff: {max_diff:.6f}")
    print(f"  Mean absolute diff: {mean_diff:.6f}")

    if max_diff < 1e-4:
        print("\n✓ Split models produce identical outputs to full model!")
    else:
        print(f"\n⚠ Warning: Outputs differ by {max_diff:.6f}")

    # Generate text with split models
    print("\n=== Generating Text ===")
    generated_ids = input_ids.clone()
    max_new_tokens = 15

    for i in range(max_new_tokens):
        with torch.no_grad():
            h1 = bottom(generated_ids)
            h2 = trunk(h1)
            output = top(h2)

        next_token_id = output.logits[0, -1].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

    generated_text = tokenizer.decode(generated_ids[0])
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()
