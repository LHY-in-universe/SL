"""
Basic example of splitting a GPT-2 model using SplitLearn.

This example demonstrates the core functionality of splitting a model
into Bottom, Trunk, and Top components.
"""

import torch
from transformers import AutoTokenizer
from splitlearn import ModelFactory


def main():
    # Configuration
    model_type = 'gpt2'
    model_name = 'gpt2'  # 12 layers total
    split_point_1 = 2    # Bottom: layers 0-1
    split_point_2 = 10   # Trunk: layers 2-9, Top: layers 10-11
    device = 'cpu'

    print(f"Loading {model_name} and splitting at layers {split_point_1} and {split_point_2}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Split the model using ModelFactory
    bottom, trunk, top = ModelFactory.create_split_models(
        model_type=model_type,
        model_name_or_path=model_name,
        split_point_1=split_point_1,
        split_point_2=split_point_2,
        device=device
    )

    print("\n=== Model Split Summary ===")
    print(f"Bottom Model: Embeddings + Layers 0-{split_point_1-1}")
    print(f"Trunk Model: Layers {split_point_1}-{split_point_2-1}")
    print(f"Top Model: Layers {split_point_2}-11 + LM Head")

    # Prepare input
    input_text = "The quick brown fox"
    print(f"\n=== Testing with input: '{input_text}' ===")

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Forward pass through split models
    print("\nStep 1: Bottom model (embeddings + first layers)")
    with torch.no_grad():
        hidden_1 = bottom(input_ids)
    print(f"  Output shape: {hidden_1.shape}")

    print("\nStep 2: Trunk model (middle layers)")
    with torch.no_grad():
        hidden_2 = trunk(hidden_1)
    print(f"  Output shape: {hidden_2.shape}")

    print("\nStep 3: Top model (final layers + LM head)")
    with torch.no_grad():
        output = top(hidden_2)
    print(f"  Logits shape: {output.logits.shape}")

    # Get predictions
    predicted_token_ids = output.logits[0, -1].argmax(dim=-1)
    predicted_token = tokenizer.decode(predicted_token_ids)

    print(f"\n=== Prediction ===")
    print(f"Next token: '{predicted_token}'")

    # Generate a few more tokens
    print("\n=== Generating text ===")
    generated_ids = input_ids.clone()
    max_new_tokens = 10

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
