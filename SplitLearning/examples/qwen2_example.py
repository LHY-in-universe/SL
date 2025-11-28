"""
Example of splitting a Qwen2 model using SplitLearn.

This demonstrates that the library works with different model architectures.
"""

import torch
from transformers import AutoTokenizer
from splitlearn import ModelFactory


def main():
    # Configuration
    model_type = 'qwen2'
    model_name = 'Qwen/Qwen2-0.5B'  # Small Qwen2 model (24 layers)
    split_point_1 = 8     # Bottom: layers 0-7
    split_point_2 = 16    # Trunk: layers 8-15, Top: layers 16-23
    device = 'cpu'

    print(f"Loading {model_name}...")
    print(f"Splitting at layers {split_point_1} and {split_point_2}...")

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
    print(f"Top Model: Layers {split_point_2}-end + LM Head")

    # Prepare input
    input_text = "人工智能的未来是"
    print(f"\n=== Testing with input: '{input_text}' ===")

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Forward pass through split models
    print("\nRunning inference through split models...")
    with torch.no_grad():
        hidden_1 = bottom(input_ids)
        print(f"  After bottom: {hidden_1.shape}")

        hidden_2 = trunk(hidden_1)
        print(f"  After trunk: {hidden_2.shape}")

        output = top(hidden_2)
        print(f"  Final logits: {output.logits.shape}")

    # Get predictions
    predicted_token_ids = output.logits[0, -1].argmax(dim=-1)
    predicted_token = tokenizer.decode(predicted_token_ids)

    print(f"\n=== Prediction ===")
    print(f"Next token: '{predicted_token}'")

    # Generate text
    print("\n=== Generating text ===")
    generated_ids = input_ids.clone()
    max_new_tokens = 20

    for i in range(max_new_tokens):
        with torch.no_grad():
            h1 = bottom(generated_ids)
            h2 = trunk(h1)
            output = top(h2)

        next_token_id = output.logits[0, -1].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()
