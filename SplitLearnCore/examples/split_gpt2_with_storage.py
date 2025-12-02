"""
Example: Create split GPT-2 models with auto-save.

This example demonstrates how to:
1. Create split models from a pretrained GPT-2
2. Automatically save them to organized directories
3. Inspect the saved metadata
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from splitlearn import ModelFactory
import json


def main():
    """Create split GPT-2 models with auto-save."""

    print("\n" + "="*60)
    print("Creating Split GPT-2 Models with Auto-Save")
    print("="*60 + "\n")

    # Configuration
    model_type = 'gpt2'
    model_name = 'gpt2'
    split_point_1 = 2   # Bottom model: layers 0-1
    split_point_2 = 10  # Trunk model: layers 2-9, Top model: layers 10-11
    storage_path = './models'

    print(f"Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Split points: {split_point_1}, {split_point_2}")
    print(f"  Storage path: {storage_path}")
    print()

    # Create split models WITH auto-save
    print("Creating split models with auto-save enabled...")
    print("-" * 60)

    bottom_model, trunk_model, top_model = ModelFactory.create_split_models(
        model_type=model_type,
        model_name_or_path=model_name,
        split_point_1=split_point_1,
        split_point_2=split_point_2,
        device='cpu',
        storage_path=storage_path,
        auto_save=True  # Enable auto-save
    )

    print("\n" + "="*60)
    print("Models created and saved successfully!")
    print("="*60 + "\n")

    # Verify saved files
    print("Verifying saved files...")
    print("-" * 60)

    split_config = f"{split_point_1}-{split_point_2}"
    storage_dir = Path(storage_path)

    # Check each component
    components = ['bottom', 'trunk', 'top']
    for component in components:
        model_file = storage_dir / component / f"{model_name}_{split_config}_{component}.pt"
        metadata_file = storage_dir / component / f"{model_name}_{split_config}_{component}_metadata.json"

        print(f"\n{component.upper()} Model:")
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  ✓ Model file: {model_file} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ Model file not found: {model_file}")

        if metadata_file.exists():
            print(f"  ✓ Metadata file: {metadata_file}")

            # Load and display metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            print(f"    - Component: {metadata['component']}")
            print(f"    - Layers: {metadata['start_layer']}-{metadata['end_layer']} ({metadata['num_layers']} layers)")
            print(f"    - Parameters: {metadata['num_parameters']:,}")
            print(f"    - Memory: {metadata['memory_mb']:.2f} MB")
            print(f"    - Saved at: {metadata['saved_at']}")
        else:
            print(f"  ✗ Metadata file not found: {metadata_file}")

    print("\n" + "="*60)
    print("Directory structure created:")
    print("="*60)
    print(f"{storage_path}/")
    for component in components:
        component_dir = storage_dir / component
        if component_dir.exists():
            print(f"├── {component}/")
            for file in sorted(component_dir.glob("*")):
                print(f"│   └── {file.name}")

    print("\n" + "="*60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("="*60 + "\n")

    print("Next steps:")
    print("  1. Load models using torch.load()")
    print("  2. Use models for split learning inference")
    print("  3. Share different components with different parties")
    print()


if __name__ == "__main__":
    main()
