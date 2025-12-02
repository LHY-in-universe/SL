"""
Storage management utilities for split models.

Provides utilities for managing model storage paths and directories.
"""

from pathlib import Path
from typing import Dict, Union
import json
from datetime import datetime


class StorageManager:
    """
    Manage storage paths and directories for split models.

    This class provides utilities for:
    - Generating standardized paths for split model components
    - Creating storage directory structure
    - Managing metadata files
    """

    @staticmethod
    def get_split_model_path(
        base_path: str,
        model_name: str,
        component: str,
        split_config: str
    ) -> Path:
        """
        Generate standardized path for split model component.

        Args:
            base_path: Base storage directory (e.g., "./models")
            model_name: Model name (e.g., "gpt2")
            component: Component type ("bottom", "trunk", or "top")
            split_config: Split configuration string (e.g., "2-10")

        Returns:
            Path object for the model file

        Example:
            >>> path = StorageManager.get_split_model_path(
            ...     "./models", "gpt2", "bottom", "2-10"
            ... )
            >>> str(path)
            './models/bottom/gpt2_2-10_bottom.pt'
        """
        if component not in ["bottom", "trunk", "top"]:
            raise ValueError(f"Invalid component: {component}. Must be 'bottom', 'trunk', or 'top'")

        base = Path(base_path)
        component_dir = base / component
        filename = f"{model_name}_{split_config}_{component}.pt"

        return component_dir / filename

    @staticmethod
    def create_storage_directories(base_path: str) -> Dict[str, Path]:
        """
        Create storage directories for split model components.

        Creates three subdirectories: bottom/, trunk/, top/

        Args:
            base_path: Base storage directory path

        Returns:
            Dictionary mapping component names to their directory paths

        Example:
            >>> dirs = StorageManager.create_storage_directories("./models")
            >>> dirs.keys()
            dict_keys(['bottom', 'trunk', 'top'])
        """
        base = Path(base_path)

        directories = {}
        for component in ["bottom", "trunk", "top"]:
            component_dir = base / component
            component_dir.mkdir(parents=True, exist_ok=True)
            directories[component] = component_dir

        return directories

    @staticmethod
    def get_model_metadata_path(model_path: Path) -> Path:
        """
        Get path for model metadata JSON file.

        Args:
            model_path: Path to model file

        Returns:
            Path to metadata JSON file

        Example:
            >>> model_path = Path("./models/bottom/gpt2_2-10_bottom.pt")
            >>> meta_path = StorageManager.get_model_metadata_path(model_path)
            >>> str(meta_path)
            './models/bottom/gpt2_2-10_bottom_metadata.json'
        """
        return model_path.parent / f"{model_path.stem}_metadata.json"
