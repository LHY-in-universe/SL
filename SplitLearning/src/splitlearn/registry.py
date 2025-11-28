"""
Model Registry - Central registration system for split models

This module provides a decorator-based registration mechanism for all
supported model types (GPT-2, LLaMA2, GPT-J, etc.)
"""
from typing import Dict, Type, List


class ModelRegistry:
    """
    Registry for split model classes

    Provides a centralized way to register and retrieve model implementations
    for different architectures and split parts (bottom/trunk/top).

    Usage:
        @ModelRegistry.register('gpt2', 'bottom')
        class GPT2BottomModel(BaseBottomModel):
            ...

        # Later retrieve:
        BottomCls = ModelRegistry.get_model_class('gpt2', 'bottom')
    """

    # Registry structure: {model_type: {part: class}}
    _registry: Dict[str, Dict[str, Type]] = {}

    @classmethod
    def register(cls, model_type: str, part: str):
        """
        Decorator to register a model class

        Args:
            model_type: Model architecture name (e.g., 'gpt2', 'llama2')
            part: Model part ('bottom', 'trunk', or 'top')

        Returns:
            Decorator function

        Example:
            @ModelRegistry.register('gpt2', 'bottom')
            class GPT2BottomModel(BaseBottomModel):
                pass
        """
        valid_parts = {'bottom', 'trunk', 'top'}
        if part not in valid_parts:
            raise ValueError(f"part must be one of {valid_parts}, got '{part}'")

        def decorator(model_class):
            # Initialize model_type dict if not exists
            if model_type not in cls._registry:
                cls._registry[model_type] = {}

            # Register the class
            cls._registry[model_type][part] = model_class

            # Add metadata to the class
            model_class._registry_info = {
                'model_type': model_type,
                'part': part,
            }

            return model_class

        return decorator

    @classmethod
    def get_model_class(cls, model_type: str, part: str) -> Type:
        """
        Retrieve a registered model class

        Args:
            model_type: Model architecture name
            part: Model part ('bottom', 'trunk', or 'top')

        Returns:
            Type: The registered model class

        Raises:
            KeyError: If model_type or part not found

        Example:
            BottomCls = ModelRegistry.get_model_class('gpt2', 'bottom')
            bottom = BottomCls(config, end_layer=2)
        """
        if model_type not in cls._registry:
            available = cls.list_supported_models()
            raise KeyError(
                f"Model type '{model_type}' not registered. "
                f"Available models: {available}"
            )

        if part not in cls._registry[model_type]:
            available_parts = list(cls._registry[model_type].keys())
            raise KeyError(
                f"Part '{part}' not registered for model '{model_type}'. "
                f"Available parts: {available_parts}"
            )

        return cls._registry[model_type][part]

    @classmethod
    def list_supported_models(cls) -> List[str]:
        """
        List all registered model types

        Returns:
            List[str]: List of model type names

        Example:
            >>> ModelRegistry.list_supported_models()
            ['gpt2', 'llama2', 'gpt-j']
        """
        return sorted(cls._registry.keys())

    @classmethod
    def is_model_registered(cls, model_type: str) -> bool:
        """
        Check if a model type is registered

        Args:
            model_type: Model architecture name

        Returns:
            bool: True if registered, False otherwise
        """
        return model_type in cls._registry

    @classmethod
    def is_complete(cls, model_type: str) -> bool:
        """
        Check if all parts (bottom/trunk/top) are registered for a model

        Args:
            model_type: Model architecture name

        Returns:
            bool: True if all three parts are registered
        """
        if model_type not in cls._registry:
            return False

        required_parts = {'bottom', 'trunk', 'top'}
        registered_parts = set(cls._registry[model_type].keys())

        return required_parts.issubset(registered_parts)

    @classmethod
    def get_model_info(cls) -> Dict[str, Dict[str, bool]]:
        """
        Get registration status for all models

        Returns:
            Dict mapping model_type to {part: is_registered}

        Example:
            >>> ModelRegistry.get_model_info()
            {
                'gpt2': {'bottom': True, 'trunk': True, 'top': True},
                'llama2': {'bottom': True, 'trunk': False, 'top': False}
            }
        """
        info = {}
        for model_type in cls._registry:
            info[model_type] = {
                'bottom': 'bottom' in cls._registry[model_type],
                'trunk': 'trunk' in cls._registry[model_type],
                'top': 'top' in cls._registry[model_type],
                'complete': cls.is_complete(model_type)
            }
        return info
