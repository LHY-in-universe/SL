"""
Model routing functionality.
"""

import logging
from typing import Optional, Dict
from enum import Enum

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"


class ModelRouter:
    """
    Routes requests to appropriate models.

    Supports multiple routing strategies:
    - Direct: Route to specific model by ID
    - Round-robin: Distribute requests evenly
    - Least-loaded: Route to model with fewest active requests
    """

    def __init__(self, model_manager):
        """
        Initialize router.

        Args:
            model_manager: ModelManager instance
        """
        self.model_manager = model_manager
        self.round_robin_index = 0

    def route_to_model(
        self,
        model_id: Optional[str] = None,
        strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN
    ) -> Optional[str]:
        """
        Route request to a model.

        Args:
            model_id: Specific model ID (overrides strategy)
            strategy: Routing strategy if model_id not specified

        Returns:
            Model ID to route to, or None if no models available
        """
        # Direct routing
        if model_id:
            if model_id in self.model_manager.models:
                return model_id
            else:
                logger.warning(f"Model {model_id} not found")
                return None

        # Get available models
        available_models = list(self.model_manager.models.keys())

        if not available_models:
            logger.warning("No models available for routing")
            return None

        # Apply routing strategy
        if strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_route(available_models)

        elif strategy == RoutingStrategy.LEAST_LOADED:
            return self._least_loaded_route(available_models)

        elif strategy == RoutingStrategy.RANDOM:
            import random
            return random.choice(available_models)

        else:
            logger.warning(f"Unknown strategy {strategy}, using round-robin")
            return self._round_robin_route(available_models)

    def _round_robin_route(self, model_ids: list) -> str:
        """Round-robin routing."""
        model_id = model_ids[self.round_robin_index % len(model_ids)]
        self.round_robin_index += 1
        return model_id

    def _least_loaded_route(self, model_ids: list) -> str:
        """Route to least loaded model (by request count)."""
        min_requests = float('inf')
        selected_model = model_ids[0]

        for model_id in model_ids:
            managed_model = self.model_manager.models[model_id]
            if managed_model.request_count < min_requests:
                min_requests = managed_model.request_count
                selected_model = model_id

        return selected_model

    def get_routing_info(self) -> Dict:
        """Get routing statistics."""
        models_info = self.model_manager.list_models()

        return {
            "available_models": len(models_info),
            "models": {
                info["model_id"]: {
                    "request_count": info["request_count"],
                    "status": info["status"]
                }
                for info in models_info
            }
        }
