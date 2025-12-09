"""
Checkpoint Manager for Distributed Training

Handles saving and loading checkpoints for split models with LoRA adapters.
Supports separate checkpoints for Bottom, Trunk, and Top models with metadata.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Check if PEFT is available
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not available, some checkpoint features will be limited")


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint"""
    # Checkpoint info
    checkpoint_version: str = "1.0"
    created_at: str = ""
    component: str = ""  # "bottom", "trunk", or "top"

    # Training state
    global_step: int = 0
    epoch: int = 0
    best_eval_loss: float = float('inf')

    # Model info
    model_type: str = ""
    model_name: str = ""
    split_point_1: int = 0
    split_point_2: int = 0

    # LoRA info
    use_lora: bool = False
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    lora_target_modules: Optional[List[str]] = None

    # Training config
    learning_rate: float = 0.0
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Metrics
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None

    # Custom metadata
    custom: Dict[str, Any] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if self.custom is None:
            self.custom = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create from dictionary"""
        return cls(**data)

    def save(self, path: Path):
        """Save metadata to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'CheckpointMetadata':
        """Load metadata from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class CheckpointManager:
    """
    Manages checkpoints for distributed split learning with LoRA

    Handles saving and loading of:
    - Model state dicts (full or LoRA-only)
    - Optimizer states
    - Training states
    - Metadata

    Example:
        >>> manager = CheckpointManager(output_dir="./checkpoints")
        >>>
        >>> # Save checkpoint
        >>> manager.save_checkpoint(
        ...     bottom_model=bottom_model,
        ...     top_model=top_model,
        ...     optimizer=optimizer,
        ...     global_step=1000,
        ...     metadata=metadata
        ... )
        >>>
        >>> # Load checkpoint
        >>> checkpoint = manager.load_checkpoint(
        ...     checkpoint_path="./checkpoints/checkpoint_step_1000",
        ...     bottom_model=bottom_model,
        ...     top_model=top_model,
        ...     optimizer=optimizer
        ... )
    """

    def __init__(
        self,
        output_dir: str = "./checkpoints",
        save_lora_only: bool = True,
        keep_last_n: Optional[int] = 5,
        save_optimizer_state: bool = True
    ):
        """
        Args:
            output_dir: Directory to save checkpoints
            save_lora_only: If True, only save LoRA adapters (much smaller)
            keep_last_n: Keep only last N checkpoints (None = keep all)
            save_optimizer_state: Whether to save optimizer state
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_lora_only = save_lora_only
        self.keep_last_n = keep_last_n
        self.save_optimizer_state = save_optimizer_state

        logger.info(f"CheckpointManager initialized: output_dir={output_dir}")
        logger.info(f"  save_lora_only={save_lora_only}, keep_last_n={keep_last_n}")

    def save_checkpoint(
        self,
        bottom_model: Optional[nn.Module] = None,
        top_model: Optional[nn.Module] = None,
        trunk_model: Optional[nn.Module] = None,
        optimizer: Optional[Any] = None,
        global_step: int = 0,
        epoch: int = 0,
        metadata: Optional[CheckpointMetadata] = None,
        checkpoint_name: Optional[str] = None
    ) -> Path:
        """
        Save checkpoint for split models

        Args:
            bottom_model: Bottom model (client-side)
            top_model: Top model (client-side)
            trunk_model: Trunk model (server-side)
            optimizer: SplitOptimizer or standard optimizer
            global_step: Current training step
            epoch: Current epoch
            metadata: Checkpoint metadata
            checkpoint_name: Custom checkpoint name (default: checkpoint_step_{step})

        Returns:
            Path to saved checkpoint directory
        """
        # Create checkpoint directory
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_step_{global_step}"

        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving checkpoint to {checkpoint_dir}")

        # Update metadata
        if metadata is None:
            metadata = CheckpointMetadata()

        metadata.global_step = global_step
        metadata.epoch = epoch

        # Save Bottom model
        if bottom_model is not None:
            self._save_component(
                model=bottom_model,
                component_dir=checkpoint_dir / "bottom",
                component_name="bottom",
                metadata=metadata
            )

        # Save Trunk model
        if trunk_model is not None:
            self._save_component(
                model=trunk_model,
                component_dir=checkpoint_dir / "trunk",
                component_name="trunk",
                metadata=metadata
            )

        # Save Top model
        if top_model is not None:
            self._save_component(
                model=top_model,
                component_dir=checkpoint_dir / "top",
                component_name="top",
                metadata=metadata
            )

        # Save optimizer state
        if optimizer is not None and self.save_optimizer_state:
            self._save_optimizer(optimizer, checkpoint_dir)

        # Save training state
        training_state = {
            'global_step': global_step,
            'epoch': epoch,
            'metadata': metadata.to_dict()
        }

        training_state_path = checkpoint_dir / "training_state.pt"
        torch.save(training_state, training_state_path)
        logger.info(f"Saved training state to {training_state_path}")

        # Cleanup old checkpoints
        if self.keep_last_n is not None:
            self._cleanup_old_checkpoints()

        logger.info(f"✓ Checkpoint saved successfully: {checkpoint_dir}")
        return checkpoint_dir

    def _save_component(
        self,
        model: nn.Module,
        component_dir: Path,
        component_name: str,
        metadata: CheckpointMetadata
    ):
        """Save a single model component (Bottom/Trunk/Top)"""
        component_dir.mkdir(parents=True, exist_ok=True)

        # Check if model has LoRA adapters
        is_peft_model = PEFT_AVAILABLE and isinstance(model, PeftModel)

        if is_peft_model and self.save_lora_only:
            # Save only LoRA adapters (PEFT format)
            lora_dir = component_dir / "lora_adapters"
            model.save_pretrained(str(lora_dir))
            logger.info(f"Saved {component_name} LoRA adapters to {lora_dir}")

            # Save metadata indicating LoRA usage
            component_metadata = CheckpointMetadata(
                component=component_name,
                global_step=metadata.global_step,
                epoch=metadata.epoch,
                use_lora=True,
                model_type=metadata.model_type,
                model_name=metadata.model_name
            )

        else:
            # Save full model state dict
            model_path = component_dir / "model.pt"
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved {component_name} full model to {model_path}")

            # Save metadata
            component_metadata = CheckpointMetadata(
                component=component_name,
                global_step=metadata.global_step,
                epoch=metadata.epoch,
                use_lora=False,
                model_type=metadata.model_type,
                model_name=metadata.model_name
            )

        # Save component metadata
        metadata_path = component_dir / "metadata.json"
        component_metadata.save(metadata_path)

    def _save_optimizer(self, optimizer: Any, checkpoint_dir: Path):
        """Save optimizer state"""
        optimizer_path = checkpoint_dir / "optimizer.pt"

        # Handle SplitOptimizer
        if hasattr(optimizer, 'state_dict'):
            state_dict = optimizer.state_dict()
            torch.save(state_dict, optimizer_path)
            logger.info(f"Saved optimizer state to {optimizer_path}")
        else:
            logger.warning("Optimizer does not have state_dict(), skipping")

    def load_checkpoint(
        self,
        checkpoint_path: str,
        bottom_model: Optional[nn.Module] = None,
        top_model: Optional[nn.Module] = None,
        trunk_model: Optional[nn.Module] = None,
        optimizer: Optional[Any] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load checkpoint for split models

        Args:
            checkpoint_path: Path to checkpoint directory
            bottom_model: Bottom model to load into
            top_model: Top model to load into
            trunk_model: Trunk model to load into
            optimizer: Optimizer to load state into
            strict: Whether to strictly enforce state dict matching

        Returns:
            Dictionary with loaded training state

        Example:
            >>> checkpoint = manager.load_checkpoint(
            ...     checkpoint_path="./checkpoints/checkpoint_step_1000",
            ...     bottom_model=bottom_model,
            ...     top_model=top_model,
            ...     optimizer=optimizer
            ... )
            >>> print(f"Resumed from step {checkpoint['global_step']}")
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Load Bottom model
        if bottom_model is not None:
            bottom_dir = checkpoint_path / "bottom"
            if bottom_dir.exists():
                self._load_component(bottom_model, bottom_dir, "bottom", strict)
            else:
                logger.warning("Bottom checkpoint not found, skipping")

        # Load Trunk model
        if trunk_model is not None:
            trunk_dir = checkpoint_path / "trunk"
            if trunk_dir.exists():
                self._load_component(trunk_model, trunk_dir, "trunk", strict)
            else:
                logger.warning("Trunk checkpoint not found, skipping")

        # Load Top model
        if top_model is not None:
            top_dir = checkpoint_path / "top"
            if top_dir.exists():
                self._load_component(top_model, top_dir, "top", strict)
            else:
                logger.warning("Top checkpoint not found, skipping")

        # Load optimizer
        if optimizer is not None:
            optimizer_path = checkpoint_path / "optimizer.pt"
            if optimizer_path.exists():
                self._load_optimizer(optimizer, optimizer_path)
            else:
                logger.warning("Optimizer checkpoint not found, skipping")

        # Load training state
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            training_state = torch.load(training_state_path)
            logger.info(f"✓ Loaded training state: step={training_state['global_step']}")
            return training_state
        else:
            logger.warning("Training state not found, returning empty state")
            return {'global_step': 0, 'epoch': 0, 'metadata': {}}

    def _load_component(
        self,
        model: nn.Module,
        component_dir: Path,
        component_name: str,
        strict: bool
    ):
        """Load a single model component"""
        # Load metadata
        metadata_path = component_dir / "metadata.json"
        if metadata_path.exists():
            metadata = CheckpointMetadata.load(metadata_path)
            logger.info(f"Loading {component_name}: step={metadata.global_step}, "
                       f"use_lora={metadata.use_lora}")

        # Check for LoRA adapters
        lora_dir = component_dir / "lora_adapters"
        is_peft_model = PEFT_AVAILABLE and isinstance(model, PeftModel)

        if lora_dir.exists() and is_peft_model:
            # Load LoRA adapters using PEFT
            from peft import set_peft_model_state_dict
            import os

            adapter_config_path = lora_dir / "adapter_config.json"
            adapter_model_path = lora_dir / "adapter_model.bin"

            if adapter_model_path.exists():
                adapter_state_dict = torch.load(adapter_model_path)
                set_peft_model_state_dict(model, adapter_state_dict)
                logger.info(f"✓ Loaded {component_name} LoRA adapters")
            else:
                logger.warning(f"LoRA checkpoint found but adapter_model.bin missing")

        else:
            # Load full model state dict
            model_path = component_dir / "model.pt"
            if model_path.exists():
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict, strict=strict)
                logger.info(f"✓ Loaded {component_name} full model")
            else:
                logger.warning(f"Model checkpoint not found at {model_path}")

    def _load_optimizer(self, optimizer: Any, optimizer_path: Path):
        """Load optimizer state"""
        if hasattr(optimizer, 'load_state_dict'):
            state_dict = torch.load(optimizer_path)
            optimizer.load_state_dict(state_dict)
            logger.info(f"✓ Loaded optimizer state from {optimizer_path}")
        else:
            logger.warning("Optimizer does not have load_state_dict(), skipping")

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only last N"""
        # Get all checkpoint directories
        checkpoints = sorted(
            [d for d in self.output_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime
        )

        # Remove oldest checkpoints
        if len(checkpoints) > self.keep_last_n:
            to_remove = checkpoints[:-self.keep_last_n]
            for checkpoint_dir in to_remove:
                logger.info(f"Removing old checkpoint: {checkpoint_dir}")
                shutil.rmtree(checkpoint_dir)

    def list_checkpoints(self) -> List[Tuple[Path, CheckpointMetadata]]:
        """
        List all available checkpoints with metadata

        Returns:
            List of (checkpoint_path, metadata) tuples sorted by step
        """
        checkpoints = []

        for checkpoint_dir in self.output_dir.iterdir():
            if not checkpoint_dir.is_dir():
                continue

            # Try to load training state
            training_state_path = checkpoint_dir / "training_state.pt"
            if training_state_path.exists():
                training_state = torch.load(training_state_path)
                metadata_dict = training_state.get('metadata', {})
                metadata = CheckpointMetadata.from_dict(metadata_dict)
                checkpoints.append((checkpoint_dir, metadata))

        # Sort by global step
        checkpoints.sort(key=lambda x: x[1].global_step)

        return checkpoints

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to the latest checkpoint"""
        checkpoints = self.list_checkpoints()

        if not checkpoints:
            return None

        return checkpoints[-1][0]

    def get_best_checkpoint(self, metric: str = 'eval_loss') -> Optional[Path]:
        """
        Get path to the best checkpoint based on a metric

        Args:
            metric: Metric to compare ('eval_loss', 'train_loss', etc.)

        Returns:
            Path to best checkpoint or None
        """
        checkpoints = self.list_checkpoints()

        if not checkpoints:
            return None

        # Find checkpoint with best metric
        best_checkpoint = None
        best_value = float('inf')

        for checkpoint_path, metadata in checkpoints:
            value = getattr(metadata, metric, None)
            if value is not None and value < best_value:
                best_value = value
                best_checkpoint = checkpoint_path

        return best_checkpoint

    def __repr__(self) -> str:
        num_checkpoints = len([d for d in self.output_dir.iterdir() if d.is_dir()])
        return (
            f"CheckpointManager(output_dir={self.output_dir}, "
            f"checkpoints={num_checkpoints}, "
            f"save_lora_only={self.save_lora_only})"
        )
