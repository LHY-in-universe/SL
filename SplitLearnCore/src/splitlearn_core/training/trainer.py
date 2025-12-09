"""
SplitTrainer - 分布式训练编排器
协调客户端（Bottom + Top）和服务端（Trunk）的训练过程
"""

import logging
import time
import uuid
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置"""
    num_epochs: int = 3
    gradient_accumulation_steps: int = 1
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    max_grad_norm: float = 1.0
    output_dir: str = "./checkpoints"  # 检查点保存目录
    mixed_precision: bool = False  # 是否使用混合精度训练


class SplitTrainer:
    """
    分布式训练编排器

    协调客户端的 Bottom 和 Top 模型与服务端的 Trunk 模型进行训练。

    训练流程：
    1. 客户端 Bottom：前向传播，生成 hidden_1
    2. 服务端 Trunk：通过 gRPC 接收 hidden_1，前向传播，返回 hidden_2
    3. 客户端 Top：接收 hidden_2，前向传播，计算损失
    4. 客户端 Top：反向传播，计算 hidden_2 的梯度
    5. 服务端 Trunk：通过 gRPC 接收梯度，反向传播，返回 hidden_1 的梯度
    6. 客户端 Bottom：接收梯度，反向传播，更新参数
    7. 更新优化器

    Example:
        >>> from splitlearn_core import ModelFactory
        >>> from splitlearn_core.training import (
        ...     SplitLoraConfig, SplitOptimizer, SplitTrainer, TrainingConfig
        ... )
        >>> from splitlearn_comm import GRPCComputeClient
        >>>
        >>> # 创建模型（带 LoRA）
        >>> bottom, trunk, top = ModelFactory.create_split_models(
        ...     model_type='gpt2',
        ...     model_name_or_path='gpt2',
        ...     split_point_1=2,
        ...     split_point_2=10,
        ...     use_lora=True
        ... )
        >>>
        >>> # 连接到 Trunk 服务器
        >>> trunk_client = GRPCComputeClient("localhost:50051")
        >>> trunk_client.connect()
        >>>
        >>> # 创建优化器
        >>> optimizer = SplitOptimizer(bottom, top)
        >>>
        >>> # 创建训练器
        >>> trainer = SplitTrainer(
        ...     bottom_model=bottom,
        ...     top_model=top,
        ...     trunk_client=trunk_client,
        ...     optimizer=optimizer
        ... )
        >>>
        >>> # 训练
        >>> trainer.train(train_dataloader, eval_dataloader)
    """

    def __init__(
        self,
        bottom_model: nn.Module,
        top_model: nn.Module,
        trunk_client: 'GRPCComputeClient',
        optimizer: 'SplitOptimizer',
        config: Optional[TrainingConfig] = None,
        device: str = 'cpu',
        schedulers: Optional[Dict] = None,
    ):
        """
        Args:
            bottom_model: Bottom 模型（客户端）
            top_model: Top 模型（客户端）
            trunk_client: Trunk 服务器的 gRPC 客户端
            optimizer: 分布式优化器
            config: 训练配置
            device: 设备
            schedulers: 学习率调度器（可选）
        """
        self.bottom_model = bottom_model.to(device)
        self.top_model = top_model.to(device)
        self.trunk_client = trunk_client
        self.optimizer = optimizer
        self.config = config or TrainingConfig()
        self.device = device
        self.schedulers = schedulers or {}

        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')

        # 统计信息
        self.train_losses = []
        self.eval_losses = []
        self.step_times = []

        # 混合精度
        self.scaler = None
        if self.config.mixed_precision:
            try:
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
                logger.info("Mixed precision training enabled")
            except ImportError:
                logger.warning("Mixed precision not available, continuing with FP32")

        logger.info(f"SplitTrainer initialized on device={device}")

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        执行一个训练步骤

        Args:
            input_ids: 输入 token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 标签 [batch_size, seq_len]

        Returns:
            包含损失等信息的字典
        """
        step_start = time.time()

        # 将数据移到设备
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        try:
            # ============ 前向传播 ============

            # 1. Bottom 模型前向
            # Bottom 需要梯度（但在推理模式下运行以节省内存）
            self.bottom_model.eval()  # 使用 eval 模式（不影响 LoRA 训练）

            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    hidden_1 = self.bottom_model(input_ids)
            else:
                hidden_1 = self.bottom_model(input_ids)

            # 确保 hidden_1 需要梯度
            if not hidden_1.requires_grad:
                hidden_1.requires_grad_(True)
            hidden_1.retain_grad()

            # 2. Trunk 模型前向（通过 gRPC）
            forward_id = str(uuid.uuid4())

            # 发送到服务器（训练模式）
            # 注意：这里需要修改 trunk_client.compute 支持 training_mode 参数
            hidden_2 = self._trunk_forward(hidden_1, forward_id, attention_mask)

            # 确保 hidden_2 需要梯度
            if not hidden_2.requires_grad:
                hidden_2.requires_grad_(True)
            hidden_2.retain_grad()

            # 3. Top 模型前向
            self.top_model.train()  # Top 在训练模式

            # Top 模型需要返回损失
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    output = self.top_model(hidden_2, attention_mask=attention_mask, labels=labels)
            else:
                output = self.top_model(hidden_2, attention_mask=attention_mask, labels=labels)

            # 提取损失
            if hasattr(output, 'loss') and output.loss is not None:
                loss = output.loss
            else:
                # 如果模型不自动计算损失，手动计算
                logits = output.logits if hasattr(output, 'logits') else output
                loss = self._compute_loss(logits, labels)

            # 梯度累积
            loss = loss / self.config.gradient_accumulation_steps

            # ============ 反向传播 ============

            # 4. Top 模型反向传播
            if self.scaler is not None:
                self.scaler.scale(loss).backward(retain_graph=True)
            else:
                loss.backward(retain_graph=True)

            # 5. Trunk 模型反向传播（通过 gRPC）
            if hidden_2.grad is None:
                raise RuntimeError("hidden_2 gradient not computed")

            grad_hidden_1 = self._trunk_backward(hidden_2.grad, forward_id)

            # 6. Bottom 模型反向传播
            self.bottom_model.train()  # 切换到训练模式以更新参数
            hidden_1.backward(grad_hidden_1)

            # ============ 参数更新 ============

            # 每 N 步累积后更新
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    # 混合精度：unscale 后裁剪
                    self.scaler.unscale_(self.optimizer.optimizer_bottom)
                    self.scaler.unscale_(self.optimizer.optimizer_top)

                # 梯度裁剪
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.bottom_model.parameters(),
                        self.config.max_grad_norm
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.top_model.parameters(),
                        self.config.max_grad_norm
                    )

                # 优化器步骤
                if self.scaler is not None:
                    self.scaler.step(self.optimizer.optimizer_bottom)
                    self.scaler.step(self.optimizer.optimizer_top)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # 学习率调度
                for scheduler in self.schedulers.values():
                    scheduler.step()

                # 清零梯度
                self.optimizer.zero_grad()

            # 统计
            step_time = time.time() - step_start
            self.step_times.append(step_time)

            # 返回实际损失（未除以累积步数）
            actual_loss = loss.item() * self.config.gradient_accumulation_steps

            return {
                'loss': actual_loss,
                'step_time': step_time,
                'lr': self.optimizer.get_lr(),
            }

        except Exception as e:
            logger.error(f"Error in train_step: {e}", exc_info=True)
            raise

    def _trunk_forward(
        self,
        hidden_1: torch.Tensor,
        forward_id: str,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        通过 gRPC 调用 Trunk 前向传播

        Args:
            hidden_1: Bottom 的输出
            forward_id: 前向传播 ID
            attention_mask: 注意力掩码（目前不支持传输）

        Returns:
            Trunk 的输出
        """
        # 注意：当前 compute 方法需要扩展以支持 training_mode 和 forward_id
        # 这里假设已经扩展
        try:
            # 将张量移到 CPU 进行传输
            hidden_1_cpu = hidden_1.detach().cpu()

            # TODO: 扩展 compute 方法支持 training_mode 参数
            # 目前的 workaround：通过 metadata 传递
            hidden_2_cpu = self.trunk_client.compute(hidden_1_cpu)

            # 移回设备
            hidden_2 = hidden_2_cpu.to(self.device)

            return hidden_2

        except Exception as e:
            logger.error(f"Trunk forward failed: {e}")
            raise

    def _trunk_backward(
        self,
        grad_output: torch.Tensor,
        forward_id: str
    ) -> torch.Tensor:
        """
        通过 gRPC 调用 Trunk 反向传播

        Args:
            grad_output: Top 计算的梯度
            forward_id: 前向传播 ID

        Returns:
            Bottom 需要的梯度
        """
        try:
            # 将梯度移到 CPU 进行传输
            grad_output_cpu = grad_output.detach().cpu()

            # 调用反向传播 RPC
            grad_input_cpu = self.trunk_client.compute_backward(
                grad_output_cpu,
                forward_id
            )

            # 移回设备
            grad_input = grad_input_cpu.to(self.device)

            return grad_input

        except Exception as e:
            logger.error(f"Trunk backward failed: {e}")
            raise

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算交叉熵损失

        Args:
            logits: 模型输出 [batch_size, seq_len, vocab_size]
            labels: 标签 [batch_size, seq_len]

        Returns:
            损失标量
        """
        # Shift logits and labels for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # 计算损失
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits, shift_labels)

        return loss

    def evaluate(self, eval_dataloader) -> float:
        """
        评估模型

        Args:
            eval_dataloader: 评估数据加载器

        Returns:
            平均评估损失
        """
        self.bottom_model.eval()
        self.top_model.eval()

        total_loss = 0.0
        num_batches = 0

        logger.info("Starting evaluation...")

        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                labels = batch.get('labels', input_ids)

                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                try:
                    # 简化的评估：不需要梯度
                    hidden_1 = self.bottom_model(input_ids)
                    hidden_2 = self.trunk_client.compute(hidden_1.cpu()).to(self.device)
                    output = self.top_model(hidden_2, attention_mask=attention_mask, labels=labels)

                    if hasattr(output, 'loss') and output.loss is not None:
                        loss = output.loss
                    else:
                        logits = output.logits if hasattr(output, 'logits') else output
                        loss = self._compute_loss(logits, labels)

                    total_loss += loss.item()
                    num_batches += 1

                except Exception as e:
                    logger.warning(f"Evaluation batch failed: {e}")
                    continue

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        self.bottom_model.train()
        self.top_model.train()

        logger.info(f"Evaluation complete: avg_loss={avg_loss:.4f}")

        return avg_loss

    def train(
        self,
        train_dataloader,
        eval_dataloader: Optional = None,
        checkpoint_callback: Optional[Callable] = None
    ):
        """
        完整的训练循环

        Args:
            train_dataloader: 训练数据加载器
            eval_dataloader: 评估数据加载器（可选）
            checkpoint_callback: 检查点保存回调函数（可选）
        """
        logger.info("="*60)
        logger.info("Starting training...")
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"Device: {self.device}")
        logger.info("="*60)

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"{'='*60}")

            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_dataloader):
                # 训练步骤
                metrics = self.train_step(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask', None),
                    labels=batch.get('labels', batch['input_ids'])
                )

                epoch_loss += metrics['loss']
                num_batches += 1
                self.global_step += 1

                # 日志
                if self.global_step % self.config.log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    avg_time = sum(self.step_times[-10:]) / min(10, len(self.step_times))

                    logger.info(
                        f"Step {self.global_step} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Avg Loss: {avg_loss:.4f} | "
                        f"LR: {metrics['lr']} | "
                        f"Time: {avg_time:.2f}s/step"
                    )

                # 评估
                if eval_dataloader and self.global_step % self.config.eval_interval == 0:
                    eval_loss = self.evaluate(eval_dataloader)
                    self.eval_losses.append(eval_loss)

                    logger.info(f"Evaluation at step {self.global_step}: loss={eval_loss:.4f}")

                    # 保存最佳模型
                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        logger.info(f"New best eval loss: {eval_loss:.4f}")

                        if checkpoint_callback:
                            checkpoint_callback(self, is_best=True)

                # 定期保存检查点
                if self.global_step % self.config.save_interval == 0:
                    if checkpoint_callback:
                        checkpoint_callback(self, is_best=False)

            # Epoch 结束统计
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            self.train_losses.append(avg_epoch_loss)

            logger.info(f"\nEpoch {epoch + 1} complete:")
            logger.info(f"  Average loss: {avg_epoch_loss:.4f}")
            logger.info(f"  Total steps: {self.global_step}")

        logger.info("\n" + "="*60)
        logger.info("Training complete!")
        logger.info(f"Total steps: {self.global_step}")
        logger.info(f"Best eval loss: {self.best_eval_loss:.4f}")
        logger.info("="*60)

    def save_checkpoint(self, path: str):
        """
        保存训练检查点

        Args:
            path: 检查点保存路径
        """
        checkpoint = {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_eval_loss': self.best_eval_loss,
            'optimizer': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'config': self.config,
        }

        if self.schedulers:
            checkpoint['schedulers'] = {
                name: scheduler.state_dict()
                for name, scheduler in self.schedulers.items()
            }

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """
        加载训练检查点

        Args:
            path: 检查点路径
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        self.best_eval_loss = checkpoint['best_eval_loss']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_losses = checkpoint['train_losses']
        self.eval_losses = checkpoint['eval_losses']

        if 'schedulers' in checkpoint and self.schedulers:
            for name, state in checkpoint['schedulers'].items():
                if name in self.schedulers:
                    self.schedulers[name].load_state_dict(state)

        logger.info(f"Checkpoint loaded from {path}")
        logger.info(f"Resuming from step {self.global_step}, epoch {self.current_epoch}")
