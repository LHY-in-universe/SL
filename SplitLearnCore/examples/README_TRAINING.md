# Distributed LoRA Training Examples

This directory contains examples for distributed LoRA fine-tuning using SplitLearn.

## Architecture

```
┌─────────────────┐          gRPC           ┌─────────────────┐
│  Client Side    │  ◄────────────────►     │  Server Side    │
│                 │                          │                 │
│  Bottom Model   │  ─── hidden states ───► │  Trunk Model    │
│  (LoRA)         │  ◄─── gradients ────    │  (LoRA)         │
│                 │                          │                 │
│  Top Model      │                          │                 │
│  (LoRA)         │                          │                 │
│  + Loss         │                          │                 │
└─────────────────┘                          └─────────────────┘
```

- **Client**: Runs Bottom model + Top model with LoRA adapters
- **Server**: Runs Trunk model (middle layers) with LoRA adapters
- **Communication**: gRPC for forward/backward tensor transmission

## Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers datasets peft splitlearn-core splitlearn-comm
```

### 2. Start Trunk Server

On the server machine (or localhost for testing):

```bash
# GPT-2 example
python examples/start_trunk_server.py \
    --model-type gpt2 \
    --model-name gpt2 \
    --split-point-1 0 \
    --split-point-2 6 \
    --use-lora \
    --lora-rank 8 \
    --device cuda \
    --port 50051

# Qwen2 example
python examples/start_trunk_server.py \
    --model-type qwen2 \
    --model-name Qwen/Qwen2-0.5B \
    --split-point-1 0 \
    --split-point-2 14 \
    --use-lora \
    --lora-rank 8 \
    --device cuda \
    --port 50051
```

**Important**: Wait for the server to fully load the model before starting training.

### 3. Run Training

On the client machine:

```bash
# GPT-2 example
python examples/train_lora_distributed.py \
    --model-type gpt2 \
    --model-name gpt2 \
    --split-point-1 0 \
    --split-point-2 6 \
    --server-host localhost \
    --server-port 50051 \
    --lora-rank 8 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --num-epochs 3 \
    --dataset-name wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --max-train-samples 1000 \
    --device cuda \
    --output-dir ./checkpoints/gpt2_lora

# Qwen2 example
python examples/train_lora_distributed.py \
    --model-type qwen2 \
    --model-name Qwen/Qwen2-0.5B \
    --split-point-1 0 \
    --split-point-2 14 \
    --server-host localhost \
    --server-port 50051 \
    --lora-rank 8 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --num-epochs 3 \
    --device cuda \
    --output-dir ./checkpoints/qwen2_lora
```

## Examples

### Example 1: GPT-2 on WikiText (Small Model, Fast Training)

**Server**:
```bash
python examples/start_trunk_server.py \
    --model-type gpt2 \
    --model-name gpt2 \
    --split-point-1 0 \
    --split-point-2 6 \
    --use-lora \
    --lora-rank 8 \
    --device cuda
```

**Client**:
```bash
python examples/train_lora_distributed.py \
    --model-type gpt2 \
    --model-name gpt2 \
    --split-point-1 0 \
    --split-point-2 6 \
    --lora-rank 8 \
    --batch-size 8 \
    --learning-rate 2e-4 \
    --num-epochs 3 \
    --max-train-samples 5000 \
    --gradient-accumulation-steps 2 \
    --device cuda
```

**Expected Results**:
- Training time: ~10 minutes on single GPU
- Initial loss: ~4.0
- Final loss: ~2.5-3.0
- LoRA parameters: ~300K (vs 124M base model)

### Example 2: Qwen2-0.5B on WikiText (Medium Model)

**Server**:
```bash
python examples/start_trunk_server.py \
    --model-type qwen2 \
    --model-name Qwen/Qwen2-0.5B \
    --split-point-1 0 \
    --split-point-2 14 \
    --use-lora \
    --lora-rank 16 \
    --device cuda
```

**Client**:
```bash
python examples/train_lora_distributed.py \
    --model-type qwen2 \
    --model-name Qwen/Qwen2-0.5B \
    --split-point-1 0 \
    --split-point-2 14 \
    --lora-rank 16 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --num-epochs 3 \
    --max-train-samples 10000 \
    --gradient-accumulation-steps 4 \
    --mixed-precision \
    --device cuda
```

**Expected Results**:
- Training time: ~30 minutes on single GPU
- Initial loss: ~3.5
- Final loss: ~2.0-2.5
- LoRA parameters: ~2M (vs 500M base model)

### Example 3: Remote Server Training

If the Trunk server is on a different machine:

**Server** (at 192.168.1.100):
```bash
python examples/start_trunk_server.py \
    --model-type qwen2 \
    --model-name Qwen/Qwen2-0.5B \
    --split-point-1 0 \
    --split-point-2 14 \
    --host 0.0.0.0 \
    --port 50051 \
    --use-lora \
    --device cuda
```

**Client** (on your local machine):
```bash
python examples/train_lora_distributed.py \
    --model-type qwen2 \
    --model-name Qwen/Qwen2-0.5B \
    --split-point-1 0 \
    --split-point-2 14 \
    --server-host 192.168.1.100 \
    --server-port 50051 \
    --device cuda
```

**Note**: Ensure firewall allows port 50051, or configure a different port.

## Configuration Guide

### Split Points

Choose split points based on your model architecture:

| Model | Total Layers | Recommended Split Points |
|-------|--------------|-------------------------|
| GPT-2 | 12 | `split_point_1=0, split_point_2=6` |
| GPT-2 Medium | 24 | `split_point_1=0, split_point_2=12` |
| Qwen2-0.5B | 24 | `split_point_1=0, split_point_2=14` |
| Qwen2-1.5B | 28 | `split_point_1=0, split_point_2=16` |
| Qwen3-VL-2B | 28 | `split_point_1=0, split_point_2=16` |

**Guidelines**:
- `split_point_1`: Always 0 (full Bottom model on client)
- `split_point_2`: Typically 50-70% of total layers (balance client/server compute)

### LoRA Parameters

| Parameter | Recommended Range | Notes |
|-----------|------------------|-------|
| `lora_rank` | 4-16 | Higher = more parameters, better quality |
| `lora_alpha` | rank * 2 | Scaling factor, typically 2x rank |
| `lora_dropout` | 0.05-0.1 | Regularization |

**Guidelines**:
- Small models (< 500M): rank=8
- Medium models (500M-2B): rank=16
- Large models (> 2B): rank=32

### Training Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `batch_size` | Batch size per device | 4-8 |
| `learning_rate` | Learning rate | 1e-4 to 2e-4 |
| `gradient_accumulation_steps` | Accumulate gradients before update | 2-4 |
| `max_grad_norm` | Gradient clipping | 1.0 |
| `warmup_steps` | LR warmup steps | 100 |
| `max_seq_length` | Maximum sequence length | 512-2048 |

**Guidelines**:
- Effective batch size = `batch_size * gradient_accumulation_steps`
- Larger effective batch size → more stable training
- Use mixed precision (`--mixed-precision`) to save memory

## Checkpointing

Checkpoints are saved to `--output-dir` every `--save-interval` steps.

### Checkpoint Structure

```
checkpoints/
├── checkpoint_step_500/
│   ├── bottom/
│   │   ├── model.pt
│   │   └── lora_adapters/
│   ├── top/
│   │   ├── model.pt
│   │   └── lora_adapters/
│   └── training_state.pt
└── checkpoint_step_1000/
    └── ...
```

### Resume Training

To resume from a checkpoint:

```python
# In your training script
from splitlearn_core.training import SplitTrainer

trainer = SplitTrainer(...)
trainer.load_checkpoint("checkpoints/checkpoint_step_500")
trainer.train(...)  # Continue training
```

## Monitoring

### Logging

Training logs include:
- **Loss**: Cross-entropy loss
- **Learning rate**: Current LR for Bottom and Top
- **Gradient norm**: Average gradient magnitude
- **Throughput**: Steps per second

Example log output:
```
2025-12-09 10:30:45 - INFO - Step 100 | Loss: 3.245 | LR: 0.00010 | Grad Norm: 0.523 | Steps/s: 2.4
2025-12-09 10:31:15 - INFO - Step 200 | Loss: 2.987 | LR: 0.00010 | Grad Norm: 0.445 | Steps/s: 2.5
2025-12-09 10:31:45 - INFO - Evaluation | Val Loss: 2.654 | Best: 2.654
```

### TensorBoard (Optional)

To enable TensorBoard logging:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./runs/experiment_1')

# In training loop
writer.add_scalar('Loss/train', loss, global_step)
writer.add_scalar('LR/bottom', optimizer.get_lr()['bottom'], global_step)
```

Then visualize:
```bash
tensorboard --logdir ./runs
```

## Troubleshooting

### Issue: Connection Refused

**Symptom**: `Failed to connect to Trunk server: Connection refused`

**Solutions**:
1. Ensure server is running and fully loaded
2. Check firewall allows the port
3. Verify correct host:port in client
4. Test with `telnet <host> <port>`

### Issue: CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce `--batch-size`
2. Increase `--gradient-accumulation-steps`
3. Enable `--mixed-precision`
4. Reduce `--max-seq-length`
5. Use smaller LoRA rank

### Issue: Loss Not Decreasing

**Symptom**: Training loss plateaus or increases

**Solutions**:
1. Lower learning rate (try 5e-5 instead of 1e-4)
2. Increase warmup steps
3. Check data quality (no corruption)
4. Verify gradient flow (check `grad_norm` logs)
5. Try different LoRA rank

### Issue: Training Too Slow

**Symptom**: < 1 step/second

**Solutions**:
1. Enable `--mixed-precision`
2. Reduce `--max-seq-length`
3. Use larger `--batch-size` if memory allows
4. Check network latency between client and server
5. Profile with `torch.profiler`

## Advanced Usage

### Custom Dataset

To use a custom dataset:

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.encodings['input_ids'][idx]
        }

# Use in training
dataset = MyDataset(my_texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=4)
trainer.train(dataloader)
```

### Multi-GPU Training

To train with multiple GPUs on the client side:

```python
# Wrap Bottom and Top models with DataParallel
bottom_model = torch.nn.DataParallel(bottom_model)
top_model = torch.nn.DataParallel(top_model)

# Rest of training code remains the same
```

**Note**: Trunk model on server can also use DataParallel independently.

### Inference with Fine-tuned Model

After training, use the fine-tuned model for inference:

```python
from splitlearn_core.training import merge_lora_weights

# Load checkpoint
trainer.load_checkpoint("checkpoints/checkpoint_step_1000")

# Merge LoRA weights into base model (optional, for faster inference)
bottom_merged = merge_lora_weights(bottom_model)
top_merged = merge_lora_weights(top_model)

# Inference
with torch.no_grad():
    hidden_1 = bottom_merged(input_ids)
    hidden_2 = trunk_client.compute(hidden_1, training_mode=False)
    output = top_merged(hidden_2)
```

## Performance Benchmarks

Hardware: NVIDIA A100 40GB, Batch size 4, Seq length 512

| Model | LoRA Rank | Steps/sec | Memory (Client) | Memory (Server) |
|-------|-----------|-----------|-----------------|-----------------|
| GPT-2 | 8 | 5.2 | 2 GB | 1.5 GB |
| Qwen2-0.5B | 8 | 3.1 | 4 GB | 3 GB |
| Qwen2-0.5B | 16 | 2.8 | 4.5 GB | 3.5 GB |
| Qwen3-VL-2B | 16 | 1.2 | 8 GB | 12 GB |

## References

- [SplitLearn Documentation](../docs/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
- [Transformers Library](https://github.com/huggingface/transformers)

## Support

For issues or questions:
- GitHub Issues: [SplitLearn Issues](https://github.com/yourusername/splitlearn/issues)
- Documentation: [Full Documentation](../docs/)
