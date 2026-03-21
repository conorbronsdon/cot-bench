# Running Open-Source Judges on Modular MAX

COT Bench uses two open-source models as judges, served locally via [Modular MAX](https://www.modular.com/max). This guide covers setup, hardware requirements, and troubleshooting.

## Why MAX?

MAX provides an OpenAI-compatible API endpoint for serving open-source models. This means:
- No code changes needed — judges call `localhost:8010/v1` just like any OpenAI endpoint
- Competitive inference performance (see [benchmarks](https://www.modular.com/blog/max-gpu-state-of-the-art-throughput-on-a-new-genai-platform))
- Hardware-agnostic — works on NVIDIA and AMD GPUs

## Hardware Requirements

### Minimum (sequential judge evaluation)

| Component | Requirement |
|-----------|-------------|
| GPU | 1× NVIDIA H100 80GB or A100 80GB |
| RAM | 64GB |
| Storage | 500GB SSD (for model weights) |

With a single GPU, you'll need to serve judges one at a time or use a smaller quantized model.

### Recommended (concurrent judges)

| Component | Requirement |
|-----------|-------------|
| GPU | 2× NVIDIA H100 80GB (one per judge model) |
| RAM | 128GB |
| Storage | 1TB SSD |

### Cloud Options

| Provider | Instance | GPUs | Approximate cost |
|----------|----------|------|-----------------|
| AWS | p5.2xlarge | 1× H100 | ~$10/hr |
| GCP | a3-highgpu-2g | 2× H100 | ~$20/hr |
| Lambda Labs | gpu_1x_h100 | 1× H100 | ~$3/hr |
| Vast.ai | H100 80GB | 1× H100 | ~$2-4/hr |

For weekly evaluation runs, a spot/preemptible instance running for 4-8 hours is typically sufficient.

## Installation

```bash
# Install MAX via pip
pip install modular --index https://whl.modular.com/nightly/simple/ --prerelease allow

# Verify installation
max --version

# Set HuggingFace token (needed to download model weights)
export HF_TOKEN="hf_..."
```

## Starting Judge Servers

### Manual start

```bash
# Terminal 1: Qwen3-235B judge
max serve --model Qwen/Qwen3-235B --port 8010

# Terminal 2: DeepSeek-V3 judge
max serve --model deepseek-ai/DeepSeek-V3-0324 --port 8011
```

### Using the helper script

```python
from infra.max_serve import start_all_judges, stop_all_judges

# Start both judges, one per GPU
processes = start_all_judges(gpu_mapping={"qwen3": "0", "deepseek": "1"})

# ... run evaluations ...

# Clean up
stop_all_judges(processes)
```

### Verify servers are running

```bash
# Check Qwen3
curl http://localhost:8010/v1/models

# Check DeepSeek
curl http://localhost:8011/v1/models
```

## Running Evaluation with MAX Judges

```bash
# All three judges (requires MAX servers running)
python -m scripts.run_eval --judges qwen3 deepseek opus

# Open-source judges only (no Anthropic API needed)
python -m scripts.run_eval --judges qwen3 deepseek

# Frontier judge only (no GPU needed)
python -m scripts.run_eval --judges opus
```

## Troubleshooting

### Model download is slow

First download takes 30-60 minutes for large models. Weights are cached in `~/.cache/huggingface/hub/` after first download.

### Out of memory

If a model doesn't fit on your GPU:
- Check that no other processes are using GPU memory: `nvidia-smi`
- Consider using a quantized version if available
- Use a single judge at a time instead of both concurrently

### Server won't start

```bash
# Check if port is already in use
lsof -i :8010

# Check MAX logs for errors
max serve --model Qwen/Qwen3-235B --port 8010 2>&1 | head -50
```

### Slow inference

- Ensure you're using the latest MAX version
- Check GPU utilization: `nvidia-smi -l 1`
- For judge workloads, batch size 1 is expected (each scenario judged independently)

## Running Without MAX

If you don't have GPU access, you can:

1. **Use frontier judge only**: `--judges opus` uses Claude Opus via API (no GPU needed)
2. **Use cloud-hosted open models**: Point judge endpoints to a hosted provider:

```python
# In eval/config.py, modify JUDGES:
JUDGES = {
    "qwen3": JudgeConfig(
        name="Qwen3-235B",
        model_id="qwen3-235b",
        provider="max",
        endpoint="https://your-hosted-endpoint/v1",  # Remote endpoint
    ),
    ...
}
```
