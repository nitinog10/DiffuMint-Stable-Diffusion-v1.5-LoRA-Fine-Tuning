# 🎨 DiffuMint — Stable Diffusion v1.5 LoRA Fine-Tuning 
<p align="center">
  <img src="https://img.shields.io/badge/Model-Stable%20Diffusion%20v1.5-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Technique-LoRA%20(PEFT)-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Hardware-T4%20GPU%20%2F%20Colab-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Precision-fp16-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/UI-Gradio-yellow?style=for-the-badge" />
</p>

> A memory-efficient, end-to-end workflow to fine-tune **Stable Diffusion v1.5** using **LoRA (Low-Rank Adaptation)** on a Google Colab Free Tier T4 GPU — without crashing.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Why Stable Diffusion v1.5?](#-why-stable-diffusion-v15)
- [Project Workflow](#-project-workflow)
- [Dataset Structure](#-dataset-structure)
- [Key Hyperparameters](#-key-hyperparameters)
- [Memory Optimization](#-memory-optimization)
- [Gradio UI](#-gradio-ui)
- [Screenshots](#-screenshots)
- [Output Weights](#-output-weights)
- [Quick Start](#-quick-start)
- [Troubleshooting OOM Errors](#-troubleshooting-oom-errors)
- [Advanced Tips](#-advanced-tips)

---

## 🧠 Overview

**DiffuMint** demonstrates a production-ready, resource-conscious approach to fine-tuning Stable Diffusion v1.5 using LoRA adapters. By targeting only the attention layers of the UNet, **less than 1% of model parameters are actually trained**, making this viable on hardware with as little as 12 GB of VRAM.

Key highlights:
- ✅ Runs on **Google Colab Free Tier** (T4 GPU, 12–16 GB VRAM)
- ✅ Uses **LoRA** — trains only ~0.5–1% of total parameters
- ✅ **fp16 mixed precision** + **8-bit AdamW** optimizer
- ✅ **Gradient checkpointing** to trade compute for memory
- ✅ Interactive **Gradio UI** for real-time inference testing

---

## 🆚 Why Stable Diffusion v1.5?

| Property | SD v1.5 | SDXL |
|---|---|---|
| Parameters | ~860M | ~2.6B |
| Minimum VRAM (LoRA training) | ~10 GB | ~24 GB |
| Training Speed (T4 GPU) | Fast | Very Slow / OOM |
| Community LoRA Ecosystem | Mature | Growing |
| Colab Free Tier Compatible | ✅ Yes | ❌ No |

**SD v1.5** is the pragmatic choice for free-tier hardware:
- **~3× smaller** than SDXL — fits comfortably in 12–16 GB VRAM with LoRA
- **Faster iteration** — more training steps per hour means better experimentation
- **Battle-tested LoRA tooling** — `diffusers`, `peft`, and `bitsandbytes` all have first-class SD v1.5 support

---

## 🔄 Project Workflow

```
1. Environment Setup
   └── pip install diffusers transformers accelerate peft bitsandbytes gradio

2. Data Preparation
   └── Create dataset/images/ → resize all images to 512×512 (Lanczos)

3. Model Loading
   └── Load UNet, VAE, CLIP text encoder, tokenizer from runwayml/stable-diffusion-v1-5

4. LoRA Configuration
   └── Apply LoraConfig (r=8, alpha=32) to UNet attention layers via peft

5. Training Loop
   └── fp16 precision · 8-bit AdamW · gradient checkpointing · batch size 1
   └── 500 steps · gradient_accumulation_steps=4

6. Save Weights
   └── sd-lora-output/pytorch_lora_weights.safetensors

7. Inference & UI
   └── Load LoRA weights into StableDiffusionPipeline → Gradio interface
```

---

## 📁 Dataset Structure

Place your training images in the following structure. All images are automatically resized to **512×512** using high-quality **Lanczos resampling**:

```
/content/
└── dataset/
    └── images/
        ├── sample_0.jpg   (512×512)
        ├── sample_1.jpg   (512×512)
        └── sample_2.jpg   (512×512)
```

> **Tip:** More training images → better generalization. Even 10–50 domain-specific images can yield compelling results with LoRA.

---

## ⚙️ Key Hyperparameters

| Parameter | Value | Reason |
|---|---|---|
| **Base Model** | `runwayml/stable-diffusion-v1-5` | Memory-efficient, well-supported |
| **Resolution** | `512 × 512` | Native SD v1.5 resolution |
| **Learning Rate** | `1e-4` | Standard for LoRA fine-tuning |
| **Training Steps** | `500` | Sufficient for small custom datasets |
| **LoRA Rank (r)** | `8` | Balances expressiveness vs. parameter count |
| **LoRA Alpha (α)** | `32` | Effective scaling factor = α/r = 4 |
| **Batch Size** | `1` | Minimizes VRAM per step |
| **Gradient Accum. Steps** | `4` | Effective batch size of 4 |
| **Precision** | `fp16` | Saves ~50% VRAM vs. fp32 |
| **Optimizer** | `AdamW8bit` | Reduces optimizer state by ~4× |
| **Gradient Checkpointing** | `True` | Trades compute for memory |
| **LoRA Target Modules** | `to_q, to_k, to_v, to_out.0` | UNet cross/self-attention layers |

---

## 💾 Memory Optimization

DiffuMint stacks multiple memory-saving techniques to stay within the T4 GPU's VRAM budget:

### Technique Stack

```
┌──────────────────────────────────────────────────────────────┐
│  fp16 Mixed Precision        → ~50% VRAM reduction on weights │
│  8-bit AdamW Optimizer       → ~75% reduction on optim states │
│  Gradient Checkpointing      → trades FLOPs for activation RAM│
│  LoRA (r=8, <1% params)      → only adapter weights computed  │
│  Batch Size = 1              → minimum per-step memory        │
│  Gradient Accumulation = 4   → effective batch w/o extra VRAM │
└──────────────────────────────────────────────────────────────┘
```

### Estimated VRAM Budget

| Component | VRAM Usage (approx.) |
|---|---|
| UNet (fp16) | ~3.5 GB |
| VAE (fp16) | ~0.5 GB |
| CLIP Text Encoder | ~0.5 GB |
| Activations + Gradients | ~3–4 GB |
| Optimizer States (8-bit) | ~1 GB |
| **Total** | **~9–10 GB** |

---

## 🖥️ Gradio UI

After training, DiffuMint launches an interactive **Gradio web interface** that lets you test your fine-tuned model in real time — no extra code needed.

### Features

| Control | Description |
|---|---|
| **Prompt** | Text description of the image to generate |
| **Negative Prompt** | What to exclude (e.g., `low quality, blurry`) |
| **Steps** | Number of denoising steps (1–50; default: 30) |
| **Guidance Scale** | Prompt adherence strength (1–15; default: 7.5) |
| **Output** | Generated image displayed inline |

### Usage

```python
# After training completes, simply run:
launch_ui()
# → Opens at http://localhost:7860 (or public share URL via share=True)
```

The UI automatically:
1. Loads your fine-tuned LoRA weights from `sd-lora-output/`
2. Applies `attention_slicing` for efficient inference
3. Provides a shareable public URL (via `share=True`) for remote testing

---

## 📸 Screenshots

### Training & Inference in Action

<p align="center">
  <img src="screenshots/Screenshot 2026-03-02 225146.png" alt="DiffuMint Training Run — Loss and VRAM monitoring" width="100%" />
  <br/>
  <em>Training run — step-by-step loss values and VRAM usage monitoring on T4 GPU</em>
</p>

<p align="center">
  <img src="screenshots/Screenshot 2026-03-02 225201.png" alt="DiffuMint Gradio UI — Interactive Image Generation" width="100%" />
  <br/>
  <em>Gradio UI — interactive prompt-based image generation with fine-tuned LoRA weights</em>
</p>

---

## 📦 Output Weights

After training completes, the fine-tuned LoRA adapters are saved in:

```
sd-lora-output/
└── pytorch_lora_weights.safetensors
```

### Loading the Weights

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe.to("cuda")

# Load your fine-tuned LoRA adapters
pipe.load_lora_weights("sd-lora-output")

# Generate an image
image = pipe("your prompt here", num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("output.png")
```

> The `.safetensors` format is preferred over `.bin` for security — it cannot execute arbitrary code on load.

---

## 🚀 Quick Start

### 1. Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SZxCfzHCeVPh2YBUL6oVF8QHga0HsafN)

### 2. Install Dependencies

```bash
pip install diffusers transformers accelerate peft bitsandbytes gradio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Prepare Your Dataset

```python
import os
from PIL import Image

os.makedirs("dataset/images", exist_ok=True)
# Copy your training images into dataset/images/
# They will be auto-resized to 512×512
```

### 4. Run Training

```bash
python stablediffusionmain.py
# or open DiffuMint_Stable_Diffusion_v1_5_LoRA_Fine_Tuning.ipynb in Colab
```

### 5. Launch the UI

The Gradio UI launches automatically at the end of the notebook / script.

---

## 🔧 Troubleshooting OOM Errors

If you encounter **CUDA Out of Memory** errors, try these steps in order:

```
Step 1 → torch.cuda.empty_cache() before training loop
Step 2 → Reduce LoRA rank: r=8 → r=4 → r=2
Step 3 → Lower resolution: 512 → 448 → 384
Step 4 → Increase gradient_accumulation_steps: 4 → 8
Step 5 → Enable VAE slicing: vae.enable_slicing()
Step 6 → Enable xformers: unet.enable_xformers_memory_efficient_attention()
```

---

## 💡 Advanced Tips

### Reduce LoRA Rank for Tighter Memory

```python
lora_config = LoraConfig(
    r=4,         # reduced from 8
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
)
```

### Enable xFormers for Attention Efficiency

```python
# Install: pip install xformers
unet.enable_xformers_memory_efficient_attention()
```

### VAE Slicing for Decode Safety

```python
vae.enable_slicing()   # decode large batches tile-by-tile
vae.enable_tiling()    # process high-res images in tiles
```

### Monitor VRAM in Real Time

```python
import torch
print(f"VRAM Used: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
# Keep below 90% of total capacity to avoid sudden spikes
```

---


---

## 🙏 Acknowledgements

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) — the backbone of this pipeline
- [PEFT](https://github.com/huggingface/peft) — LoRA implementation
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — 8-bit optimizer magic
- [Gradio](https://gradio.app/) — interactive UI without writing frontend code
- [RunwayML](https://huggingface.co/runwayml/stable-diffusion-v1-5) — Stable Diffusion v1.5 weights
