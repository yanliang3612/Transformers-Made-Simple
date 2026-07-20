# Conda 环境安装说明

这个仓库只需要一个很简单的 PyTorch 环境。你的机器是 NVIDIA H100，`nvidia-smi` 显示 CUDA Driver 支持到 `12.8`，这里按你的要求使用 PyTorch `2.6.0` + CUDA `12.4` wheel。

## 1. 创建环境

```bash
conda create -n transformers-simple python=3.11 -y
conda activate transformers-simple
```

## 2. 升级基础安装工具

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 3. 安装 PyTorch GPU 版

推荐安装固定版本：

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

说明：

- `cu124` 表示 PyTorch 自带 CUDA 12.4 运行时。
- 你的 driver 显示支持 CUDA 12.8，向下运行 CUDA 12.4 wheel 没问题。
- 你不需要自己安装完整 CUDA Toolkit。
- 只要 NVIDIA driver 足够新，PyTorch 自带的 CUDA runtime 就能用 GPU。

## 4. 安装本仓库依赖

当前仓库很简单，主要依赖就是 PyTorch：

```bash
pip install -r requirements.txt
```

如果上一步已经装过 `torch torchvision torchaudio`，这一步通常不会再做很多事情。

## 5. 检查 GPU 是否可用

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version in torch:", torch.version.cuda)

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    x = torch.randn(2, 3, device="cuda")
    print("test tensor device:", x.device)
PY
```

你期望看到类似：

```text
cuda available: True
gpu: NVIDIA H100 80GB HBM3
test tensor device: cuda:0
```

## 6. 运行训练脚本

先从最简单的模型跑起：

```bash
python -m train.train_encoder_only
```

然后可以逐个运行：

```bash
python -m train.train_decoder_only
python -m train.train_encoder_decoder
python -m train.train_vision_transformer
python -m train.train_efficient_transformer
python -m train.train_moe_transformer
python -m train.train_multimodal_transformer
python -m train.train_diffusion_transformer
python -m train.train_retrieval_transformer
python -m train.train_hybrid_transformer
```

## 7. 可选：让脚本只使用某一张 GPU

如果机器以后有多张 GPU，可以这样指定：

```bash
CUDA_VISIBLE_DEVICES=0 python -m train.train_decoder_only
```

## 8. 常见问题

### `torch.cuda.is_available()` 是 `False`

先确认驱动能看到 GPU：

```bash
nvidia-smi
```

如果 `nvidia-smi` 正常，再确认 PyTorch 是否装成了 CUDA 版：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
PY
```

如果 `torch.version.cuda` 是 `None`，说明装成了 CPU 版。可以重装：

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 显存占用太大

本仓库默认模型都很小，一般不会占很多显存。如果你自己把 `dim`、`num_layers`、`seq_len` 调大，可以先降低：

- `batch_size`
- `seq_len`
- `image_size`
- `dim`
- `num_layers`

## 9. 删除环境

不用时可以删除：

```bash
conda deactivate
conda env remove -n transformers-simple -y
```
