# GPU使用ガイド - GPU Usage Guide

## 概要 / Overview

このガイドでは、オセロAIの学習をGPUで高速化する方法を説明します。
GPUを使用することで、学習速度が約3-5倍向上します。

This guide explains how to accelerate Othello AI training using GPU.
Using GPU can improve training speed by approximately 3-5x.

---

## 必要な環境 / Requirements

### ハードウェア / Hardware
- **NVIDIA GPU**: CUDA対応のGPU (CUDA-compatible GPU)
- **推奨**: GeForce GTX 1060以上 (Recommended: GeForce GTX 1060 or better)
- **最低**: 4GB以上のGPUメモリ (Minimum: 4GB+ GPU memory)

### ソフトウェア / Software
1. **CUDAドライバ** (CUDA Drivers)
   - CUDA 11.8以上を推奨 (CUDA 11.8+ recommended)
   - [CUDA Toolkit ダウンロード](https://developer.nvidia.com/cuda-downloads)

2. **PyTorch (GPU版)** (PyTorch GPU version)
   ```bash
   # CUDA 11.8の場合 (For CUDA 11.8)
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   
   # CUDA 12.1の場合 (For CUDA 12.1)
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

---

## GPU動作確認 / Verify GPU Setup

GPUが正しく認識されているか確認:
Check if GPU is properly recognized:

```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**期待される出力 (Expected output):**
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3080
```

---

## 使い方 / Usage

### 1. 学習 (Training)

#### 基本的な使い方 (Basic usage)
```bash
# GPUを自動検出 (Auto-detect GPU)
python -m Scripts.train

# GPUを明示的に指定 (Explicitly use GPU)
python -m Scripts.train --device cuda

# CPUを使用 (Use CPU)
python -m Scripts.train --device cpu
```

#### 推奨設定 (Recommended settings)
```bash
# GPU使用 + 多めのエピソード数 (GPU + More episodes)
python -m Scripts.train --device cuda --episodes 20000

# バッチサイズも増やす（GPUメモリに余裕がある場合）
# Increase batch size (if GPU memory allows)
python -m Scripts.train --device cuda --episodes 20000 --batch-size 64
```

#### バックグラウンド実行 (Background execution)
```bash
# 対話なしで実行 (Non-interactive)
nohup python -m Scripts.train --device cuda --episodes 50000 --no-interactive > training.log 2>&1 &
```

### 2. 評価 (Evaluation)

```bash
# GPUで評価 (Evaluate on GPU)
python -m Scripts.evaluate --device cuda

# より多くのゲームで評価 (More evaluation games)
python -m Scripts.evaluate --device cuda --games 500
```

---

## パフォーマンス比較 / Performance Comparison

| 環境 / Environment | 1000エピソード / 1000 Episodes | 10000エピソード / 10000 Episodes |
|-------------------|------------------------------|--------------------------------|
| CPU (Intel i7) | 約40分 / ~40 min | 約6.5時間 / ~6.5 hours |
| GPU (RTX 3060) | 約12分 / ~12 min | 約2時間 / ~2 hours |
| GPU (RTX 3080) | 約8分 / ~8 min | 約1.3時間 / ~1.3 hours |

※実際の速度は環境により異なります
*Actual speed varies by environment

---

## トラブルシューティング / Troubleshooting

### 問題: "CUDA not available" と表示される
### Issue: Shows "CUDA not available"

**原因と解決方法 / Causes and Solutions:**

1. **PyTorchがCPU版**
   - GPU版PyTorchを再インストール
   - Reinstall GPU version of PyTorch
   ```bash
   pip uninstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

2. **CUDAドライバが古い**
   - CUDAドライバを更新
   - Update CUDA drivers
   - [ダウンロード](https://developer.nvidia.com/cuda-downloads)

3. **GPU未対応**
   - `nvidia-smi` コマンドでGPUを確認
   - Check GPU with `nvidia-smi` command

### 問題: "CUDA out of memory" エラー
### Issue: "CUDA out of memory" error

**解決方法 / Solutions:**

1. **バッチサイズを減らす (Reduce batch size)**
   ```bash
   python -m Scripts.train --device cuda --batch-size 16
   ```

2. **他のプログラムを終了 (Close other programs)**
   - ブラウザやゲームなどGPUメモリを使用するプログラムを終了
   - Close programs using GPU memory (browsers, games, etc.)

3. **GPUメモリの監視 (Monitor GPU memory)**
   ```bash
   watch -n 1 nvidia-smi
   ```

### 問題: 学習が遅い
### Issue: Training is slow

**確認事項 / Check:**

1. **GPUが実際に使用されているか**
   ```bash
   # 別ターミナルで実行 (Run in another terminal)
   nvidia-smi
   ```
   - GPU使用率が高い（>80%）ことを確認
   - Verify GPU utilization is high (>80%)

2. **CPUボトルネックの可能性**
   - データ準備がCPUで行われるため、CPUも重要
   - Data preparation happens on CPU, so CPU is also important
   - 並列データローダーの使用を検討
   - Consider using parallel data loaders

---

## GPU最適化のヒント / GPU Optimization Tips

### 1. バッチサイズの調整
### Batch Size Tuning

```bash
# GPUメモリ 4GB: batch-size 16-32
python -m Scripts.train --device cuda --batch-size 32

# GPUメモリ 8GB: batch-size 32-64
python -m Scripts.train --device cuda --batch-size 64

# GPUメモリ 12GB+: batch-size 64-128
python -m Scripts.train --device cuda --batch-size 128
```

### 2. Mixed Precision Training（将来の実装）
### Mixed Precision Training (Future implementation)

現在は未実装ですが、将来的には以下で高速化可能:
Currently not implemented, but will be possible in the future:

```python
# torch.cuda.amp を使用してメモリ使用量を削減
# Use torch.cuda.amp to reduce memory usage
# 速度も向上（約1.5-2倍）
# Speed also improves (about 1.5-2x)
```

### 3. 複数GPU対応（将来の実装）
### Multi-GPU Support (Future implementation)

複数のGPUがある場合は、データ並列化で高速化可能:
If you have multiple GPUs, data parallelism can accelerate training:

```bash
# 将来的には以下のような使い方を予定
# Future planned usage
python -m Scripts.train --device cuda --multi-gpu
```

---

## よくある質問 / FAQ

**Q: GPUがなくても学習できますか？**
**Q: Can I train without a GPU?**

A: はい、CPUでも学習可能です。ただし時間がかかります。
A: Yes, you can train on CPU, but it takes longer.

**Q: Google ColabやKaggleのGPUは使えますか？**
**Q: Can I use GPUs from Google Colab or Kaggle?**

A: はい、使用可能です。ノートブック形式で実行してください。
A: Yes, you can. Run in notebook format.

```python
# Colabでの実行例 (Example in Colab)
!git clone https://github.com/norinori1/Othello-AI.git
%cd Othello-AI
!pip install -r requirements.txt
!python -m Scripts.train --device cuda --episodes 10000 --no-interactive
```

**Q: 学習中にGPU使用率が低い場合は？**
**Q: What if GPU utilization is low during training?**

A: バッチサイズを増やすか、データ準備の並列化を検討してください。
A: Increase batch size or consider parallelizing data preparation.

**Q: どのGPUを購入すべきですか？**
**Q: Which GPU should I buy?**

A: 機械学習用途には以下を推奨:
A: For machine learning, we recommend:
- エントリー: GTX 1660 Super / RTX 3050 (Entry: GTX 1660 Super / RTX 3050)
- 中級: RTX 3060 / RTX 4060 (Mid-level: RTX 3060 / RTX 4060)
- 上級: RTX 3080 / RTX 4080 (Advanced: RTX 3080 / RTX 4080)

---

## 参考リンク / References

- [PyTorch GPU Installation](https://pytorch.org/get-started/locally/)
- [CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)
- [NVIDIA GPU Cloud](https://www.nvidia.com/en-us/gpu-cloud/)
- [Google Colab (Free GPU)](https://colab.research.google.com/)

---

**最終更新 / Last Updated**: 2026-01-29
**バージョン / Version**: 1.0
