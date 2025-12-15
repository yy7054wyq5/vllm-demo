# 环境安装

本文档简要说明如何为 vLLM 创建并配置 Python 环境（推荐使用 Conda）。适配的 Python 版本：3.8–3.11。

## 1. 创建并激活 Conda 环境

创建环境（示例使用 Python 3.10）：

```bash
conda create -n vllm-env python=3.10 -y
```

激活环境（激活后终端前缀会显示 (vllm-env)）：

```bash
conda activate vllm-env
```

退出环境：

```bash
conda deactivate
```

## 2. 安装依赖

推荐根据是否有 NVIDIA GPU 选择下面的方式：

- GPU（推荐）

	如果有 NVIDIA 显卡（在 WSL2 下也支持），先通过 Conda 安装与 CUDA 兼容的 PyTorch：

	```bash
	conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
	```

	然后安装 vLLM 及常用依赖：

	```bash
	pip install vllm fastapi requests pydantic
	```

- 纯 CPU（无 GPU）

	```bash
	pip install vllm fastapi requests pydantic
	```

## 3. 运行与更多信息

这份文档只包含安装与环境准备的最小步骤。有关如何运行模型、启动服务或示例用法，请参考 vLLM 官方文档或在仓库中添加运行示例。

如果需要，我可以帮你：

- 添加一个快速运行示例（启动 API 或本地推理脚本）；
- 补充依赖版本锁（`requirements.txt` 或 `environment.yml`）；
- 提供 GPU/CPU 的性能比较说明。


