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
