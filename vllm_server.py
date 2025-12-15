import argparse
from vllm.entrypoints.openai import api_server
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

def start_vllm_server(
    model_name: str = "Qwen/Qwen-1.8B-Chat",  # 轻量中文模型，适合POC
    host: str = "0.0.0.0",
    port: int = 8000,
    gpu_memory_utilization: float = 0.8,
    max_num_batched_tokens: int = 1024,
    enable_streaming: bool = True
):
    """启动vLLM OpenAI兼容API服务"""
    # 1. 配置引擎参数
    engine_args = AsyncEngineArgs(
        model=model_name,
        host=host,
        port=port,
        gpu_memory_utilization=gpu_memory_utilization,  # 显存利用率（CPU可忽略）
        max_num_batched_tokens=max_num_batched_tokens,  # 批处理最大token数
        tensor_parallel_size=1,  # 单GPU（多GPU可调整）
        load_in_4bit=True,  # 4bit量化，降低显存占用（CPU/GPU都适用）
        trust_remote_code=True,  # 加载自定义模型（如Qwen/ChatGLM）需开启
    )

    # 2. 初始化异步引擎
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # 3. 启动OpenAI兼容API服务
    print(f"🚀 启动vLLM服务：模型={model_name}，地址=http://{host}:{port}")
    print(f"📌 API兼容OpenAI格式，支持 /v1/chat/completions /v1/models 等接口")
    api_server.run_server(
        engine=engine,
        engine_args=engine_args,
        host=host,
        port=port,
        allow_credentials=True,
        enable_streaming=enable_streaming  # 开启流式响应
    )

if __name__ == "__main__":
    # 命令行参数解析（方便快速切换模型/端口）
    parser = argparse.ArgumentParser(description="vLLM OpenAI API POC Server")
    parser.add_argument("--model", type=str, default="Qwen/Qwen-1.8B-Chat", 
                        help="模型名称（HF仓库名，如Llama-3-8B-Instruct、ChatGLM3-6B）")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="绑定地址")
    parser.add_argument("--port", type=int, default=8000, help="端口")
    args = parser.parse_args()

    # 启动服务
    start_vllm_server(
        model_name=args.model,
        host=args.host,
        port=args.port
    )