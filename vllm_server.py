import argparse
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from vllm import LLM, SamplingParams

# 初始化FastAPI应用
app = FastAPI(title="vLLM AI API Server", version="1.0")
llm = None  # 全局LLM引擎实例
sampling_params = None  # 全局采样参数

# 定义请求体格式（兼容OpenAI格式）
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False

# 健康检查接口
@app.get("/health")
async def health_check():
    if llm is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {
        "status": "running",
        "model": llm.model_config.model,
        "gpu_memory_utilization": llm.engine_args.gpu_memory_utilization
    }

# 核心聊天接口（非流式）
@app.post("/v1/chat/completions")
async def chat_completion(req: ChatRequest):
    try:
        # 提取用户最后一条消息（兼容多轮对话）
        user_prompt = req.messages[-1]["content"]
        # 设置采样参数
        params = SamplingParams(
            temperature=req.temperature,
            max_tokens=req.max_tokens
        )
        # 推理生成
        outputs = llm.generate([user_prompt], params)
        response_text = outputs[0].outputs[0].text
        
        # 兼容OpenAI响应格式
        return {
            "id": f"cmpl-{outputs[0].request_id}",
            "object": "chat.completion",
            "created": int(outputs[0].created_time.timestamp()),
            "model": llm.model_config.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(outputs[0].prompt_token_ids),
                "completion_tokens": len(outputs[0].outputs[0].token_ids),
                "total_tokens": len(outputs[0].prompt_token_ids) + len(outputs[0].outputs[0].token_ids)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="vLLM FastAPI Server (All Versions)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name/path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="GPU memory utilization")
    args = parser.parse_args()

    # 全局初始化LLM引擎（所有vLLM版本通用）
    global llm
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        trust_remote_code=True
    )

    # 启动FastAPI服务
    print(f"🚀 vLLM FastAPI Server started: http://{args.host}:{args.port}")
    print(f"📌 Model: {args.model}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()