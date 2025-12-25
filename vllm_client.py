import requests
import json
import time
from typing import Dict, List

# 服务端地址（WSL部署后，Windows用localhost:8000，局域网用Windows IP）
BASE_URL = "http://localhost:8000"
# API Key（本地部署无需真实key，填任意值即可）
API_KEY = "dummy-key"

def get_headers() -> Dict[str, str]:
    """获取通用请求头"""
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

def health_check() -> Dict:
    """健康检查：确认服务端和模型是否正常"""
    url = f"{BASE_URL}/health"
    try:
        response = requests.get(url, headers=get_headers(), timeout=10)
        response.raise_for_status()  # 抛出HTTP错误
        return {
            "status": "success",
            "data": response.json()
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }

def list_models() -> Dict:
    """模拟列出模型（适配服务端逻辑）"""
    # 服务端未实现/v1/models接口时，返回模拟数据（和真实逻辑对齐）
    url = f"{BASE_URL}/v1/models"
    try:
        response = requests.get(url, headers=get_headers(), timeout=10)
        return response.json()
    except:
        # 兼容服务端未实现该接口的情况，返回模拟数据
        return {
            "object": "list",
            "data": [
                {
                    "id": "Qwen/Qwen3-0.6B",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "vllm",
                    "root": "Qwen/Qwen3-0.6B",
                    "parent": None
                }
            ]
        }

def chat_completion_non_stream(
    messages: List[Dict],
    temperature: float = 0.7,
    max_tokens: int = 512
) -> Dict:
    """非流式调用（一次性返回结果）"""
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": "Qwen/Qwen3-0.6B",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    try:
        response = requests.post(
            url,
            headers=get_headers(),
            json=payload,
            timeout=60  # 推理超时时间（按需调整）
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "response": response.text if 'response' in locals() else ""
        }

def chat_completion_stream(
    messages: List[Dict],
    temperature: float = 0.7,
    max_tokens: int = 512
):
    """流式调用（逐字返回结果，模拟实时聊天）"""
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": "Qwen/Qwen3-0.6B",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True
    }
    try:
        print("📝 流式响应（实时输出）：")
        response = requests.post(
            url,
            headers=get_headers(),
            json=payload,
            stream=True,  # 开启流式响应
            timeout=60
        )
        response.raise_for_status()
        
        # 解析流式数据（兼容OpenAI SSE格式）
        full_content = ""
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    line = line[6:]  # 去掉前缀"data: "
                    if line == "[DONE]":
                        break
                    try:
                        data = json.loads(line)
                        content = data["choices"][0]["message"].get("content", "")
                        if content:
                            full_content += content
                            print(content, end="", flush=True)  # 实时打印
                    except:
                        continue
        print("\n")
        return {
            "status": "success",
            "full_content": full_content
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }

if __name__ == "__main__":
    # ========== 1. 健康检查 ==========
    print("===== 1. 服务端健康检查 =====")
    health_result = health_check()
    print(json.dumps(health_result, indent=2, ensure_ascii=False))
    if health_result["status"] == "failed":
        print("❌ 服务端未正常运行，请先启动服务端！")
        exit(1)

    # ========== 2. 列出模型 ==========
    print("\n===== 2. 已加载模型列表 =====")
    models = list_models()
    print(json.dumps(models, indent=2, ensure_ascii=False))

    # ========== 3. 测试消息（可自定义） ==========
    test_messages = [
        {"role": "system", "content": "你是一个简洁的AI助手，回答准确、易懂，用中文回复"},
        {"role": "user", "content": "介绍一下vLLM的核心优势，用3句话概括"}
    ]

    # ========== 4. 非流式调用 ==========
    print("\n===== 3. 非流式调用结果 =====")
    non_stream_result = chat_completion_non_stream(test_messages)
    print(json.dumps(non_stream_result, indent=2, ensure_ascii=False))

    # ========== 5. 流式调用 ==========
    print("\n===== 4. 流式调用结果 =====")
    stream_result = chat_completion_stream(test_messages)
    if stream_result["status"] == "success":
        print(f"\n✅ 流式调用完成，完整内容：{stream_result['full_content']}")
    else:
        print(f"❌ 流式调用失败：{stream_result['error']}")