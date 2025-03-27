from langserve import RemoteRunnable

if __name__ == "__main__":
    client = RemoteRunnable("http://localhost:8000/chat")
    
    # 配置会话ID
    config = {
        "configurable": {
            "session_id": "123"
        }
    }
    
    # 第一轮对话
    result = client.invoke(
        {
            "input": "你好",
            "language": "中文"
        },
        config=config
    )
    print("Bot:", result)
    
    # 第二轮对话
    result = client.invoke(
        {
            "input": "你叫什么名字？",
            "language": "中文"
        },
        config=config
    )
    print("Bot:", result) 