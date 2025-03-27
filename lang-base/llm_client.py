from langserve import RemoteRunnable

if __name__ == "__main__":
    client = RemoteRunnable("http://localhost:8000/translate")
    
    # 使用 invoke 方法调用服务
    result = client.invoke({
        "input": "你好",
        "input_language": "中文",
        "output_language": "英语"
    })
    print(result)
    
    # 批量测试
    texts = [
        "我好想买一辆特斯拉model3",
        "今天天气真好",
        "人工智能正在改变世界"
    ]
    
    for text in texts:
        result = client.invoke({
            "input": text,
            "input_language": "中文",
            "output_language": "英语"
        })
        print(f"原文: {text}")
        print(f"译文: {result}\n")
