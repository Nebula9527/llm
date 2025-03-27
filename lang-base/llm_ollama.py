from fastapi import FastAPI
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
import uvicorn

def clean_output(text: str) -> str:
    return text.split("\n")[-1].strip() 

llm = ChatOllama(model="deepseek-r1:1.5b",
                 base_url="http://localhost:11434",
                 temperature=0.5,
                 max_tokens=1024,
                 stream=True)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的翻译助手，能把{input_language}翻译成{output_language}，翻译用户输入的句子。"),
    ("user", "{input}")
])

chain = prompt | llm | StrOutputParser() | clean_output

app = FastAPI(title="陆烟儿丶", description="翻译服务",version="1.0.0")

# 只使用 langserve 路由
add_routes(
    app,
    chain,
    path="/translate",
    # 这里可以配置额外的路由选项
    config_keys=["configurable"]
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)




