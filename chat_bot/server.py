from fastapi import FastAPI
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
import uvicorn
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder

def clean_output(text: str) -> str:
    return text.split("\n")[-1].strip() 

llm = ChatOllama(model="deepseek-r1:14b",
                 base_url="http://localhost:11434",
                 temperature=0.5,
                 max_tokens=1024,
                 stream=True)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个知无不言得陪伴者，能使用{language}尽所有可能回答所有问题。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser() | clean_output

# 消息记录
message_store = {}

def get_message_history(session_id: str):
    if session_id not in message_store:
        message_store[session_id] = ChatMessageHistory()
    return message_store[session_id]

do_message = RunnableWithMessageHistory(
    chain,
    get_message_history,
    input_messages_key="input",
    history_messages_key="history",
)

config = {
    'configurable': {
        'session_id': '123'
    }
}

# 第一轮对话
resp = do_message.invoke(
    {
        "input": "你好啊，我是陆烟儿",
        "language": "中文"
    },
    config=config
)
print(resp)

# 第二轮对话
resp1 = do_message.invoke(
    {
        "input": "请问我的名字是什么？",
        "language": "中文"
    },
    config=config
)
print(resp1)

# app = FastAPI(title="陆烟儿丶", description="知无不言得陪伴者",version="1.0.0")

# # 添加路由
# add_routes(
#     app,
#     do_message,
#     path="/chat",
#     config_keys=["configurable"]
# )

# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8000)


