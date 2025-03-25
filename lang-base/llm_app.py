from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

dp_key = os.getenv("GPT-API-KEY")
model = os.getenv("GPT-Model")
print(dp_key)

# 初始化模型
llm = ChatOpenAI()

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions."),
    ("user", "{input}")
])

# 创建链
chain = prompt | llm

# 调用链
print(chain.invoke({"input": "复旦大学在哪里?"}))




