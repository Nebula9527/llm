from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="deepseek-r1:1.5b",
                 base_url="http://localhost:11434",
                 temperature=0.5,
                 max_tokens=1024,
                 stream=True)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的翻译助手，能把{input_language}翻译成{output_language}，翻译用户输入的句子。"),
    ("user", "{input}")
])

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"input_language": "中文", "output_language": "英语", "input": "帮我把'今天天气晴朗，打算和女朋友去川西玩'翻译一下"}))


