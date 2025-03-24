from sentence_transformers import SentenceTransformer  # 导入句子转换器模型，用于将文本转换为向量
import json  # 导入json模块，用于处理JSON格式数据
import numpy as np  # 导入numpy库，用于数值计算和数组操作
import chromadb  # 导入chromadb，这是一个向量数据库，用于存储和检索向量数据
import sys

# 添加父目录到系统路径，以便能找到dp.py模块
sys.path.append('..')
import dp  # 导入dp模块，用于AI接口调用


# 获取知识库对应的向量信息
# instruction_embeddings = np.load('instruction_embeddings.npy')

# 使用模型将文本编码为向量表示
# convert_to_numpy=True表示将结果转换为numpy数组
# texts_embeddings = model.encode(texts, convert_to_numpy=True)

client = chromadb.PersistentClient(path="./collection.pkl")
collection = client.get_collection(name="demo")
print(collection.count())

# 加载预训练的句子转换模型，使用本地保存的模型文件
model = SentenceTransformer('../maidalun1020')

def retrieve_response(query,top_n=5):
    results = collection.query(
        query_embeddings=model.encode([query], convert_to_numpy=True).tolist(),
        n_results=top_n
    )
    # 双重循环
    return [metadata['output'] for metadata_list in results['metadatas'] for metadata in metadata_list]


prompt_template = """你是一个问答机器人。
你的任务是根据下述给定的已知信息回答用户问题。
确保你的回复完全依据下述已知信息。不要编造答案。
如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。
已知信息：
__INFO__

用户问：
__QUERY__
"""


def build_prompt(prompt_template,**kwargs):
    prompt = prompt_template
    for k,v in kwargs.items():
        v = v if isinstance(v,str) else '\n'.join(v) if isinstance(v,list) else str(v)
        prompt = prompt.replace(f"__{k.upper()}__",v)
    return prompt


def get_completion(prompt):
    response = dp.generate_sql_with_ai(prompt)
    return response


class RAG_BOT:
    def __init__(self,llm_api,n_results=5):
        self.llm_api = llm_api
        self.n_results = n_results

    def chat(self,user_query):
        # 冲向量哭获取检索信息
        info = retrieve_response(user_query,self.n_results)
        # 构建prompt
        prompt = build_prompt(prompt_template,info=info,query=user_query)
        print(prompt)
        # 结合向量库检索信息，调用信息
        response = self.llm_api(prompt)
        return response
    

if __name__ == "__main__":
    bot = RAG_BOT(get_completion)
    user_query = "做包皮手术的过程痛苦吗？"
    response = bot.chat(user_query)
    print(response)
    



