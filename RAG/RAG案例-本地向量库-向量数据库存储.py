from sentence_transformers import SentenceTransformer  # 导入句子转换器模型，用于将文本转换为向量
import json  # 导入json模块，用于处理JSON格式数据
import numpy as np  # 导入numpy库，用于数值计算和数组操作
import chromadb  # 导入chromadb，这是一个向量数据库，用于存储和检索向量数据
import os

# 加载预训练的句子转换模型，使用本地保存的模型文件
# '../maidalun1020'指向上一级目录中的模型文件夹
model = SentenceTransformer('../maidalun1020')

# 打开并读取训练数据文件，使用utf-8编码
with open('train_zh.json', 'r', encoding='utf-8') as f:
    # 逐行读取JSON数据并转换为Python对象，存入data列表
    data = [json.loads(line) for line in f]

# 从数据中提取前1000问题内容
instruction = [item['instruction'] for item in data[:1000]]
# 从数据中提取前1000回答内容
output = [item['output'] for item in data[:1000]]

# 使用模型将文本编码为向量表示
# convert_to_numpy=True表示将结果转换为numpy数组
instruction_embeddings = model.encode(instruction, convert_to_numpy=True)

# 存在向量文件，则删除
if os.path.exists('instruction_embeddings.npy'):
    os.remove('instruction_embeddings.npy')

# 将生成的向量表示保存到本地文件，便于后续使用
np.save('instruction_embeddings.npy', instruction_embeddings)

# 创建持久化的向量数据库客户端，指定存储路径
client = chromadb.PersistentClient(path="./collection.pkl")
# 创建一个名为"demo"的集合，用于存储文档和对应的向量，若果已经存在，先删除
if client.get_or_create_collection(name="demo"):
    client.delete_collection(name="demo")
collection = client.create_collection(name="demo")

# 遍历文本和对应的向量表示
for i, (instruction, embedding) in enumerate(zip(instruction, instruction_embeddings)):
    # 向集合中添加数据：
    collection.add(
        documents=[instruction],  # 文档内容（原始文本）
        embeddings=[embedding.tolist()],  # 向量表示（需要转换为列表格式）
        ids=[str(i)],  # 唯一标识符
        metadatas=[{"output": str(output[i])}]  # 元数据，包含关系信息
    ) 