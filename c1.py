# -*- coding: utf-8 -*-

# 导入必要的库
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain
import os
import gradio as gr
from datetime import datetime
# Even if you do not know the full answer, generate a one-paragraph hypothetical answer to the below question in Chinese
# 定义假设性回答模板
hyde_template = """Even if you do not know the full answer, generate a one-paragraph hypothetical answer to the below question:

{question}"""

# 定义最终回答模板
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

# 定义函数来处理问题
def process_question(url, api_key, model_name, question):
    # 初始化加载器并加载数据
    loader = WebBaseLoader(url)
    docs = loader.load()
    print('docs:',docs)
    # 设置环境变量
    os.environ['NVIDIA_API_KEY'] = api_key

    # 初始化嵌入层
    embeddings = NVIDIAEmbeddings(model="ai-embed-qa-4", truncate="END")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()

    # 初始化模型
    model = ChatNVIDIA(model=model_name)

    # 创建提示模板
    hyde_prompt = ChatPromptTemplate.from_template(hyde_template)
    hyde_query_transformer = hyde_prompt | model | StrOutputParser()

    # 定义检索函数
    @chain
    def hyde_retriever(question):
        hypothetical_document = hyde_query_transformer.invoke({"question": question})
        return retriever.invoke(hypothetical_document)

    # 定义最终回答链
    prompt = ChatPromptTemplate.from_template(template)
    answer_chain = prompt | model | StrOutputParser()

    @chain
    def final_chain(question):
        documents = hyde_retriever.invoke(question)
        response = ""
        for s in answer_chain.stream({"question": question, "context": documents}):
            response += s
        return response

    # 调用最终链获取答案
    return str(datetime.now())+final_chain.invoke(question)

# 定义可用的模型列表
models = ["ai-mixtral-8x7b-instruct","meta/llama-3.1-405b-instruct"]

#启动Gradio应用
iface = gr.Interface(
    fn=process_question,
    inputs=[
        gr.Textbox(label="输入需要学习的网址"),
        gr.Textbox(label="NVIDIA API Key"),
        gr.Dropdown(models, label="选择语言模型"),
        gr.Textbox(label="输入问题")
    ],
    outputs="text",
    title="网页知识问答系统"
)

# 启动Gradio界面
iface.launch()
# url = 'https://docs.api.nvidia.com/'
# api_key = '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'
# model_name = 'ai-mixtral-8x7b-instruct'
# question = 'what is nvidia'
# res = process_question(url,api_key,model_name,question)
# print(res)
