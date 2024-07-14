from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.vectorstores import FAISS
#from llama_index.embeddings import LangchainEmbedding
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_transformers import LongContextReorder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore

import gradio as gr
from functools import partial
from operator import itemgetter
import os
from functools import partial
from rich.console import Console
from rich.style import Style
from rich.theme import Theme

import os
nvidia_api_key = "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"
# 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
assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
os.environ["NVIDIA_API_KEY"] = nvidia_api_key

llm = ChatNVIDIA(model="ai-mixtral-8x7b-instruct")
embedder = NVIDIAEmbeddings(model="ai-embed-qa-4", truncate="END")
judge_llm =  ChatNVIDIA(model="ai-mixtral-8x7b-instruct")

from langchain.document_transformers import LongContextReorder
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.schema.runnable.passthrough import RunnableAssign
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from functools import partial
from operator import itemgetter

########################################################################
## Utility Runnables/Methods
def RPrint(preface=""):
    """Simple passthrough "prints, then returns" chain"""
    def print_and_return(x, preface):
        if preface: print(preface, end="")
        pprint(x)
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

def docs2str(docs, title="Document"):
    """Useful utility for making chunks into context string. Optional, but useful"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name:
            out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

console = Console()
base_style = Style(color="#76B900", bold=True)
pprint = partial(console.print, style=base_style)
## Optional; Reorders longer documents to center of output text
long_reorder = RunnableLambda(LongContextReorder().transform_documents)

from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore

embed_dims = len(embedder.embed_query("test"))
def default_FAISS():
    '''Useful utility for making an empty FAISS vectorstore'''
    return FAISS(
        embedding_function=embedder,
        index=IndexFlatL2(embed_dims),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )

def aggregate_vstores(vectorstores):
    ## Initialize an empty FAISS Index and merge others into it
    ## We'll use default_faiss for simplicity, though it's tied to your embedder by reference
    agg_vstore = default_FAISS()
    for vstore in vectorstores:
        agg_vstore.merge_from(vstore)
    return agg_vstore





import os
import json
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 使用unstructured是因为机器上上有这个库的支持
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredEPubLoader, UnstructuredHTMLLoader, UnstructuredFileLoader


# 指定目录路径
directory_path = './books'
ps = os.listdir(directory_path)
# 仅过滤出文件
file_list = [f for f in ps if os.path.isfile(os.path.join(directory_path, f))]

docs = []
docs_name = [] # 文档名list

print("Loading Documents")
for p in file_list:
    print(p)
    path2file=os.path.join(directory_path, p)
    if p.endswith('.txt'): # 更多的支持格式可以以元组形式传入
        docs_name.append(path2file)
        doc = UnstructuredFileLoader(path2file).load() # The file loader uses the unstructured partition function and will automatically detect the file type.
        docs.append(doc)
    elif p.endswith('.pdf'):
        docs_name.append(path2file)
        doc = UnstructuredPDFLoader(path2file).load()
        docs.append(doc)
    # elif p.endswith('.epub'):
    #     docs_name.append(path2file)
    #     doc = UnstructuredEPubLoader(path2file).load()
    #     docs.append(doc)
    elif p.endswith('.html'):
        docs_name.append(path2file)
        doc = UnstructuredFileLoader(path2file).load()
        docs.append(doc)
    # else:
    #     try:
    #         raise TypeError
    #     except TypeError as e:
    #         print(f"unsupported format！")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", ";", ",", " ", ""],
)

## Split the documents and also filter out stubs (overly short chunks)
print("Chunking Documents")
docs_chunks = [text_splitter.split_documents(doc) for doc in docs]
docs_chunks = [[c for c in dchunks if len(c.page_content) > 200] for dchunks in docs_chunks]

## Make some custom Chunks to give big-picture details
doc_string = "Available Documents:"
doc_metadata = []
for chunks in docs_chunks:
    # print(chunks[0])
    metadata = getattr(chunks[0], 'metadata', {}) # 运行查看是否有相关key
    # doc_string += "\n - " + metadata.get('Title')
    doc_metadata += [str(metadata)]

# extra_chunks = [doc_string] + doc_metadata
extra_chunks = doc_metadata

## Printing out some summary information for reference
pprint(doc_string, '\n')
for i, chunks in enumerate(docs_chunks):
    print(f"Document {i}")
    print(f" - # Chunks: {len(chunks)}")
    print(f" - Metadata: ")
    pprint(chunks[0].metadata)
    print()

print(len(extra_chunks))

def save_index():
    print("Constructing Vector Stores")
    vecstores = [FAISS.from_texts(extra_chunks, embedder)]
    from tqdm import tqdm
    vecstores += [FAISS.from_documents(doc_chunks, embedder) for doc_chunks in tqdm(docs_chunks)]

    ## Unintuitive optimization; merge_from seems to optimize constituent vector stores away
    docstore = aggregate_vstores(vecstores)

    print(f"Constructed aggregate docstore with {len(docstore.docstore._dict)} chunks")
    docstore.save_local("docstore_index")