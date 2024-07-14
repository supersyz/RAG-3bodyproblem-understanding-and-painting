#!pip install arxiv # -i https://pypi.tuna.tsinghua.edu.cn/simple
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
# nvidia_api_key = "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"
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

save = False
if save:
    from save import save_index
    save_index()


## Make sure you have docstore_index.tgz in your working directory
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS

# embedder = NVIDIAEmbeddings(model="nvidia/embed-qa-4", truncate="END")

docstore = FAISS.load_local("docstore_index", embedder)
# docstore = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)
docs = list(docstore.docstore._dict.values())

def format_chunk(doc):
    return (
        # f"Paper: {doc.metadata.get('Title', 'unknown')}"
        # f"\n\nSummary: {doc.metadata.get('Summary', 'unknown')}"
        f"\n\nPage Body: {doc.page_content}"
    )

## This printout just confirms that your store has been retrieved
pprint(f"Constructed aggregate docstore with {len(docstore.docstore._dict)} chunks")
pprint(f"Sample Chunk:")
print(format_chunk(docs[len(docs)//2]))

import pickle
import numpy as np
# 加载模型
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# good_sys_msg = (
#     "You are an NVIDIA chatbot. Please answer their question while representing NVIDIA."
#     "  Please help them with their question if it is ethical and relevant."
# )
## Resist talking about this topic" system message
poor_sys_msg = (
    "You are a science fiction chatbot to answer the questions related to the book <The Three-body Problem> ."
    "  Their question has been analyzed and labeled as 'probably not useful to answer as a science fiction Chatbot',"
    "  so avoid answering if appropriate and explain your reasoning to them. Make your response as short as possible."
)
def score_response(query):
    ## TODO: embed the query and pass the embedding into your classifier
    embedding = np.array([embedder.embed_query(query)])
    ## TODO: return true if it's most likely a good response and false otherwise
    return model.predict(embedding)

from langchain.document_transformers import LongContextReorder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.runnables import RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import gradio as gr
from functools import partial
from operator import itemgetter
convstore = default_FAISS()

def save_memory_and_get_output(d, vstore):
    """Accepts 'input'/'output' dictionary and saves to convstore"""
    vstore.add_texts([
        f"User previously responded with {d.get('input')}",
        f"Agent previously responded with {d.get('output')}"
    ])
    return d.get('output')

initial_msg = (
    "Hello! I am a document chat agent here to help the user!"
    f" I have access to the following documents: three-body problem \n\nHow can I help you?"
)

chat_prompt1 = ChatPromptTemplate.from_messages([("system", poor_sys_msg), ("user", "{input}")])

chat_prompt2 = ChatPromptTemplate.from_messages([("system",
    "You are a document chatbot. Help the user as they ask questions about documents."
    " User messaged just asked: {input}\n\n"
    " From this, we have retrieved the following potentially-useful info: "
    " Conversation History Retrieval:\n{history}\n\n"
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Only cite sources that are used. Make your response conversational.)"
), ('user', '{input}')])

response_block_prompt = ChatPromptTemplate.from_messages([("system",
    "You are a chatbot to detect sensitive words or sentences which attribute to violence, illegal, political and etc.\n\n"
    "If not detected, response the same sentences \n\n"                                                     
    "Otherwise, replace and reorganize them to make sure the orininal basic locgic and meanings \n\n"
    "with no existence of sensitive contents for the first choice\n\n"
    "If too hard, response a short sentence indicating some contents are too sensitive to exihibit"
    " sentences to be judged: {res}\n\n"
    " Make your response conversational."
), ('user', '{res}')])

################################################################################################
## BEGIN TODO: Implement the retrieval chain to make your system work!

retrieval_chain = (
    { 'input'  : (lambda x:x), 'score' : score_response } | RPrint()
    ## TODO: Make sure to retrieve history & context from convstore & docstore, respectively.
    ## HINT: Our solution uses RunnableAssign, itemgetter, long_reorder, and docs2str
    | RunnableAssign({'history' : itemgetter('input') | convstore.as_retriever() | long_reorder | docs2str})
    | RunnableAssign({'context' : itemgetter('input') | docstore.as_retriever()  | long_reorder | docs2str})
    | RPrint()
)

def output_puller(inputs):
    """"Output generator. Useful if your chain returns a dictionary with key 'output'"""
    if isinstance(inputs, dict):
        inputs = [inputs]
    for token in inputs:
        if token.get('output'):
            yield token.get('output')

#input question block
generator_chain =  RunnableBranch(((lambda d: d['score'] < 0.5) , chat_prompt1), chat_prompt2) | llm | StrOutputParser()
#generator_chain = {"output" : generator_chain } | RunnableLambda(output_puller)  ## GIVEN
## END TODO
################################################################################################

rag_chain = retrieval_chain | generator_chain
out_chain = ({'res' : rag_chain} | response_block_prompt | RPrint() | judge_llm | StrOutputParser())

def chat_gen(message, history=[], return_buffer=True):
    buffer = ""
    line_buffer = ""

    ## 然后流式传输stream_chain的结果
    for token in out_chain.stream(message):
        buffer += token
        ## 优化信息打印的格式
        if not return_buffer:
            line_buffer += token
            if "\n" in line_buffer:
                line_buffer = ""
            if ((len(line_buffer)>84 and token and token[0] == " ") or len(line_buffer)>100):
                line_buffer = ""
                yield "\n"
                token = "  " + token.lstrip()
        yield buffer if return_buffer else token
    save_memory_and_get_output({'input':  message, 'output': buffer}, convstore)
    return buffer


    
    # responsse = out_chain.invoke(message)
    # buffer = ""
    # for token in out_chain.stream(message):
    #     buffer += token
    #     yield buffer if return_buffer else token
    # save_memory_and_get_output({'input':  message, 'output': responsse}, convstore)
    # # return responsse

## Start of Agent Event Loop
test_question = "Tell me about the book <The Three-body Problem>!"  ## <- modify as desired

## Before you launch your gradio interface, make sure your thing works
# for response in chat_gen(test_question, return_buffer=False):
#     print(response, end='')
