from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

import os

embeddings = OllamaEmbeddings(
    model='all-minilm:l6-v2'
)

vector_store = Chroma(
    collection_name='test_collection',
    embedding_function=embeddings
)

for resume in os.listdir('doc'):
    loader = PyPDFLoader(f'doc/{resume}') 
    pages = loader.load()
    
    text_splitter = CharacterTextSplitter(
        separator = '\n',
        chunk_size = 10000,
        chunk_overlap = 200,
        length_function = len
    )

    texts = text_splitter.split_documents(pages)

    ids = vector_store.add_documents(texts)

# loader = PyPDFLoader('doc/Resume-(08_01_2026)-Richie-Teoh.pdf')
# pages = loader.load()

# text_splitter = CharacterTextSplitter(
#     separator = '\n',
#     chunk_size = 10000,
#     chunk_overlap = 200,
#     length_function = len
# )

# texts = text_splitter.split_documents(pages)

# ids = vector_store.add_documents(texts)

# results = vector_store.similarity_search(
#     "When did Richie start his work as a Data Analyst Engineer at Doo Group?",
#     k=3,
#     threshold=0.7
# )

# for res in results:
#     print(res.page_content)
#     print("-"*100)
    # print(res.metadata)

retriever = vector_store.as_retriever()
llm = ChatOllama(
    model = 'gemma3:4b'
)

def format_doc(docs):
    return "\n\n".join([doc.page_content for doc in docs])

from langchain_core.prompts import PromptTemplate

prompt_template = '''
Use the context provided to answer the user's questions below. If you do not know the answer based on the context provided, tell the user that you do not know the answer to their question based on the context provided and that you are sorry.

Context: {context}

Question: {query}

Answer:
'''

custom_rag_prompt = PromptTemplate.from_template(prompt_template)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Create the RAG Chain
rag_chain = (
    {"context": retriever | format_doc, "query": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

# Query the RAG Chain
rag_chain.invoke(
  "What is Jack Internship company name?"
)





