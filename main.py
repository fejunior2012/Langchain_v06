import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from dotenv import load_dotenv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Carrega a chave OpenAI
load_dotenv()

# Carrega o PDF
loader = PyPDFLoader("manual_crosser150sabs_2024.pdf")
pages = loader.load()
text = "\n".join([page.page_content for page in pages])

# Tokenização
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Divide o texto em pedaços menores
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=24,
    length_function=count_tokens,
)

chunks = text_splitter.create_documents([text])

# Gera embeddings e armazena no FAISS
print(os.environ["OPENAI_API_KEY"])
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"], model="text-embedding-ada-002")
db = FAISS.from_documents(chunks, embeddings)

# Criando o contexto para a pergunta
contexto = """
Você é um assistente especializado em mecânica de motos.
O documento a seguir contém informações detalhadas sobre um moto.
Use esse contexto para responder às perguntas de forma clara, precisa e direta.
"""

# Faz a pesquisa semântica
query = "Qual é óleo do motor?"
docs = db.similarity_search(query)

# Verifica se há documentos retornados antes de continuar
if docs:
    print(f"Documento mais relevante:\n{docs[0]}")

    # Carrega a chain e executa a resposta
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
        # Invocando a chain com contexto
    
    response = chain.invoke({
        "input_documents": docs,
        "question": f"{contexto}\nPergunta: {query}"
    })

    print("Resposta:", response["output_text"])  # A resposta fica na chave 'output_text'
else:
    print("Nenhum documento encontrado para responder à pergunta.")