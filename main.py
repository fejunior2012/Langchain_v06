import os
import pandas as pd
import streamlit as st
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

# Definindo se vai criar o banco de dados (True) ou usar um existente (False)
criar_bd = False  # Mude para False quando quiser usar o banco existente

# Caminho do banco de dados
db_path = "faiss_db"  # Caminho para salvar o banco de dados FAISS

# Carrega o PDF
# loader = PyPDFLoader("IC_1001591201.pdf")
loader = PyPDFLoader("manual_crosser150sabs_2024.pdf")
pages = loader.load()
text = "\n".join([page.page_content for page in pages])

# Tokenização
tokenizer = GPT2TokenizerFast.from_pretrained("openai-gpt")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Divide o texto em pedaços menores
text_splitter = RecursiveCharacterTextSplitter(
    # chunk_size=512,
    chunk_size=256,
    chunk_overlap=12,
    length_function=count_tokens,
)

chunks = text_splitter.create_documents([text])

# Gera embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"], model="text-embedding-ada-002")

# Se a variável `criar_bd` for True, cria o banco de dados FAISS
if criar_bd:
    db = FAISS.from_documents(chunks, embeddings)
    
    db.save_local(db_path)  # Salva o banco de dados localmente
    print(f"Banco de dados criado e salvo em {db_path}")


    # print(f"Banco de dados carregado de {db_path}")

# Criando o contexto para a pergunta
contexto = """
Você é um especialista em contratro e deverá responder informações da documentação disponibilizada. O contrato é público.
"""

# Faz a pesquisa semântica
def generate_response(query):

    # Caso contrário, carrega o banco de dados FAISS existente
    # Carregando o banco de dados FAISS, permitindo deserialização "perigosa" se for seguro
    try:
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        print(f"Banco de dados carregado de {db_path}")
    except Exception as e:
        print(f"Erro ao carregar o banco de dados: {str(e)}")

    docs = db.similarity_search(query, k=3) 

    # Verifica se há documentos retornados antes de continuar
    if docs:
        print(f"Documento mais relevante:\n{docs[0]}")
   
        chain = load_qa_chain(OpenAI(temperature=0), chain_type="refine")
            # Invocando a chain com contexto
        
        response = chain.invoke({
            "input_documents": docs,
            "question": f"{contexto}\nPergunta: {query}"
        })

        return response["output_text"]  # A resposta fica na chave 'output_text'
    else:
        return "Nenhum documento encontrado para responder à pergunta."


def main():
    st.set_page_config(
        page_icon="Pergunte ao Contrato"
    )
    st.header("Documento PDF")
    query = st.text_area("Pergunte ao documento sobre o contrato?")

    if query:
        st.write("Buscando informações dentro do contrato...")        
        result = generate_response(query)
        st.info(result)

if __name__ == '__main__':
    main()

    # python.exe -m pip install --upgrade pip
# pip install pandas streamlit langchain langchain-community langchain-openai faiss-cpu transformers python-dotenv pypdf
# pip install --upgrade --force-reinstall cryptography
# pip install --upgrade --force-reinstall pypdf
# pip install torch ou com GPU pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121







# Carrega a chain e executa a resposta
# "stuff" → Quando os documentos são curtos e cabem no contexto do modelo. ⚡ (Rápido)
# "map_reduce" → Quando há muitos documentos ou textos longos. 📖
# "refine" → Quando precisa de respostas bem elaboradas e coerentes. 🔍
# "map_rerank" → Quando precisa encontrar a resposta mais relevante entre várias. 🎯     