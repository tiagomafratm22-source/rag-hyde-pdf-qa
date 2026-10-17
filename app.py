import os

import streamlit as st
from dotenv import load_dotenv
from pymilvus import MilvusClient
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_milvus import Milvus
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

URI = "./milvus_example.db"
COLLECTION = "documentos_rag"
K = 5

HYDE_TEMPLATE = """
Escreva um parágrafo curto, em português, que responderia à pergunta abaixo como se fosse um trecho de um documento oficial. Não adicione mais nada.
Pergunta: {query}
Trecho:
"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda usando exclusivamente o contexto fornecido. Se o contexto não tiver a informação, diga que não encontrou. Seja breve na resposta com no máximo 100 palavras.\n\nContexto:\n{contexto}"),
        ("human", "{query}"),
    ]
)

EXEMPLOS = [
    "Como fazer um seguro viagem?",
    "O que fazer se tiver meu cartão Gold roubado?",
    "Quais os beneficios de um cartão platinum?",
]


@st.cache_resource(show_spinner="Carregando banco vetorial e modelos...")
def carregar_pipeline():
    client = MilvusClient(uri=URI)
    pronta = (
        client.has_collection(COLLECTION)
        and client.get_collection_stats(COLLECTION)["row_count"] > 0
    )
    client.close()
    if not pronta:
        return None

    vector_store = Milvus(
        embedding_function=OllamaEmbeddings(model="bge-m3:567m"),
        collection_name=COLLECTION,
        connection_args={"uri": URI},
        index_params={"index_type": "FLAT", "metric_type": "L2"},
        auto_id=False,
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )
    hyde_chain = PromptTemplate.from_template(HYDE_TEMPLATE) | llm | StrOutputParser()
    return vector_store, llm, hyde_chain


st.set_page_config(page_title="RAG com HyDE", page_icon="🔎", layout="centered")
st.title("🔎 RAG com HyDE")
st.caption("Perguntas e respostas sobre os guias de cartões Gold, Platinum e Standard")

pipeline = carregar_pipeline()
if pipeline is None:
    st.error("Banco vetorial vazio. Execute `python main.py` uma vez para indexar os PDFs.")
    st.stop()
vector_store, llm, hyde_chain = pipeline

if "pergunta" not in st.session_state:
    st.session_state.pergunta = ""

st.write("Exemplos:")
for col, exemplo in zip(st.columns(len(EXEMPLOS)), EXEMPLOS):
    if col.button(exemplo, use_container_width=True):
        st.session_state.pergunta = exemplo

pergunta = st.text_input("Sua pergunta", key="pergunta")

if st.button("Perguntar", type="primary") and pergunta.strip():
    with st.status("Executando o pipeline...", expanded=True) as status:
        st.write("**1. HyDE:** gerando trecho hipotético")
        hipotetico = hyde_chain.invoke({"query": pergunta})
        st.info(hipotetico)

        st.write(f"**2. Retrieve:** buscando os {K} chunks mais similares no Milvus")
        docs = vector_store.similarity_search(hipotetico, k=K)

        st.write("**3. Generate:** gerando a resposta com o contexto")
        contexto = "\n\n".join(d.page_content for d in docs)
        resposta = StrOutputParser().invoke(
            llm.invoke(RAG_PROMPT.invoke({"query": pergunta, "contexto": contexto}))
        )
        status.update(label="Pronto!", state="complete", expanded=False)

    st.subheader("Resposta")
    st.success(resposta)

    with st.expander(f"Fontes recuperadas ({len(docs)} chunks)"):
        for i, d in enumerate(docs, 1):
            arquivo = os.path.basename(str(d.metadata.get("source", "?")))
            st.markdown(f"**{i}. {arquivo}**, página {d.metadata.get('page', '?')}")
            st.text(d.page_content)
