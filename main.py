import os
import json
import hashlib
import random
from typing import List, Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pymilvus import MilvusClient
from transformers import AutoTokenizer

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

load_dotenv()

google_api_key=os.getenv("GOOGLE_API_KEY") 
langsmith_api_key=os.getenv("LANGSMITH_API_KEY")


pdfs = DirectoryLoader("documentos", glob="*.pdf", loader_cls=PyPDFLoader).load()

print(f"{len(pdfs)} páginas de PDF")

embeddings_model = "BAAI/bge-m3"
embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model)

splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=embeddings_tokenizer,
    chunk_size=1250, chunk_overlap=150
)

chunks = splitter.split_documents(pdfs)

# Remove chunks idênticos (mesmo arquivo, página e texto) dentro da mesma execução
def id_conteudo(texto):
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()

# Gera um ID único para cada chunk, baseado no arquivo, página e conteúdo
def gerar_id(doc):
    return id_conteudo(f"{doc.metadata.get('source')}|{doc.metadata.get('page')}|{doc.page_content}") 


# Remove chunks idênticos (mesmo arquivo, página e texto) dentro da mesma execução
unicos = {}
for c in chunks:
    unicos.setdefault(gerar_id(c), c)
ids = list(unicos.keys())
chunks = list(unicos.values())
print(f"{len(chunks)} chunks únicos")

embeddings=OllamaEmbeddings(
    model="bge-m3:567m"
)

#embeddings.embed_query(chunks[0].page_content)

# Criação do banco vetorial
URI = "./milvus_example.db"
COLLECTION = "documentos_rag"  # nome fixo: reabre a mesma coleção a cada execução
REINDEX = False  # True = apaga a coleção e indexa tudo de novo
CONNECTION_ARGS = {
    "uri": URI,
    "grpc_options": {
        # O padrão de 10 s do pymilvus pode exceder o limite de pings do servidor.
        "grpc.keepalive_time_ms": 60000,
        "grpc.keepalive_timeout_ms": 20000,
        "grpc.keepalive_permit_without_calls": False,
    },
}

# Verifica se a coleção já existe no arquivo .db e se tem dados
client = MilvusClient(**CONNECTION_ARGS)
colecao_existe = (
    client.has_collection(COLLECTION)
    and client.get_collection_stats(COLLECTION)["row_count"] > 0
)
client.close()

vector_store = Milvus(
    embedding_function=embeddings,
    collection_name=COLLECTION,
    connection_args=CONNECTION_ARGS,  # Utiliza um banco vetorial local no disco
    index_params={"index_type": "FLAT", "metric_type": "L2"},
    auto_id=False,                 # os IDs vêm do hash do conteúdo
    drop_old=REINDEX,
)

if REINDEX or not colecao_existe:
    vector_store.add_documents(chunks, ids=ids)
    print(f"{len(chunks)} chunks indexados")
else:
    print("Coleção já existe: ingestão ignorada (use REINDEX = True para reindexar)")


# Gerando prompt Hyde: cria um trecho hipotético que será usado para recuperar documentos similares
hyde_prompt_template = """
Escreva um parágrafo curto, em português, que responderia à pergunta abaixo como se fosse um trecho de um documento oficial. Não adicione mais nada.
Pergunta: {query}
Trecho:
"""

llm=ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    api_key=google_api_key,
    temperature=0
)

hyde_prompt = PromptTemplate.from_template(hyde_prompt_template)
hyde_chain = hyde_prompt | llm | StrOutputParser()

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda usando exclusivamente o contexto fornecido. Se o contexto não tiver a informação, diga que não encontrou. Seja breve na resposta com no máximo 100 palavras.\n\nContexto:\n{contexto}"),
        ("human", "{query}")
    ]
)

# LANGGRAPH: cria um grafo de estados para o fluxo de RAG

K = 5 # chunks recuperados por pergunta

# Define state for application
class State(TypedDict):
    question: str
    hypothetical: str
    context: List[Document]
    answer: str


# Define application steps
def hyde(state: State):
    return {"hypothetical": hyde_chain.invoke({"query": state["question"]})} # Define um trecho hipotético que será usado para recuperar documentos similares


# Recupera documentos similares com base no trecho hipotético
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["hypothetical"], k=K)
    return {"context": retrieved_docs}

# Gera a resposta final com base na pergunta e nos documentos recuperados
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = rag_prompt.invoke({"query": state["question"], "contexto": docs_content})
    response = llm.invoke(messages)
    return {"answer": StrOutputParser().invoke(response)}


# Constrói o grafo da aplicação
graph_builder = StateGraph(State)
graph_builder.add_node("hyde", hyde)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_edge(START, "hyde")
graph_builder.add_edge("hyde", "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)
graph = graph_builder.compile()

print(graph.invoke({"question": "Como fazer um seguro viagem?"})["answer"])
print(graph.invoke({"question": "O que fazer se tiver meu cartão Gold roubado?"})["answer"])
print(graph.invoke({"question": "Quais os beneficios de um cartão platinum?"})["answer"])


# =============================================================================
# Avaliação: geração do dataset de perguntas e respostas
# =============================================================================

# Define a estrutura de saída para o dataset de avaliação criando uma classe Pydantic
class QandA(BaseModel):
    pergunta: str = Field(description="Pergunta autossuficiente, em português")
    resposta: str = Field(description="Resposta correta em até duas frases")


qa_prompt = ChatPromptTemplate.from_template(
    "Com base APENAS no trecho abaixo, crie UMA pergunta que um cliente faria e a resposta correta.\n"
    'A pergunta deve ser autossuficiente (não use "segundo o texto" ou "neste trecho") e específica '
    "o bastante para ser respondida por este trecho. A resposta deve ter no máximo duas frases.\n\n"
    "Trecho:\n{trecho}"
)
qa_chain = qa_prompt | llm.with_structured_output(QandA)

N_AVALIACAO = 30                # quantidade de perguntas do dataset de avaliação
ARQ_QA = "qa_dataset.json"      # apague este arquivo se mudar pdfs ou chunking
ARQ_RESULTADOS = "resultados_avaliacao.json"
MAX_CONCORRENCIA = 4            # reduza se bater no limite de requisições do llm

if os.path.exists(ARQ_QA):
    with open(ARQ_QA, "r", encoding="utf-8") as file:
        eval_data = json.load(file)
    print(f"Dataset carregado de {ARQ_QA}: {len(eval_data)} perguntas")
else:
    random.seed(42)
    candidatos = [c for c in chunks if len(c.page_content) >= 300]
    amostra = random.sample(candidatos, min(N_AVALIACAO, len(candidatos)))

    perguntas_respostas = qa_chain.batch(
        [{"trecho": c.page_content} for c in amostra],
        config={"max_concurrency": MAX_CONCORRENCIA},
        return_exceptions=True,
    )

    eval_data = []
    for chunk, qa in zip(amostra, perguntas_respostas):
        if isinstance(qa, Exception):
            continue
        eval_data.append({
            "query": qa.pergunta,
            "answer": qa.resposta,
            "chunk_hash": id_conteudo(chunk.page_content),
            "fonte": chunk.metadata.get("source"),
            "pagina": chunk.metadata.get("page"),
        })

    with open(ARQ_QA, "w", encoding="utf-8") as file:
        json.dump(eval_data, file, indent=2, ensure_ascii=False)
    print(f"Dataset gerado: {len(eval_data)} perguntas")


# Avaliação: predições do RAG (só a pergunta entra no pipeline)
saidas = graph.batch(
    [{"question": d["query"]} for d in eval_data],
    config={"max_concurrency": MAX_CONCORRENCIA},
)
predictions = [s["answer"] for s in saidas]


# Avaliação: juiz
# Define a estrutura de saída para o veredito do juiz
class Veredito(BaseModel):
    veredito: Literal["CORRETO", "INCORRETO"]
    justificativa: str = Field(description="Uma frase explicando o veredito")


juiz_prompt = ChatPromptTemplate.from_template(
    "Você é um avaliador. Compare a resposta do sistema com a resposta de referência.\n"
    "Marque CORRETO se a resposta do sistema contém a informação essencial da referência e não a "
    "contradiz (a redação pode ser diferente). Caso contrário, marque INCORRETO.\n\n"
    "Pergunta: {pergunta}\n"
    "Resposta de referência: {referencia}\n"
    "Resposta do sistema: {resposta}"
)
juiz_chain = juiz_prompt | llm.with_structured_output(Veredito)

graded_outputs = juiz_chain.batch(
    [
        {"pergunta": d["query"], "referencia": d["answer"], "resposta": p}
        for d, p in zip(eval_data, predictions)
    ],
    config={"max_concurrency": MAX_CONCORRENCIA},
    return_exceptions=True,
)


# Avaliação Métricas: acurácia, hit@k, acurácia com hit@k
resultados = []
for d, s, g in zip(eval_data, saidas, graded_outputs):
    hashes_recuperados = {id_conteudo(doc.page_content) for doc in s["context"]}
    falhou = isinstance(g, Exception)
    resultados.append({
        **d,
        "resposta_sistema": s["answer"],
        "recuperou_chunk_origem": d["chunk_hash"] in hashes_recuperados,
        "veredito": None if falhou else g.veredito,
        "justificativa": None if falhou else g.justificativa,
    })

validos = [r for r in resultados if r["veredito"] is not None]
corrects = [r for r in validos if r["veredito"] == "CORRETO"]
com_hit = [r for r in validos if r["recuperou_chunk_origem"]]
corrects_com_hit = [r for r in com_hit if r["veredito"] == "CORRETO"]

print(f"Perguntas avaliadas:                  {len(validos)}")
if validos:
    print(f"Taxa de acerto da resposta:           {len(corrects) / len(validos):.1%}")
    print(f"Recuperação (hit@{K}):                 {len(com_hit) / len(validos):.1%}")
    if com_hit:
        print(f"Acerto quando o chunk foi recuperado: {len(corrects_com_hit) / len(com_hit):.1%}")
else:
    print("Nenhuma pergunta foi avaliada pelo juiz (verifique erros/limite de requisições do LLM)")

with open(ARQ_RESULTADOS, "w", encoding="utf-8") as file:
    json.dump(resultados, file, indent=2, ensure_ascii=False)
print(f"Resultados detalhados salvos em {ARQ_RESULTADOS}")