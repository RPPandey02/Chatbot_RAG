from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from decouple import config

# embedding functions
emb_func = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="../vector_db",
    collection_name="CDA_rules",
    embedding_function=emb_func
)

llm = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"),temperature=0.6)

memory = ConversationBufferMemory(
    return_messages=True, memory_key="chat_history"
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    retriever=vector_db.as_retriever(search_kwargs={"fetch_k":4,"k":3},search_type="mmr"),
    chain_type="refine"
    )


def rag_func(question:str) ->str:

    response = qa_chain({"question":question})

    return response.get("answer")