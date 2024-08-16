from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

path_document = "E:\\Git\\chatbot_onboarding_employee\\document\\02_CDA.pdf"

# Loader
Loader = PyPDFLoader(path_document)

# splitting document
pages = Loader.load_and_split()

# embedding functions
emb_func = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# vector store
vectordb = Chroma.from_documents(
    documents=pages,
    embedding=emb_func,
    persist_directory="../vector_db",
    collection_name="CDA_rules"
)

# Persistent
vectordb.persist()
