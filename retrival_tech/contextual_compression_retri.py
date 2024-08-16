from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from decouple import config

Text = [
    "Medical Rehabilitation: The government provides access to medical facilities for rehabilitation, including physical therapy, occupational therapy, and psychological counseling.",
    "Assistive Devices: Provision of aids and appliances to improve mobility and functionality, like prosthetics, wheelchairs, hearing aids, etc.",
    "Education and Employment: Special education programs and reservation of jobs in government sectors to enhance social inclusion and employment opportunities.",
    "Financial Support: Various schemes provide financial assistance to individuals with disabilities and their families.",
    "Legal Rights and Protections: Ensuring non-discrimination and promoting accessibility in public spaces and services."
]
meta_data = [{"source": "document 1", "page": 1},
             {"source": "document 2", "page": 2},
             {"source": "document 3", "page": 3},
             {"source": "document 4", "page": 4}]

# embedding functions
emb_func = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_db = Chroma.from_texts(
    texts=Text,
    embedding=emb_func,
    metadatas=meta_data
)

llm = OpenAI(temperature=0, openai_api_key=config("OPENAI_API_KEY"))
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_db.as_retriever()
)

compressed_docs = compression_retriever.get_relevant_documents("what for financial support")

print(compressed_docs)
