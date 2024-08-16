from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

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

response = vector_db.similarity_search(
    query="what is for improve mobility", k=2
)
print(response)