from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
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

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="This is the source documents there are 4 main documents,  `document 1`, `document 2`, `document 3`, `document 4`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the details of Python",
        type="integer",
    ),
]

document_content_description = "disability"
llm = OpenAI(temperature=0, openai_api_key=config("OPENAI_API_KEY"))

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vector_db,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    verbose=True
)

docs = retriever.get_relevant_documents(
    "What was mentioned in the 4th document about  Python")
print(docs)
