from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrivalQA
from langchain.chat_models import ChatOpenAI
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

QA_prompt = PromptTemplate(
    template="""Use the following piece of context to answer the user question
    Context:{text}
    Question : {question}
    Answer:""",
    input_variables=["text", "question"]

)

llm = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"),temperature=0)
qa_chain = RetrivalQA.from_chain_type(
    llm=llm,
    retrival=vector_db.as_retriever(),
    return_source_documents=True,
    chain_type= "refine"
)

question = "what for financial support"

response = qa_chain({"query": question})

print(response["result"])

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

print(response["source_documents"])
