from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# Base dir = repo root (two levels up from this script)
BASE_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = BASE_DIR / "src" / "vectorstore"

# Load the saved vectorstore
vectorstore = FAISS.load_local(
    str(VECTORSTORE_DIR), embedding_model, allow_dangerous_deserialization=True
)
# Create a retriever
retriever = vectorstore.as_retriever()

# Test a query
query = "What columns are in the sap_oitm table?"
docs = retriever.get_relevant_documents(query)

# Display results
for i, doc in enumerate(docs, 1):
    print(f"Match {i}:\n{doc.page_content}\n")
