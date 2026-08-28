from src.pdf_processor import extract_text_from_pdf
from src.text_chunker import create_chunks
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.rag_pipeline import RAGPipeline


# --------------------------------------------------
# 1. Load PDF
# --------------------------------------------------

pdf_path = "data/sample_papers/research_paper1.pdf"

pages = extract_text_from_pdf(pdf_path)

chunks = create_chunks(
    pages,
    chunk_size=500,
    overlap=100
)


# --------------------------------------------------
# 2. Generate embeddings
# --------------------------------------------------

embedding_model = EmbeddingModel()

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = embedding_model.encode_texts(
    texts
)


# --------------------------------------------------
# 3. Create FAISS store
# --------------------------------------------------

vector_store = VectorStore(
    dimension=embeddings.shape[1]
)

vector_store.add_embeddings(
    embeddings
)


# --------------------------------------------------
# 4. Create retriever
# --------------------------------------------------

retriever = Retriever(
    embedding_model=embedding_model,
    vector_store=vector_store,
    chunks=chunks
)


# --------------------------------------------------
# 5. Create RAG pipeline
# --------------------------------------------------

rag = RAGPipeline(
    retriever=retriever
)


# --------------------------------------------------
# 6. Ask question
# --------------------------------------------------

query = "What is the main purpose of this research?"

retrieved_chunks = rag.retrieve_context(
    query,
    top_k=3
)


# --------------------------------------------------
# 7. Build context
# --------------------------------------------------

context = rag.build_context(
    retrieved_chunks
)


# --------------------------------------------------
# 8. Build prompt
# --------------------------------------------------

prompt = rag.build_prompt(
    query,
    context
)


print("\n" + "=" * 80)
print("RETRIEVED CONTEXT")
print("=" * 80)

print(context)


print("\n" + "=" * 80)
print("GENERATED PROMPT")
print("=" * 80)

print(prompt)