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

print(f"Pages: {len(pages)}")
print(f"Chunks: {len(chunks)}")


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

print(f"Embedding shape: {embeddings.shape}")


# --------------------------------------------------
# 3. Build vector store
# --------------------------------------------------

vector_store = VectorStore(
    dimension=embeddings.shape[1]
)

vector_store.add_embeddings(
    embeddings
)


# --------------------------------------------------
# 4. Build retriever
# --------------------------------------------------

retriever = Retriever(
    embedding_model=embedding_model,
    vector_store=vector_store,
    chunks=chunks
)


# --------------------------------------------------
# 5. Build RAG pipeline
# --------------------------------------------------

rag = RAGPipeline(
    retriever=retriever
)


# --------------------------------------------------
# 6. Ask question
# --------------------------------------------------

query = "What is the main purpose of this research?"

result = rag.generate_answer(
    query,
    top_k=3
)


# --------------------------------------------------
# 7. Display answer
# --------------------------------------------------

print("\n" + "=" * 80)
print("ANSWER")
print("=" * 80)

print(result["answer"])


# --------------------------------------------------
# 8. Display sources
# --------------------------------------------------

print("\n" + "=" * 80)
print("SOURCES")
print("=" * 80)

for source in result["sources"]:

    print(
        f"- {source['source']} | "
        f"Page {source['page_number']} | "
        f"Score {source['score']:.4f}"
    )