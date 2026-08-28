from src.pdf_processor import extract_text_from_pdf
from src.text_chunker import create_chunks
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever


# --------------------------------------------------
# 1. Load document
# --------------------------------------------------

pdf_path = "data/sample_papers/research_paper1.pdf"

pages = extract_text_from_pdf(pdf_path)

chunks = create_chunks(
    pages,
    chunk_size=500,
    overlap=100
)

texts = [chunk["text"] for chunk in chunks]


# --------------------------------------------------
# 2. Create embeddings
# --------------------------------------------------

embedding_model = EmbeddingModel()

embeddings = embedding_model.encode_texts(texts)


# --------------------------------------------------
# 3. Create vector store
# --------------------------------------------------

vector_store = VectorStore(
    dimension=embeddings.shape[1]
)

vector_store.add_embeddings(embeddings)


# --------------------------------------------------
# 4. Create retriever
# --------------------------------------------------

retriever = Retriever(
    embedding_model=embedding_model,
    vector_store=vector_store,
    chunks=chunks
)


# --------------------------------------------------
# 5. Ask a question
# --------------------------------------------------

query = "What is the main purpose of this research?"

results = retriever.retrieve(
    query,
    top_k=3
)


# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print(f"\nQuery: {query}")

for rank, result in enumerate(results, start=1):

    print("\n" + "=" * 70)

    print(f"Rank: {rank}")
    print(f"Score: {result['score']:.4f}")
    print(f"Source: {result['source']}")
    print(f"Page: {result['page_number']}")
    print(f"Chunk ID: {result['chunk_id']}")

    print("\nText:")
    print(result["text"][:700])