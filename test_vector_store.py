from src.pdf_processor import extract_text_from_pdf
from src.text_chunker import create_chunks
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


# --------------------------------------------------
# 1. Extract PDF text
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
# 2. Generate embeddings
# --------------------------------------------------

embedding_model = EmbeddingModel()

embeddings = embedding_model.encode_texts(texts)

print("Embedding shape:", embeddings.shape)


# --------------------------------------------------
# 3. Create FAISS vector store
# --------------------------------------------------

dimension = embeddings.shape[1]

vector_store = VectorStore(
    dimension=dimension
)

vector_store.add_embeddings(embeddings)

print("Vectors stored:", vector_store.size())


# --------------------------------------------------
# 4. Ask a question
# --------------------------------------------------

query = "What is the main purpose of this research?"

query_embedding = embedding_model.encode_query(query)


# --------------------------------------------------
# 5. Search
# --------------------------------------------------

scores, indices = vector_store.search(
    query_embedding,
    top_k=3
)


# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print("\nQuery:")
print(query)

print("\nTop Results:")

for rank, (score, index) in enumerate(
    zip(scores, indices),
    start=1
):

    chunk = chunks[index]

    print("\n" + "=" * 70)

    print(f"Rank: {rank}")
    print(f"Similarity Score: {score:.4f}")
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Source: {chunk['source']}")
    print(f"Page: {chunk['page_number']}")

    print("\nText:")
    print(chunk["text"][:700])