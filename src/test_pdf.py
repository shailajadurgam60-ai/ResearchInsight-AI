from src.pdf_processor import extract_text_from_pdf
from src.text_chunker import create_chunks
from src.embeddings import EmbeddingModel


pdf_path = "data/sample_papers/research_paper1.pdf"

pages = extract_text_from_pdf(pdf_path)

chunks = create_chunks(
    pages,
    chunk_size=500,
    overlap=100
)

texts = [chunk["text"] for chunk in chunks]

embedding_model = EmbeddingModel()

embeddings = embedding_model.encode_texts(texts)

print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)

print("\nFirst embedding:")
print(embeddings[0])