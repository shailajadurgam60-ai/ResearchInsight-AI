from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


class Retriever:

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        chunks: list[dict]
    ):
        """
        Connect the embedding model, vector store,
        and original chunks.
        """

        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.chunks = chunks

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        # 1. Convert query into an embedding
        query_embedding = self.embedding_model.encode_query(
            query
        )

        # 2. Search the vector store
        scores, indices = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )

        results = []

        # 3. Convert FAISS indices back to actual chunks
        for score, index in zip(scores, indices):

            if index < 0:
                continue

            chunk = self.chunks[index]

            results.append(
                {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "page_number": chunk["page_number"],
                    "chunk_id": chunk["chunk_id"],
                    "score": float(score)
                }
            )

        return results