import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension: int):
        """
        Create a FAISS index for normalized embeddings.

        Args:
            dimension: Number of dimensions in each embedding.
        """

        self.dimension = dimension

        # Inner Product works as cosine similarity
        # when embeddings are normalized.
        self.index = faiss.IndexFlatIP(dimension)

    def add_embeddings(self, embeddings: np.ndarray):
        """
        Add embedding vectors to the FAISS index.
        """

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embeddings with dimension "
                f"{self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        self.index.add(embeddings)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ):
        """
        Search for the most similar vectors.

        Returns:
            distances: Similarity scores
            indices: Indices of matching vectors
        """

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Expected query dimension "
                f"{self.dimension}, "
                f"got {query_embedding.shape[1]}"
            )

        top_k = min(top_k, self.index.ntotal)

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        return distances[0], indices[0]

    def size(self) -> int:
        """Return the number of vectors stored."""
        return self.index.ntotal