import os
from typing import List, Dict

from groq import Groq
from dotenv import load_dotenv

from src.prompts import build_rag_prompt


load_dotenv()


class RAGPipeline:

    def __init__(self, retriever):

        self.retriever = retriever

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment."
            )

        self.client = Groq(api_key=api_key)

    def retrieve_context(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:

        return self.retriever.retrieve(
            query,
            top_k=top_k
        )

    def build_context(
        self,
        retrieved_chunks: List[Dict]
    ) -> str:

        context_parts = []

        for i, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            context_parts.append(
                f"""
SOURCE {i}
Document: {chunk['source']}
Page: {chunk['page_number']}
Similarity Score: {chunk['score']:.4f}

Content:
{chunk['text']}
"""
            )

        return "\n".join(context_parts)

    def build_prompt(
        self,
        query: str,
        context: str
    ) -> str:
        return build_rag_prompt(query, context)

    def generate_answer(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict:

        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve_context(
            query,
            top_k=top_k
        )

        # Build context
        context = self.build_context(
            retrieved_chunks
        )

        # Build RAG prompt
        prompt = self.build_prompt(
            query,
            context
        )

        # Generate answer using Groq (Llama 3)
        response = self.client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": retrieved_chunks
        }