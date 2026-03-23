"""
RAG Document Q&A System - Response Generator Module.

Generates answers using LLM with retrieved context and source citations.
"""

import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

logger = logging.getLogger(__name__)

# System prompt for RAG
RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context from documents. Follow these rules strictly:

1. **Answer ONLY from the provided context.** Do not use prior knowledge.
2. **Cite your sources.** Reference the source document and page number for each claim.
3. **If the context doesn't contain the answer**, say: "I couldn't find this information in the provided documents."
4. **Be concise but thorough.** Provide complete answers without unnecessary filler.
5. **Use structured formatting** (bullet points, headers) for complex answers."""

RAG_USER_PROMPT = """Context from documents:
---
{context}
---

Question: {question}

Please provide a detailed answer based on the context above, with source citations."""


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""

    answer: str
    sources: list[dict] = field(default_factory=list)
    context_chunks: list[str] = field(default_factory=list)
    model: str = ""
    usage: Optional[dict] = None


class ResponseGenerator:
    """Generates LLM responses grounded in retrieved context."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        api_key: Optional[str] = None,
        streaming: bool = False,
    ):
        self.model_name = model_name
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key,
            streaming=streaming,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RAG_SYSTEM_PROMPT),
                ("human", RAG_USER_PROMPT),
            ]
        )

        self.chain = self.prompt | self.llm

        logger.info(
            f"ResponseGenerator initialized "
            f"(model={model_name}, temp={temperature})"
        )

    def _format_context(self, documents: list[Document]) -> str:
        """Format retrieved documents into context string."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "N/A")
            context_parts.append(
                f"[Source {i}: {source}, Page {page}]\n" f"{doc.page_content}"
            )
        return "\n\n".join(context_parts)

    def _extract_sources(self, documents: list[Document]) -> list[dict]:
        """Extract source metadata from documents."""
        sources = []
        seen = set()

        for doc in documents:
            source_file = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "N/A")
            key = f"{source_file}:{page}"

            if key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "file": source_file,
                        "page": page,
                        "chunk_index": doc.metadata.get("chunk_index", -1),
                    }
                )
        return sources

    def generate(
        self,
        question: str,
        retrieved_documents: list[Document],
    ) -> RAGResponse:
        """
        Generate a response using retrieved context.

        Args:
            question: User's question.
            retrieved_documents: Relevant documents from retrieval.

        Returns:
            RAGResponse with answer, sources, and metadata.
        """
        if not retrieved_documents:
            return RAGResponse(
                answer="No relevant documents were found to answer "
                "your question. Please try uploading relevant "
                "documents first.",
                model=self.model_name,
            )

        context = self._format_context(retrieved_documents)

        response = self.chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        return RAGResponse(
            answer=response.content,
            sources=self._extract_sources(retrieved_documents),
            context_chunks=[doc.page_content for doc in retrieved_documents],
            model=self.model_name,
            usage=response.response_metadata.get("token_usage"),
        )

    async def agenerate_stream(
        self,
        question: str,
        retrieved_documents: list[Document],
    ) -> AsyncIterator[str]:
        """Stream the response token by token."""
        context = self._format_context(retrieved_documents)

        async for chunk in self.chain.astream(
            {
                "context": context,
                "question": question,
            }
        ):
            if chunk.content:
                yield chunk.content
