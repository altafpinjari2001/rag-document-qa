"""
RAG Document Q&A System - Evaluation Module.

Evaluates RAG pipeline quality using custom metrics.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from .pipeline import RAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Evaluation result for a single Q&A pair."""

    question: str
    expected_answer: str
    generated_answer: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


FAITHFULNESS_PROMPT = """Given the context and the generated answer, rate the faithfulness of the answer on a scale of 0.0 to 1.0.
Faithfulness measures whether the answer is grounded in the provided context (no hallucinations).

Context: {context}
Answer: {answer}

Rate (0.0 to 1.0) and respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}"""

RELEVANCY_PROMPT = """Given the question and the generated answer, rate the relevancy of the answer on a scale of 0.0 to 1.0.
Relevancy measures whether the answer actually addresses the question asked.

Question: {question}
Answer: {answer}

Rate (0.0 to 1.0) and respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}"""


class RAGEvaluator:
    """Evaluates RAG pipeline performance."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        eval_model: str = "gpt-4o-mini",
    ):
        self.pipeline = pipeline
        self.eval_llm = ChatOpenAI(
            model=eval_model, temperature=0.0
        )

    def _score_faithfulness(
        self, context: str, answer: str
    ) -> float:
        """Score the faithfulness of answer to context."""
        prompt = ChatPromptTemplate.from_template(
            FAITHFULNESS_PROMPT
        )
        chain = prompt | self.eval_llm
        response = chain.invoke({
            "context": context, "answer": answer
        })
        try:
            result = json.loads(response.content)
            return float(result["score"])
        except (json.JSONDecodeError, KeyError):
            return 0.0

    def _score_relevancy(
        self, question: str, answer: str
    ) -> float:
        """Score the relevancy of answer to question."""
        prompt = ChatPromptTemplate.from_template(
            RELEVANCY_PROMPT
        )
        chain = prompt | self.eval_llm
        response = chain.invoke({
            "question": question, "answer": answer
        })
        try:
            result = json.loads(response.content)
            return float(result["score"])
        except (json.JSONDecodeError, KeyError):
            return 0.0

    def evaluate(
        self, test_dataset_path: str
    ) -> list[EvalResult]:
        """
        Run evaluation on a test dataset.

        Args:
            test_dataset_path: Path to JSON file with Q&A pairs.
                Format: [{"question": "...", "answer": "..."}]

        Returns:
            List of EvalResult objects.
        """
        with open(test_dataset_path) as f:
            test_data = json.load(f)

        results = []
        for item in test_data:
            question = item["question"]
            expected = item["answer"]

            # Get RAG response
            response = self.pipeline.query(question)
            context = "\n".join(response.context_chunks)

            # Compute metrics
            faithfulness = self._score_faithfulness(
                context, response.answer
            )
            relevancy = self._score_relevancy(
                question, response.answer
            )

            result = EvalResult(
                question=question,
                expected_answer=expected,
                generated_answer=response.answer,
                faithfulness=faithfulness,
                answer_relevancy=relevancy,
                context_precision=0.0,  # Requires ground truth
                context_recall=0.0,
            )
            results.append(result)

            logger.info(
                f"Evaluated: faithfulness={faithfulness:.2f}, "
                f"relevancy={relevancy:.2f}"
            )

        return results

    def summary(self, results: list[EvalResult]) -> dict:
        """Compute aggregate metrics from evaluation results."""
        n = len(results)
        if n == 0:
            return {}

        return {
            "num_questions": n,
            "avg_faithfulness": sum(
                r.faithfulness for r in results
            ) / n,
            "avg_answer_relevancy": sum(
                r.answer_relevancy for r in results
            ) / n,
            "avg_context_precision": sum(
                r.context_precision for r in results
            ) / n,
            "avg_context_recall": sum(
                r.context_recall for r in results
            ) / n,
        }
