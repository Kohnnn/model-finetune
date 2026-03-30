from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from rag import prepare_passage_text, prepare_query_text


class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    @staticmethod
    def _to_lists(vectors) -> list[list[float]]:
        return [vector.tolist() for vector in vectors]

    def encode_passages(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load_model()
        embeddings = list(
            model.embed(
                [prepare_passage_text(text) for text in texts],
                batch_size=32,
            )
        )
        return self._to_lists(embeddings)

    def encode_query(self, query: str) -> list[float]:
        return self.encode_queries([query])[0]

    def encode_queries(self, queries: Sequence[str]) -> list[list[float]]:
        model = self._load_model()
        embeddings = list(
            model.embed(
                [prepare_query_text(query) for query in queries],
                batch_size=32,
            )
        )
        return self._to_lists(embeddings)


@lru_cache
def get_embedding_model(model_name: str) -> EmbeddingModel:
    return EmbeddingModel(model_name)
