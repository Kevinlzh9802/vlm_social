from dataclasses import dataclass
from typing import Sequence

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


TURNOVER_THRESHOLD = 0.8


@dataclass(frozen=True)
class UtteranceMetrics:
    clip_count: int
    clip_to_final_similarities: list[float]
    neighboring_similarities: list[float]
    semantic_turnover: int
    semantic_turnover_ratio: float


def is_error_text(text: str) -> bool:
    return text.startswith("[ERROR]")


def filter_valid_texts(ordered_clips: Sequence[tuple[int, str]]) -> list[str]:
    return [
        text
        for _, text in ordered_clips
        if text.strip() and not is_error_text(text)
    ]


def compute_utterance_metrics(
    model: SentenceTransformer,
    ordered_clips: Sequence[tuple[int, str]],
    turnover_threshold: float = TURNOVER_THRESHOLD,
) -> UtteranceMetrics | None:
    texts = filter_valid_texts(ordered_clips)
    if len(texts) < 2:
        return None

    embeddings = model.encode(texts, convert_to_numpy=True)
    final_embedding = embeddings[-1].reshape(1, -1)
    clip_to_final = cosine_similarity(embeddings, final_embedding).ravel().tolist()
    neighboring = cosine_similarity(embeddings[:-1], embeddings[1:]).diagonal().tolist()

    semantic_turnover = sum(
        1 for similarity in neighboring if similarity < turnover_threshold
    )
    clip_count = len(texts)

    return UtteranceMetrics(
        clip_count=clip_count,
        clip_to_final_similarities=[float(value) for value in clip_to_final],
        neighboring_similarities=[float(value) for value in neighboring],
        semantic_turnover=semantic_turnover,
        semantic_turnover_ratio=float(semantic_turnover / clip_count),
    )
