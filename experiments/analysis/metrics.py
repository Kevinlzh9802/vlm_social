from dataclasses import dataclass
import re
from typing import Sequence

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_TURNOVER_THRESHOLDS = (0.3, 0.5, 0.7, 0.9)
CUDA_OOM_PATTERN = re.compile(
    r"(cuda.*out of memory|outofmemoryerror|torch\.cuda\..*outofmemory|cuda out of memory)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class UtteranceMetrics:
    clip_count: int
    clip_to_final_similarities: list[float]
    neighboring_similarities: list[float]


def is_error_text(text: str) -> bool:
    normalized = text.strip()
    return normalized.startswith("[ERROR]") or CUDA_OOM_PATTERN.search(normalized) is not None


def has_error_clip(ordered_clips: Sequence[tuple[int, str]]) -> bool:
    return any(is_error_text(text) for _, text in ordered_clips)


def filter_valid_texts(ordered_clips: Sequence[tuple[int, str]]) -> list[str]:
    return [
        text
        for _, text in ordered_clips
        if text.strip() and not is_error_text(text)
    ]


def compute_utterance_metrics(
    model: SentenceTransformer,
    ordered_clips: Sequence[tuple[int, str]],
) -> UtteranceMetrics | None:
    if has_error_clip(ordered_clips):
        return None

    texts = filter_valid_texts(ordered_clips)
    if len(texts) < 2:
        return None

    embeddings = model.encode(texts, convert_to_numpy=True)
    final_embedding = embeddings[-1].reshape(1, -1)
    clip_to_final = cosine_similarity(embeddings, final_embedding).ravel().tolist()
    neighboring = cosine_similarity(embeddings[:-1], embeddings[1:]).diagonal().tolist()
    clip_count = len(texts)

    return UtteranceMetrics(
        clip_count=clip_count,
        clip_to_final_similarities=[float(value) for value in clip_to_final],
        neighboring_similarities=[float(value) for value in neighboring],
    )


def compute_semantic_turnover(
    neighboring_similarities: Sequence[float],
    turnover_threshold: float,
) -> int:
    return sum(1 for similarity in neighboring_similarities if similarity < turnover_threshold)


def compute_semantic_turnover_ratio(
    clip_count: int,
    neighboring_similarities: Sequence[float],
    turnover_threshold: float,
) -> float:
    semantic_turnover = compute_semantic_turnover(
        neighboring_similarities=neighboring_similarities,
        turnover_threshold=turnover_threshold,
    )
    return float(semantic_turnover / clip_count)


def compute_weighted_average_st_position(
    clip_count: int,
    neighboring_similarities: Sequence[float],
) -> float | None:
    weighted_sum = 0.0
    total_weight = 0.0

    for position_index, similarity in enumerate(neighboring_similarities, start=1):
        turnover_weight = max(0.0, 1.0 - similarity)
        if turnover_weight <= 0:
            continue
        weighted_sum += turnover_weight * (position_index / clip_count)
        total_weight += turnover_weight

    if total_weight == 0:
        return None
    return float(weighted_sum / total_weight)
