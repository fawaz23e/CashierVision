from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


def format_topk_predictions(predictions: list[Prediction], k: int = 3) -> list[str]:
    top_predictions = sorted(predictions, key=lambda item: item.confidence, reverse=True)[:k]
    return [f"{item.label}: {item.confidence:.1%}" for item in top_predictions]


def confidence_message(confidence: float) -> str:
    if confidence >= 0.80:
        return "High confidence. Still verify PLU details before checkout."
    if confidence >= 0.50:
        return "Medium confidence. Review the top suggestions."
    return "Low confidence. Ask the cashier to verify manually."

