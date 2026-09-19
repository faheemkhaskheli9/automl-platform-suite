"""Feature-app registry backing the dashboard's "pick a feature" shell
(README.md Section 2/4, issue #4).

Mirrors the string-keyed registry pattern docs/architecture.md already
earmarks for automl_core's feature apps (itself mirroring
medical-imaging-suite's BaseImagingTask / @register_task) -- applied here one
layer up, to whole feature apps (Train/Tune/Evaluate), so a later phase plugs
in a feature by adding one `register_feature(...)` call instead of editing
the dashboard template.

Train now has its own feature app (`train`, issue #5: target-column
selection + task-type detection, on top of automl_core's upload/preview
flow). Tune and Evaluate still register with `url_name=None` so the
dashboard shows them as "coming soon" instead of a dangling/guessed link,
until Phase 3/4 land their own views (README.md Section 5).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureCard:
    key: str
    label: str
    description: str
    url_name: str | None = None  # None => not implemented yet ("coming soon")

    @property
    def is_available(self) -> bool:
        return self.url_name is not None


_REGISTRY: dict[str, FeatureCard] = {}


def register_feature(card: FeatureCard) -> FeatureCard:
    if card.key in _REGISTRY:
        raise ValueError(f"duplicate feature key: {card.key!r}")
    _REGISTRY[card.key] = card
    return card


def all_features() -> list[FeatureCard]:
    """Registered features in registration order (Train, then Tune/Evaluate
    in their planned build order -- README.md Section 5)."""
    return list(_REGISTRY.values())


register_feature(
    FeatureCard(
        key="train",
        label="Train",
        description=(
            "Upload a dataset, pick a target column, and let the platform "
            "detect classification vs. regression before training."
        ),
        url_name="train:select_target",
    )
)
register_feature(
    FeatureCard(
        key="tune",
        label="Tune",
        description=(
            "YAML-driven search spaces with Optuna hyperparameter "
            "optimization and MLflow-tracked runs, ranked on a leaderboard."
        ),
    )
)
register_feature(
    FeatureCard(
        key="evaluate",
        label="Evaluate",
        description=(
            "One dashboard across classification/regression/CV/LLM run "
            "metrics, read through the shared unified metrics schema."
        ),
    )
)
