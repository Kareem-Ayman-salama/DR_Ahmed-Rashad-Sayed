"""Train a seed reward ranker for simulated coach actions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class RewardRankerPaths:
    """Filesystem paths for reward-ranker training."""

    phase_root: Path
    preferences_path: Path
    model_path: Path
    metrics_path: Path
    report_path: Path


def main() -> None:
    """Train and save the seed reward ranker."""
    phase_root = Path(__file__).resolve().parent
    paths = RewardRankerPaths(
        phase_root=phase_root,
        preferences_path=phase_root / "data" / "coach_reaction_rlhf_preference_seed.csv",
        model_path=phase_root / "models" / "seed_reward_ranker.joblib",
        metrics_path=phase_root / "data" / "seed_reward_ranker_metrics.json",
        report_path=phase_root / "outputs" / "seed_reward_ranker_report.md",
    )
    preferences = pd.read_csv(paths.preferences_path)
    training = build_binary_training_rows(preferences)
    model, metrics = train_ranker(training)
    paths.model_path.parent.mkdir(parents=True, exist_ok=True)
    paths.report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, paths.model_path)
    paths.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    paths.report_path.write_text(build_report(paths, metrics), encoding="utf-8")
    print("Seed reward ranker trained")
    print(f"Model: {paths.model_path}")
    print(f"Metrics: {paths.metrics_path}")
    print(f"Report: {paths.report_path}")


def build_binary_training_rows(preferences: pd.DataFrame) -> pd.DataFrame:
    """Convert chosen/rejected preference rows into binary training rows."""
    chosen = pd.DataFrame(
        {
            "text": preferences["state"].astype(str) + " ACTION: " + preferences["chosen_action"].astype(str),
            "label": 1,
        }
    )
    rejected = pd.DataFrame(
        {
            "text": preferences["state"].astype(str) + " ACTION: " + preferences["rejected_action"].astype(str),
            "label": 0,
        }
    )
    return pd.concat([chosen, rejected], ignore_index=True)


def train_ranker(training: pd.DataFrame) -> tuple[Pipeline, dict[str, object]]:
    """Train text reward ranker and return compact metrics."""
    train_df, test_df = train_test_split(
        training,
        test_size=0.20,
        random_state=42,
        stratify=training["label"],
    )
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(train_df["text"], train_df["label"])
    proba = model.predict_proba(test_df["text"])[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "accuracy": round(float(accuracy_score(test_df["label"], pred)), 4),
        "roc_auc": round(float(roc_auc_score(test_df["label"], proba)), 4),
        "classification_report": classification_report(test_df["label"], pred, output_dict=True),
        "important_note": (
            "This ranker is trained on simulated seed preferences. Replace or recalibrate "
            "with expert-reviewed preferences before making scientific claims."
        ),
    }
    return model, metrics


def build_report(paths: RewardRankerPaths, metrics: dict[str, object]) -> str:
    """Build reward-ranker report."""
    return "\n".join(
        [
            "# Seed Reward Ranker Report",
            "",
            "## Status",
            "",
            "- Trained a first reward/ranking model from the simulated preference seed.",
            "- The model learns to score chosen coach reactions above rejected unsafe alternatives.",
            "- This is a pipeline readiness step, not final RLHF evidence.",
            "",
            "## Outputs",
            "",
            f"- Model: `{paths.model_path}`",
            f"- Metrics: `{paths.metrics_path}`",
            "",
            "## Metrics",
            "",
            f"- Train rows: {metrics['train_rows']}",
            f"- Test rows: {metrics['test_rows']}",
            f"- Accuracy: {metrics['accuracy']}",
            f"- ROC-AUC: {metrics['roc_auc']}",
            "",
            "## Important Limitation",
            "",
            str(metrics["important_note"]),
        ]
    )


if __name__ == "__main__":
    main()
