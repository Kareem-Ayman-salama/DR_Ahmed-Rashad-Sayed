"""Simulate expert review labels and build reviewed RLHF training data."""

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
class ReviewDecision:
    """Simulated expert review decision."""

    acceptance: str
    reward: float
    correction: str
    notes: str


PHASE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PHASE_ROOT / "data"
OUTPUT_DIR = PHASE_ROOT / "outputs"
MODEL_DIR = PHASE_ROOT / "models"

REVIEW_INPUT_PATH = DATA_DIR / "coach_reaction_expert_review_sample.csv"
REVIEW_OUTPUT_PATH = DATA_DIR / "coach_reaction_expert_review_sample_simulated_filled.csv"
PREFERENCES_INPUT_PATH = DATA_DIR / "coach_reaction_rlhf_preference_seed.csv"
PREFERENCES_OUTPUT_PATH = DATA_DIR / "coach_reaction_rlhf_preference_reviewed_simulated.csv"
MODEL_OUTPUT_PATH = MODEL_DIR / "simulated_expert_reward_ranker.joblib"
METRICS_OUTPUT_PATH = DATA_DIR / "simulated_expert_reward_ranker_metrics.json"
REPORT_OUTPUT_PATH = OUTPUT_DIR / "simulated_expert_reward_ranker_report.md"


def main() -> None:
    """Run simulated expert review and train reviewed reward ranker."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    review = pd.read_csv(REVIEW_INPUT_PATH)
    filled_review = fill_review(review)
    filled_review.to_csv(REVIEW_OUTPUT_PATH, index=False, encoding="utf-8")

    preferences = pd.read_csv(PREFERENCES_INPUT_PATH)
    reviewed_preferences = apply_review_policy(preferences)
    reviewed_preferences.to_csv(PREFERENCES_OUTPUT_PATH, index=False, encoding="utf-8")

    training = build_binary_training_rows(reviewed_preferences)
    model, metrics = train_ranker(training)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    METRICS_OUTPUT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    REPORT_OUTPUT_PATH.write_text(build_report(metrics), encoding="utf-8")

    print("Simulated expert review and reward model complete")
    print(f"Filled review: {REVIEW_OUTPUT_PATH}")
    print(f"Reviewed preferences: {PREFERENCES_OUTPUT_PATH}")
    print(f"Reward model: {MODEL_OUTPUT_PATH}")
    print(f"Report: {REPORT_OUTPUT_PATH}")


def fill_review(review: pd.DataFrame) -> pd.DataFrame:
    """Fill expert-review columns using deterministic tactical heuristics."""
    filled = review.copy()
    decisions = [simulate_review(row) for row in filled.to_dict(orient="records")]
    filled["expert_acceptance"] = [decision.acceptance for decision in decisions]
    filled["expert_reward"] = [decision.reward for decision in decisions]
    filled["expert_correction"] = [decision.correction for decision in decisions]
    filled["expert_notes"] = [decision.notes for decision in decisions]
    filled["review_status"] = "simulated_reviewed"
    filled["reviewer_name"] = "Simulated Tactical Reviewer"
    filled["review_date"] = "2026-08-13"
    return filled


def simulate_review(row: dict[str, object]) -> ReviewDecision:
    """Simulate tactical expert judgement for one row."""
    reason = str(row.get("rare_event_reason", ""))
    reaction_type = str(row.get("reaction_type", ""))
    evidence = str(row.get("evidence_confidence", "low"))
    shift = float(row.get("tactical_shift_score", 0.0) or 0.0)
    pass_success = float(row.get("pass_success_rate", 0.0) or 0.0)

    if "Red Card" in reason or "Second Yellow" in reason:
        return ReviewDecision(
            acceptance="good",
            reward=1.0,
            correction="",
            notes="Numerical disadvantage requires immediate compactness and reduced high pressing.",
        )
    if reaction_type == "goal_state_reset" and shift >= 8:
        return ReviewDecision(
            acceptance="good",
            reward=1.0,
            correction="",
            notes="Goal-state shock plus tactical shift supports stabilization before risk-taking.",
        )
    if reaction_type == "shape_transition_control":
        return ReviewDecision(
            acceptance="acceptable",
            reward=0.5,
            correction="Add an explicit communication step from captain/assistant coach to confirm roles.",
            notes="Formation-change response is suitable but should emphasize communication.",
        )
    if reaction_type == "substitution_management":
        return ReviewDecision(
            acceptance="acceptable",
            reward=0.5,
            correction="Specify whether the substitution is defensive protection, fresh pressing, or direct attack.",
            notes="Substitution logic is coherent but needs clearer role objective.",
        )
    if reaction_type == "error_suppression" and pass_success < 0.55:
        return ReviewDecision(
            acceptance="good",
            reward=1.0,
            correction="",
            notes="Low pass success and error cluster justify safer build-up and local support.",
        )
    if reaction_type == "error_suppression" and evidence == "low":
        return ReviewDecision(
            acceptance="acceptable",
            reward=0.5,
            correction="Add video/manual confirmation before making a substitution from error evidence alone.",
            notes="Error-cluster rule is useful, but evidence confidence is low.",
        )
    if shift >= 20:
        return ReviewDecision(
            acceptance="acceptable",
            reward=0.5,
            correction="Treat as high instability: simplify possession and confirm defensive distances.",
            notes="High tactical shift needs stabilizing reaction even if event type is less specific.",
        )
    return ReviewDecision(
        acceptance="acceptable",
        reward=0.5,
        correction="",
        notes="Reaction is reasonable as a first simulated label and should be confirmed by expert review.",
    )


def apply_review_policy(preferences: pd.DataFrame) -> pd.DataFrame:
    """Apply the same simulated policy to all preference rows."""
    reviewed = preferences.copy()
    rewards = []
    labels = []
    notes = []
    for row in reviewed.to_dict(orient="records"):
        reward, label, note = infer_preference_reward(row)
        rewards.append(reward)
        labels.append(label)
        notes.append(note)
    reviewed["expert_preference"] = labels
    reviewed["expert_reward_chosen"] = rewards
    reviewed["expert_reward_rejected"] = -0.5
    reviewed["expert_review_source"] = "simulated_tactical_review_policy"
    reviewed["expert_notes"] = notes
    reviewed["needs_expert_review"] = False
    return reviewed


def infer_preference_reward(row: dict[str, object]) -> tuple[float, str, str]:
    """Infer chosen-action reward from serialized state."""
    state = str(row.get("state", ""))
    if "Red Card" in state or "Second Yellow" in state:
        return 1.0, "chosen_strongly_better", "Compact response preferred for numerical disadvantage."
    if "shift=2" in state or "shift=3" in state:
        return 0.5, "chosen_better", "Moderate/low instability; chosen action remains safer."
    if "Error Cluster" in state:
        return 0.75, "chosen_better", "Error suppression is safer than continuing risky central build-up."
    if "Goal" in state:
        return 1.0, "chosen_strongly_better", "Goal-state reset is safer than chaotic overload."
    return 0.5, "chosen_better", "Chosen action is preferred as conservative simulated policy."


def build_binary_training_rows(preferences: pd.DataFrame) -> pd.DataFrame:
    """Build binary rows from reviewed preference data."""
    chosen = pd.DataFrame(
        {
            "text": preferences["state"].astype(str) + " ACTION: " + preferences["chosen_action"].astype(str),
            "label": 1,
            "weight": preferences["expert_reward_chosen"].clip(lower=0.1),
        }
    )
    rejected = pd.DataFrame(
        {
            "text": preferences["state"].astype(str) + " ACTION: " + preferences["rejected_action"].astype(str),
            "label": 0,
            "weight": 1.0,
        }
    )
    return pd.concat([chosen, rejected], ignore_index=True)


def train_ranker(training: pd.DataFrame) -> tuple[Pipeline, dict[str, object]]:
    """Train reward ranker on reviewed simulated preferences."""
    train_df, test_df = train_test_split(
        training,
        test_size=0.20,
        random_state=42,
        stratify=training["label"],
    )
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=25000, ngram_range=(1, 2), min_df=2)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(train_df["text"], train_df["label"], classifier__sample_weight=train_df["weight"])
    proba = model.predict_proba(test_df["text"])[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "accuracy": round(float(accuracy_score(test_df["label"], pred)), 4),
        "roc_auc": round(float(roc_auc_score(test_df["label"], proba)), 4),
        "classification_report": classification_report(test_df["label"], pred, output_dict=True),
        "important_note": (
            "This is trained on simulated expert labels. It is useful for pipeline progression, "
            "but final PhD claims require real expert review."
        ),
    }
    return model, metrics


def build_report(metrics: dict[str, object]) -> str:
    """Build report for simulated expert reward ranker."""
    return "\n".join(
        [
            "# Simulated Expert Reward Ranker Report",
            "",
            "## Status",
            "",
            "- Filled the expert-review sample with deterministic simulated tactical labels.",
            "- Rebuilt the RLHF/RLF preference seed with simulated expert rewards.",
            "- Trained a reward/ranking model for the next agent iteration.",
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
