"""Train the simulated coach agent policy models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class TrainPaths:
    """Training input/output paths."""

    phase_root: Path
    rare_path: Path
    policy_path: Path
    bundle_path: Path
    metrics_path: Path
    predictions_path: Path
    report_path: Path


TEXT_FEATURE = "rare_event_reason"
NUMERIC_FEATURES = [
    "event_count",
    "pass_count",
    "avg_pass_length",
    "pass_success_rate",
    "forward_rate",
    "long_pass_rate",
    "n_goals",
    "n_own_goals",
    "n_red_cards",
    "n_second_yellows",
    "n_subs",
    "n_injuries",
    "n_unexpected_goals",
    "n_penalty_goals",
    "n_error_events",
    "tactical_shift_score",
    "delta_pass_count",
    "delta_avg_pass_length",
    "delta_pass_success_rate",
    "delta_forward_rate",
    "delta_long_pass_rate",
]
CATEGORICAL_FEATURES = [
    "has_formation_change",
    "is_rare_red_card",
    "is_rare_second_yellow",
    "is_rare_subs",
    "is_rare_unexpected_goal",
    "is_rare_own_goal",
    "is_rare_formation_change",
    "is_rare_penalty_goal",
    "is_rare_error_event",
]


def main() -> None:
    """Train and save simulated coach policy models."""
    phase_root = Path(__file__).resolve().parent
    paths = TrainPaths(
        phase_root=phase_root,
        rare_path=phase_root / "data" / "coach_reaction_dataset_rare_only.csv",
        policy_path=phase_root / "data" / "reward_guided_policy_decisions.csv",
        bundle_path=phase_root / "models" / "simulated_coach_agent_bundle.joblib",
        metrics_path=phase_root / "data" / "simulated_coach_agent_training_metrics.json",
        predictions_path=phase_root / "data" / "simulated_coach_agent_holdout_predictions.csv",
        report_path=phase_root / "outputs" / "simulated_coach_agent_training_report.md",
    )
    paths.bundle_path.parent.mkdir(parents=True, exist_ok=True)
    paths.report_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_training_dataset(paths)
    train_df, test_df = grouped_split(dataset)
    action_model = build_classifier()
    urgency_model = build_classifier()
    action_model.fit(train_df[feature_columns()], train_df["selected_policy_action"])
    urgency_model.fit(train_df[feature_columns()], train_df["urgency"])

    predictions = evaluate_models(action_model, urgency_model, test_df)
    metrics = build_metrics(test_df, predictions)
    predictions.to_csv(paths.predictions_path, index=False, encoding="utf-8")
    paths.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "action_model": action_model,
            "urgency_model": urgency_model,
            "feature_columns": feature_columns(),
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "text_feature": TEXT_FEATURE,
            "metrics": metrics,
            "note": (
                "Trained on simulated expert/reward-guided labels. Use as engineering agent "
                "prototype until real expert labels are available."
            ),
        },
        paths.bundle_path,
    )
    paths.report_path.write_text(build_report(paths, metrics), encoding="utf-8")
    print("Simulated coach agent model trained")
    print(f"Bundle: {paths.bundle_path}")
    print(f"Metrics: {paths.metrics_path}")
    print(f"Predictions: {paths.predictions_path}")
    print(f"Report: {paths.report_path}")


def load_training_dataset(paths: TrainPaths) -> pd.DataFrame:
    """Load and merge rare-event features with reward-guided labels."""
    rare = pd.read_csv(paths.rare_path)
    policy = pd.read_csv(paths.policy_path)
    key_cols = ["team", "match_id", "time_window", "rare_event_reason"]
    dataset = rare.merge(
        policy[
            key_cols
            + [
                "selected_policy_action",
                "selected_reward_score",
                "rejected_action",
                "rejected_reward_score",
                "urgency",
            ]
        ],
        on=key_cols,
        how="inner",
    )
    if len(dataset) != len(rare):
        raise RuntimeError("Policy labels did not merge cleanly with rare-event rows.")
    dataset[TEXT_FEATURE] = dataset[TEXT_FEATURE].fillna("Unknown")
    for column in CATEGORICAL_FEATURES:
        dataset[column] = dataset[column].astype(str)
    return dataset


def grouped_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by match IDs when possible."""
    unique_matches = pd.Series(dataset["match_id"].unique())
    train_matches, test_matches = train_test_split(
        unique_matches,
        test_size=0.20,
        random_state=42,
    )
    train_df = dataset[dataset["match_id"].isin(set(train_matches))].copy()
    test_df = dataset[dataset["match_id"].isin(set(test_matches))].copy()
    return train_df, test_df


def feature_columns() -> list[str]:
    """Return model feature columns."""
    return [TEXT_FEATURE] + NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_classifier() -> Pipeline:
    """Build a mixed text/tabular classifier pipeline."""
    preprocess = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2), TEXT_FEATURE),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=0.0,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=160,
                    learning_rate=0.08,
                    random_state=42,
                    l2_regularization=0.05,
                ),
            ),
        ]
    )


def evaluate_models(action_model: Pipeline, urgency_model: Pipeline, test_df: pd.DataFrame) -> pd.DataFrame:
    """Create holdout predictions."""
    predictions = test_df[
        [
            "team",
            "match_id",
            "time_window",
            "rare_event_reason",
            "selected_policy_action",
            "urgency",
            "selected_reward_score",
            "rejected_reward_score",
        ]
    ].copy()
    predictions["predicted_policy_action"] = action_model.predict(test_df[feature_columns()])
    predictions["predicted_urgency"] = urgency_model.predict(test_df[feature_columns()])
    action_proba = action_model.predict_proba(test_df[feature_columns()])
    urgency_proba = urgency_model.predict_proba(test_df[feature_columns()])
    predictions["policy_action_confidence"] = np.max(action_proba, axis=1)
    predictions["urgency_confidence"] = np.max(urgency_proba, axis=1)
    predictions["policy_action_correct"] = (
        predictions["predicted_policy_action"] == predictions["selected_policy_action"]
    )
    predictions["urgency_correct"] = predictions["predicted_urgency"] == predictions["urgency"]
    return predictions


def build_metrics(test_df: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, object]:
    """Build training/evaluation metrics."""
    action_true = predictions["selected_policy_action"]
    action_pred = predictions["predicted_policy_action"]
    urgency_true = predictions["urgency"]
    urgency_pred = predictions["predicted_urgency"]
    return {
        "train_label_source": "simulated_expert_reward_guided_policy",
        "holdout_rows": int(len(predictions)),
        "holdout_matches": int(test_df["match_id"].nunique()),
        "policy_action_accuracy": round(float(accuracy_score(action_true, action_pred)), 4),
        "policy_action_macro_f1": round(float(f1_score(action_true, action_pred, average="macro")), 4),
        "urgency_accuracy": round(float(accuracy_score(urgency_true, urgency_pred)), 4),
        "urgency_macro_f1": round(float(f1_score(urgency_true, urgency_pred, average="macro")), 4),
        "avg_policy_action_confidence": round(float(predictions["policy_action_confidence"].mean()), 4),
        "avg_urgency_confidence": round(float(predictions["urgency_confidence"].mean()), 4),
        "policy_action_report": classification_report(action_true, action_pred, output_dict=True),
        "urgency_report": classification_report(urgency_true, urgency_pred, output_dict=True),
        "important_note": (
            "These are holdout metrics against simulated reward-guided labels. They close the engineering "
            "loop, but real expert-reviewed labels are still required for publishable claims."
        ),
    }


def build_report(paths: TrainPaths, metrics: dict[str, object]) -> str:
    """Build training report."""
    return "\n".join(
        [
            "# Simulated Coach Agent Training Report",
            "",
            "## Status",
            "",
            "- Trained a runnable simulated coach agent model bundle.",
            "- The agent learns policy action and urgency from rare-event tactical-window features.",
            "- Labels come from the simulated expert reward-guided policy selector.",
            "",
            "## Outputs",
            "",
            f"- Model bundle: `{paths.bundle_path}`",
            f"- Metrics: `{paths.metrics_path}`",
            f"- Holdout predictions: `{paths.predictions_path}`",
            "",
            "## Holdout Metrics",
            "",
            f"- Holdout rows: {metrics['holdout_rows']}",
            f"- Holdout matches: {metrics['holdout_matches']}",
            f"- Policy action accuracy: {metrics['policy_action_accuracy']}",
            f"- Policy action macro-F1: {metrics['policy_action_macro_f1']}",
            f"- Urgency accuracy: {metrics['urgency_accuracy']}",
            f"- Urgency macro-F1: {metrics['urgency_macro_f1']}",
            f"- Avg policy confidence: {metrics['avg_policy_action_confidence']}",
            f"- Avg urgency confidence: {metrics['avg_urgency_confidence']}",
            "",
            "## Important Limitation",
            "",
            str(metrics["important_note"]),
        ]
    )


if __name__ == "__main__":
    main()
