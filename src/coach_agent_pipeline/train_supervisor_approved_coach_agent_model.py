"""Train the coach agent with supervisor-approved review metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from train_simulated_coach_agent_model import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TEXT_FEATURE,
    build_classifier,
    build_metrics,
    evaluate_models,
    feature_columns,
    grouped_split,
    load_training_dataset,
)


@dataclass(frozen=True)
class SupervisorTrainPaths:
    """Input and output paths for supervisor-approved training."""

    phase_root: Path
    rare_path: Path
    policy_path: Path
    supervisor_review_path: Path
    bundle_path: Path
    metrics_path: Path
    predictions_path: Path
    report_path: Path


def main() -> None:
    """Train and save a supervisor-approved coach-agent model bundle."""
    phase_root = Path(__file__).resolve().parent
    paths = SupervisorTrainPaths(
        phase_root=phase_root,
        rare_path=phase_root / "data" / "coach_reaction_dataset_rare_only.csv",
        policy_path=phase_root / "data" / "reward_guided_policy_decisions.csv",
        supervisor_review_path=phase_root / "data" / "coach_reaction_expert_review_supervisor_accepted.csv",
        bundle_path=phase_root / "models" / "supervisor_approved_coach_agent_bundle.joblib",
        metrics_path=phase_root / "data" / "supervisor_approved_coach_agent_training_metrics.json",
        predictions_path=phase_root / "data" / "supervisor_approved_coach_agent_holdout_predictions.csv",
        report_path=phase_root / "outputs" / "supervisor_approved_coach_agent_training_report.md",
    )
    paths.bundle_path.parent.mkdir(parents=True, exist_ok=True)
    paths.report_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_training_dataset(paths)  # type: ignore[arg-type]
    supervisor_summary = load_supervisor_summary(paths.supervisor_review_path)
    train_df, test_df = grouped_split(dataset)
    action_model = build_classifier()
    urgency_model = build_classifier()
    action_model.fit(train_df[feature_columns()], train_df["selected_policy_action"])
    urgency_model.fit(train_df[feature_columns()], train_df["urgency"])

    predictions = evaluate_models(action_model, urgency_model, test_df)
    metrics = build_metrics(test_df, predictions)
    metrics.update(
        {
            "train_label_source": "reward_guided_policy_with_supervisor_accepted_review_checkpoint",
            "supervisor_review_rows": supervisor_summary["review_rows"],
            "supervisor_decision_acceptance_rate": supervisor_summary["decision_acceptance_rate"],
            "supervisor_urgency_correct_rate": supervisor_summary["urgency_correct_rate"],
            "supervisor_no_better_decision_rate": supervisor_summary["no_better_decision_rate"],
            "important_note": (
                "The policy model is trained on the full reward-guided rare-event dataset and supported "
                "by a supervisor-accepted review checkpoint. Report as supervisor-approved simulated "
                "supervision, not independent multi-expert ground truth."
            ),
        }
    )

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
                "Trained on reward-guided rare-event labels with a supervisor-accepted review checkpoint. "
                "Use as the current approved coach-agent bundle for demos and internal validation."
            ),
        },
        paths.bundle_path,
    )
    paths.report_path.write_text(build_report(paths, metrics), encoding="utf-8")
    print("Supervisor-approved coach agent model trained")
    print(f"Bundle: {paths.bundle_path}")
    print(f"Metrics: {paths.metrics_path}")
    print(f"Predictions: {paths.predictions_path}")
    print(f"Report: {paths.report_path}")


def load_supervisor_summary(review_path: Path) -> dict[str, float | int]:
    """Load supervisor acceptance values from the accepted review dataset."""
    review = pd.read_csv(review_path)
    total_rows = int(len(review))
    if total_rows == 0:
        raise RuntimeError("Supervisor review dataset is empty.")
    accepted_rows = int((review["expert_acceptance"] == "accepted").sum())
    urgency_correct_rows = int((review["urgency_correct"] == "yes").sum())
    no_better_rows = int((review["better_decision_available"] == "no").sum())
    return {
        "review_rows": total_rows,
        "decision_acceptance_rate": round(accepted_rows / total_rows, 4),
        "urgency_correct_rate": round(urgency_correct_rows / total_rows, 4),
        "no_better_decision_rate": round(no_better_rows / total_rows, 4),
    }


def build_report(paths: SupervisorTrainPaths, metrics: dict[str, object]) -> str:
    """Build a supervisor-approved model report."""
    return "\n".join(
        [
            "# Supervisor-Approved Coach Agent Training Report",
            "",
            "## Status",
            "",
            "- Trained a runnable coach-agent policy bundle using the full rare-event reward-guided labels.",
            "- Added supervisor acceptance as an explicit review checkpoint.",
            "- The model predicts tactical policy action and urgency from match-window features.",
            "",
            "## Supervisor Review Checkpoint",
            "",
            f"- Reviewed rows: {metrics['supervisor_review_rows']}",
            f"- Decision acceptance rate: {metrics['supervisor_decision_acceptance_rate']}",
            f"- Urgency correct rate: {metrics['supervisor_urgency_correct_rate']}",
            f"- No better decision required rate: {metrics['supervisor_no_better_decision_rate']}",
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
            "## Scientific Note",
            "",
            str(metrics["important_note"]),
        ]
    )


if __name__ == "__main__":
    main()
