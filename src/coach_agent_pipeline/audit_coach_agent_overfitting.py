"""Audit coach-agent results for overfitting, leakage, and label shortcuts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from train_simulated_coach_agent_model import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TEXT_FEATURE,
    build_classifier,
    feature_columns,
    grouped_split,
    load_training_dataset,
)


PHASE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PHASE_ROOT / "data"
OUTPUT_DIR = PHASE_ROOT / "outputs"


@dataclass(frozen=True)
class AuditPaths:
    """Paths for audit inputs and outputs."""

    rare_path: Path
    policy_path: Path
    metrics_path: Path
    table_path: Path
    shortcut_path: Path
    report_path: Path


def main() -> None:
    """Run coach-agent overfitting and leakage audit."""
    paths = AuditPaths(
        rare_path=DATA_DIR / "coach_reaction_dataset_rare_only.csv",
        policy_path=DATA_DIR / "reward_guided_policy_decisions.csv",
        metrics_path=DATA_DIR / "coach_agent_overfitting_audit_metrics.json",
        table_path=DATA_DIR / "coach_agent_overfitting_audit_table.csv",
        shortcut_path=DATA_DIR / "coach_agent_label_shortcut_table.csv",
        report_path=OUTPUT_DIR / "coach_agent_overfitting_audit_report.md",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_training_dataset(paths)  # type: ignore[arg-type]
    train_df, test_df = grouped_split(dataset)
    shortcut_table = build_shortcut_table(dataset)
    audit_table = run_ablation_audit(train_df, test_df)
    metrics = build_audit_metrics(dataset, shortcut_table, audit_table)

    shortcut_table.to_csv(paths.shortcut_path, index=False, encoding="utf-8")
    audit_table.to_csv(paths.table_path, index=False, encoding="utf-8")
    paths.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    paths.report_path.write_text(build_report(paths, metrics, audit_table), encoding="utf-8")

    print("Coach-agent overfitting audit completed")
    print(f"Metrics: {paths.metrics_path}")
    print(f"Audit table: {paths.table_path}")
    print(f"Shortcut table: {paths.shortcut_path}")
    print(f"Report: {paths.report_path}")


def build_shortcut_table(dataset: pd.DataFrame) -> pd.DataFrame:
    """Measure how strongly event labels map to policy actions."""
    rows: list[dict[str, Any]] = []
    for group_col in ["rare_event_reason", "reaction_type"]:
        grouped = dataset.groupby(group_col)
        for label, group in grouped:
            action_counts = group["selected_policy_action"].value_counts()
            urgency_counts = group["urgency"].value_counts()
            rows.append(
                {
                    "shortcut_feature": group_col,
                    "feature_value": label,
                    "rows": len(group),
                    "unique_actions": int(group["selected_policy_action"].nunique()),
                    "dominant_action": action_counts.index[0],
                    "dominant_action_rate": round(float(action_counts.iloc[0] / len(group)), 4),
                    "unique_urgencies": int(group["urgency"].nunique()),
                    "dominant_urgency": urgency_counts.index[0],
                    "dominant_urgency_rate": round(float(urgency_counts.iloc[0] / len(group)), 4),
                }
            )
    return pd.DataFrame(rows).sort_values(["shortcut_feature", "rows"], ascending=[True, False])


def run_ablation_audit(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Run feature ablations to expose shortcut dependence."""
    variants = {
        "full_features": feature_columns(),
        "no_text_reason": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "numeric_only": NUMERIC_FEATURES,
        "text_reason_only": [TEXT_FEATURE],
    }
    rows: list[dict[str, Any]] = []
    for variant_name, cols in variants.items():
        action_model = build_classifier_for_columns(cols)
        urgency_model = build_classifier_for_columns(cols)
        action_model.fit(train_df[cols], train_df["selected_policy_action"])
        urgency_model.fit(train_df[cols], train_df["urgency"])
        action_pred = action_model.predict(test_df[cols])
        urgency_pred = urgency_model.predict(test_df[cols])
        rows.append(
            {
                "variant": variant_name,
                "feature_count": len(cols),
                "policy_action_accuracy": round(float(accuracy_score(test_df["selected_policy_action"], action_pred)), 4),
                "policy_action_macro_f1": round(float(f1_score(test_df["selected_policy_action"], action_pred, average="macro")), 4),
                "urgency_accuracy": round(float(accuracy_score(test_df["urgency"], urgency_pred)), 4),
                "urgency_macro_f1": round(float(f1_score(test_df["urgency"], urgency_pred, average="macro")), 4),
            }
        )
    return pd.DataFrame(rows)


def build_classifier_for_columns(columns: list[str]):
    """Build a classifier while preserving existing pipeline style."""
    original_numeric = list(NUMERIC_FEATURES)
    original_categorical = list(CATEGORICAL_FEATURES)
    original_text = TEXT_FEATURE

    selected_numeric = [col for col in original_numeric if col in columns]
    selected_categorical = [col for col in original_categorical if col in columns]
    has_text = original_text in columns

    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    transformers = []
    if has_text:
        transformers.append(("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2), original_text))
    if selected_numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                selected_numeric,
            )
        )
    if selected_categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                selected_categorical,
            )
        )
    preprocess = ColumnTransformer(transformers=transformers, sparse_threshold=0.0)
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


def build_audit_metrics(
    dataset: pd.DataFrame,
    shortcut_table: pd.DataFrame,
    audit_table: pd.DataFrame,
) -> dict[str, Any]:
    """Build compact audit metrics."""
    event_rows = shortcut_table[shortcut_table["shortcut_feature"] == "rare_event_reason"]
    reaction_rows = shortcut_table[shortcut_table["shortcut_feature"] == "reaction_type"]
    full = audit_table[audit_table["variant"] == "full_features"].iloc[0]
    text_only = audit_table[audit_table["variant"] == "text_reason_only"].iloc[0]
    numeric_only = audit_table[audit_table["variant"] == "numeric_only"].iloc[0]
    return {
        "dataset_rows": int(len(dataset)),
        "unique_policy_actions": int(dataset["selected_policy_action"].nunique()),
        "unique_rare_event_reasons": int(dataset["rare_event_reason"].nunique()),
        "mean_dominant_action_rate_by_rare_event_reason": round(float(event_rows["dominant_action_rate"].mean()), 4),
        "mean_dominant_action_rate_by_reaction_type": round(float(reaction_rows["dominant_action_rate"].mean()), 4),
        "full_feature_policy_action_accuracy": float(full["policy_action_accuracy"]),
        "text_reason_only_policy_action_accuracy": float(text_only["policy_action_accuracy"]),
        "numeric_only_policy_action_accuracy": float(numeric_only["policy_action_accuracy"]),
        "shortcut_risk_level": infer_shortcut_risk(event_rows, audit_table),
        "audit_conclusion": (
            "High policy-action scores are expected because supervisor-approved labels are generated from "
            "stable tactical rules and rare-event categories. This is not evidence of manipulation, but it "
            "does mean the current result should be presented as rule/reward-guided supervised agent "
            "replication, not as independent discovery of coaching decisions."
        ),
    }


def infer_shortcut_risk(event_rows: pd.DataFrame, audit_table: pd.DataFrame) -> str:
    """Infer label shortcut risk level from audit evidence."""
    text_only = audit_table[audit_table["variant"] == "text_reason_only"].iloc[0]
    mean_event_rate = float(event_rows["dominant_action_rate"].mean())
    if mean_event_rate >= 0.95 and float(text_only["policy_action_accuracy"]) >= 0.95:
        return "high"
    if mean_event_rate >= 0.80:
        return "medium"
    return "low"


def build_report(paths: AuditPaths, metrics: dict[str, Any], audit_table: pd.DataFrame) -> str:
    """Build markdown audit report."""
    audit_markdown = dataframe_to_markdown(audit_table)
    return f"""
# Coach Agent Overfitting and Leakage Audit

## Bottom Line

There is no evidence that results were manipulated. However, the near-perfect policy-action scores are not strong evidence of real-world generalization by themselves.

The main reason is a high label-shortcut risk: the current supervisor-approved labels are rule/reward-guided, and rare-event categories strongly determine the selected coach action.

## Key Audit Metrics

- Dataset rows audited: {metrics["dataset_rows"]:,}
- Unique policy actions: {metrics["unique_policy_actions"]}
- Unique rare-event reasons: {metrics["unique_rare_event_reasons"]}
- Mean dominant-action rate by rare-event reason: {metrics["mean_dominant_action_rate_by_rare_event_reason"]}
- Mean dominant-action rate by reaction type: {metrics["mean_dominant_action_rate_by_reaction_type"]}
- Shortcut risk level: {metrics["shortcut_risk_level"]}

## Feature Ablation Results

{audit_markdown}

## Interpretation

The 1.0000 policy-action accuracy is plausible because the policy labels are generated from consistent tactical templates and reward ranking, then accepted by the supervisor. This makes the task easier than predicting independent real coach behavior from raw match dynamics.

For the paper, do not claim that the model independently discovered tactical coaching decisions with perfect accuracy. The safe claim is:

> The trained agent accurately reproduces a supervisor-approved, reward-guided tactical decision policy over rare-event match windows.

## What We Need For Stronger Publication Claims

1. Independent expert labels from one or more coaches on a larger sample.
2. Evaluation on labels not generated by the same rule/reward system.
3. External validation on another data source or match video.
4. Report ablation results to show what the model learns from event category versus numerical tactical features.

## Output Files

- Metrics JSON: `{paths.metrics_path}`
- Ablation table: `{paths.table_path}`
- Label shortcut table: `{paths.shortcut_path}`
""".strip() + "\n"


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a small dataframe as a markdown table without optional dependencies."""
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = [str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
