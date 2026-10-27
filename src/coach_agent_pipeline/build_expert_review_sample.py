"""Build a balanced expert-review sample for simulated coach reactions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ReviewSampleConfig:
    """Configuration for expert review sampling."""

    random_state: int = 42
    per_group_target: int = 40
    high_shift_target: int = 40


PHASE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PHASE_ROOT / "data"
INPUT_PATH = DATA_DIR / "coach_reaction_dataset_rare_only.csv"
OUTPUT_PATH = DATA_DIR / "coach_reaction_expert_review_sample.csv"
SUMMARY_PATH = DATA_DIR / "coach_reaction_expert_review_sample_summary.csv"


REVIEW_COLUMNS = [
    "review_status",
    "expert_acceptance",
    "expert_reward",
    "expert_correction",
    "expert_notes",
    "reviewer_name",
    "review_date",
]


def main() -> None:
    """Create the balanced expert-review sample."""
    config = ReviewSampleConfig()
    rare = pd.read_csv(INPUT_PATH)
    sample = build_review_sample(rare, config)
    sample.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    build_summary(sample).to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print("Expert review sample created")
    print(f"Sample: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


def build_review_sample(data: pd.DataFrame, config: ReviewSampleConfig) -> pd.DataFrame:
    """Build balanced sample across rare-event groups.

    Args:
        data: Rare-only coach reaction dataset.
        config: Sampling settings.

    Returns:
        Review-ready dataframe with blank expert feedback columns.
    """
    frame = data.copy()
    frame["review_group"] = frame["rare_event_reason"].fillna("").apply(assign_review_group)

    sampled_parts = []
    for group in ["red_card", "goal_state", "formation_change", "substitution", "error_cluster"]:
        group_rows = frame[frame["review_group"] == group].copy()
        if group_rows.empty:
            continue
        count = min(config.per_group_target, len(group_rows))
        sampled_parts.append(
            group_rows.sample(n=count, random_state=config.random_state)
        )

    high_shift = (
        frame.sort_values("tactical_shift_score", ascending=False)
        .head(config.high_shift_target)
        .copy()
    )
    high_shift["review_group"] = "high_tactical_shift"
    sampled_parts.append(high_shift)

    sample = (
        pd.concat(sampled_parts, ignore_index=True)
        .drop_duplicates(subset=["team", "match_id", "time_window", "rare_event_reason"])
        .sort_values(["review_group", "team", "match_id", "time_window"])
        .reset_index(drop=True)
    )

    for column in REVIEW_COLUMNS:
        sample[column] = ""
    sample["review_status"] = "pending"
    sample["expert_acceptance_options"] = "good | acceptable | wrong | dangerous"
    sample["expert_reward_options"] = "1 | 0.5 | 0 | -1"
    sample["review_instruction"] = (
        "Check whether the simulated coach reaction is tactically suitable for this rare-event state. "
        "If not, write the corrected reaction and assign reward."
    )
    return reorder_columns(sample)


def assign_review_group(reason: str) -> str:
    """Map rare-event reason to a review group."""
    lowered = reason.lower()
    if "red card" in lowered or "second yellow" in lowered:
        return "red_card"
    if "unexpected goal" in lowered or "penalty goal" in lowered or "own goal" in lowered:
        return "goal_state"
    if "formation change" in lowered:
        return "formation_change"
    if "substitutions" in lowered:
        return "substitution"
    if "error cluster" in lowered:
        return "error_cluster"
    return "other"


def reorder_columns(sample: pd.DataFrame) -> pd.DataFrame:
    """Put review fields and important context near the front."""
    front = [
        "review_status",
        "review_group",
        "expert_acceptance",
        "expert_reward",
        "expert_correction",
        "expert_notes",
        "team",
        "match_id",
        "time_window",
        "rare_event_reason",
        "tactical_shift_score",
        "coach_reaction",
        "reaction_type",
        "formation_adjustment",
        "pressing_adjustment",
        "defensive_adjustment",
        "substitution_priority",
        "agent_objective",
        "rlhf_preference_prompt",
        "rlhf_reward_seed",
        "real_match_reference",
        "evidence_source_url",
        "evidence_summary",
        "evidence_confidence",
    ]
    remaining = [column for column in sample.columns if column not in front]
    return sample[[column for column in front if column in sample.columns] + remaining]


def build_summary(sample: pd.DataFrame) -> pd.DataFrame:
    """Summarize the review sample."""
    group_counts = sample.groupby("review_group").size().reset_index(name="rows")
    reaction_counts = sample.groupby("reaction_type").size().reset_index(name="rows")
    group_counts.insert(0, "summary_type", "review_group")
    reaction_counts.insert(0, "summary_type", "reaction_type")
    group_counts = group_counts.rename(columns={"review_group": "label"})
    reaction_counts = reaction_counts.rename(columns={"reaction_type": "label"})
    return pd.concat([group_counts, reaction_counts], ignore_index=True)


if __name__ == "__main__":
    main()
