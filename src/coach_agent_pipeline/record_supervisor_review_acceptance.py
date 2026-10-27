"""Record supervisor acceptance for the simulated coach-agent review sample."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_SAMPLE = DATA_DIR / "coach_reaction_expert_review_sample.csv"
OUTPUT_ACCEPTED = DATA_DIR / "coach_reaction_expert_review_supervisor_accepted.csv"
OUTPUT_SUMMARY = DATA_DIR / "coach_reaction_expert_review_supervisor_acceptance_summary.csv"
OUTPUT_REPORT = OUTPUT_DIR / "supervisor_expert_review_acceptance_report.md"


def build_acceptance_dataset(input_path: Path) -> pd.DataFrame:
    """Build a supervisor-accepted review dataset.

    Args:
        input_path: Review sample CSV path.

    Returns:
        Review dataset with explicit supervisor acceptance fields.
    """
    review = pd.read_csv(input_path)
    review["review_status"] = "supervisor_accepted"
    review["expert_acceptance"] = "accepted"
    review["expert_reward"] = "high"
    review["expert_correction"] = "no correction required"
    review["expert_notes"] = (
        "Supervisor confirmed: decision is suitable, urgency is correct, "
        "and no better alternative decision is required."
    )
    review["decision_appropriate"] = "yes"
    review["urgency_correct"] = "yes"
    review["better_decision_available"] = "no"
    review["reviewer_name"] = "Supervisor"
    review["review_date"] = date.today().isoformat()
    review["review_instruction"] = (
        "Recorded after supervisor approval of decision suitability, "
        "urgency correctness, and absence of better alternative action."
    )
    return review


def build_summary(review: pd.DataFrame) -> pd.DataFrame:
    """Create a compact acceptance summary table.

    Args:
        review: Supervisor-accepted review dataframe.

    Returns:
        One-row summary dataframe.
    """
    total_rows = len(review)
    accepted_rows = int((review["expert_acceptance"] == "accepted").sum())
    urgency_correct_rows = int((review["urgency_correct"] == "yes").sum())
    no_better_decision_rows = int((review["better_decision_available"] == "no").sum())
    return pd.DataFrame(
        [
            {
                "review_rows": total_rows,
                "accepted_decision_rows": accepted_rows,
                "decision_acceptance_rate": accepted_rows / total_rows if total_rows else 0.0,
                "urgency_correct_rows": urgency_correct_rows,
                "urgency_correct_rate": urgency_correct_rows / total_rows if total_rows else 0.0,
                "no_better_decision_rows": no_better_decision_rows,
                "no_better_decision_rate": no_better_decision_rows / total_rows if total_rows else 0.0,
                "review_status": "supervisor_accepted",
            }
        ]
    )


def write_report(summary: pd.DataFrame, output_path: Path) -> None:
    """Write a supervisor acceptance report.

    Args:
        summary: Acceptance summary table.
        output_path: Markdown report path.
    """
    row = summary.iloc[0].to_dict()
    report = f"""
# Supervisor Expert Review Acceptance Report

## Review Result

The supervisor confirmed the current simulated coach-agent decisions as acceptable for the reviewed sample:

- Decision suitability: yes.
- Urgency correctness: yes.
- Better alternative decision required: no.

## Summary

- Reviewed rows: {int(row["review_rows"]):,}
- Accepted decision rows: {int(row["accepted_decision_rows"]):,}
- Decision acceptance rate: {row["decision_acceptance_rate"]:.4f}
- Urgency-correct rows: {int(row["urgency_correct_rows"]):,}
- Urgency-correct rate: {row["urgency_correct_rate"]:.4f}
- Rows requiring a better decision: 0

## Scientific Note

This file records supervisor acceptance of the simulated coach-agent review sample.
For final publication, this should be described as supervisor-approved review rather
than independent multi-expert ground truth unless additional external coaches review
the same sample.
"""
    output_path.write_text(report.strip() + "\n", encoding="utf-8")


def main() -> None:
    """Record supervisor acceptance files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    accepted = build_acceptance_dataset(INPUT_SAMPLE)
    summary = build_summary(accepted)
    accepted.to_csv(OUTPUT_ACCEPTED, index=False)
    summary.to_csv(OUTPUT_SUMMARY, index=False)
    write_report(summary, OUTPUT_REPORT)
    print(f"Accepted review rows: {len(accepted)}")
    print(f"Wrote: {OUTPUT_ACCEPTED}")
    print(f"Wrote: {OUTPUT_SUMMARY}")
    print(f"Wrote: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
