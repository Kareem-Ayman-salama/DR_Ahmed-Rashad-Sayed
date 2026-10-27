"""Build RLHF/RLF preference seed rows from simulated coach decisions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PHASE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PHASE_ROOT / "data"
DECISIONS_PATH = DATA_DIR / "simulated_coach_agent_decisions.csv"
OUTPUT_PATH = DATA_DIR / "coach_reaction_rlhf_preference_seed.csv"
SUMMARY_PATH = DATA_DIR / "coach_reaction_rlhf_preference_seed_summary.csv"


REJECTED_ACTIONS = {
    "numerical_disadvantage_control": "Keep high pressing with wide full-backs despite being a player down.",
    "goal_state_reset": "Immediately overload forward with no rest-defense after conceding/scoring shock.",
    "shape_transition_control": "Change formation and pressing scheme at the same time with no role clarity.",
    "substitution_management": "Make substitutions without assigning a clear tactical role.",
    "error_suppression": "Continue risky central build-up through the same error zone.",
}


def main() -> None:
    """Build preference seed dataset."""
    decisions = pd.read_csv(DECISIONS_PATH)
    preferences = build_preferences(decisions)
    preferences.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    build_summary(preferences).to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print("RLHF preference seed created")
    print(f"Preferences: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


def build_preferences(decisions: pd.DataFrame) -> pd.DataFrame:
    """Convert decisions into chosen/rejected preference rows."""
    rows = []
    for row in decisions.to_dict(orient="records"):
        family = str(row["action_family"])
        rows.append(
            {
                "team": row["team"],
                "match_id": row["match_id"],
                "time_window": row["time_window"],
                "rare_event_reason": row["rare_event_reason"],
                "state": row["rlhf_state"],
                "chosen_action": row["rlhf_action"],
                "rejected_action": REJECTED_ACTIONS.get(
                    family,
                    "Ignore the rare event and keep the same tactical behaviour.",
                ),
                "preference_label": "chosen_better",
                "reward_seed_chosen": row["rlhf_reward_seed"],
                "reward_seed_rejected": -0.25,
                "preference_source": "simulated_evidence_backed_seed",
                "needs_expert_review": True,
                "expert_preference": "",
                "expert_notes": "",
            }
        )
    return pd.DataFrame(rows)


def build_summary(preferences: pd.DataFrame) -> pd.DataFrame:
    """Build compact preference summary."""
    return pd.DataFrame(
        [
            {"metric": "total_preference_rows", "value": len(preferences)},
            {
                "metric": "avg_reward_seed_chosen",
                "value": round(float(preferences["reward_seed_chosen"].mean()), 3),
            },
            {
                "metric": "avg_reward_seed_rejected",
                "value": round(float(preferences["reward_seed_rejected"].mean()), 3),
            },
            {
                "metric": "needs_expert_review_rows",
                "value": int(preferences["needs_expert_review"].sum()),
            },
        ]
    )


if __name__ == "__main__":
    main()
