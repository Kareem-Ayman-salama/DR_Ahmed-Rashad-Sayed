"""Select coach actions using the simulated expert reward ranker."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


PHASE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PHASE_ROOT / "data"
OUTPUT_DIR = PHASE_ROOT / "outputs"
MODEL_PATH = PHASE_ROOT / "models" / "simulated_expert_reward_ranker.joblib"
INPUT_PATH = DATA_DIR / "simulated_coach_agent_decisions.csv"
OUTPUT_PATH = DATA_DIR / "reward_guided_policy_decisions.csv"
SUMMARY_PATH = DATA_DIR / "reward_guided_policy_summary.csv"
REPORT_PATH = OUTPUT_DIR / "reward_guided_policy_selector_report.md"


ALTERNATIVE_ACTIONS = {
    "numerical_disadvantage_control": [
        "Switch to compact 4-4-1 or 5-3-1, reduce high press, and protect central zones.",
        "Keep the same attacking shape and try to dominate possession despite the red card.",
        "Make an immediate attacking substitution and press higher to regain momentum.",
    ],
    "goal_state_reset": [
        "Reset shape, protect transition lanes, and rebuild possession before chasing.",
        "Send more players forward immediately and attack through both full-backs.",
        "Ignore the goal state and continue the previous tactical plan without adjustment.",
    ],
    "shape_transition_control": [
        "Clarify new formation roles, simplify pressing triggers, and protect weak side.",
        "Change formation and pressing triggers at the same time without role confirmation.",
        "Keep players improvising until the next stoppage.",
    ],
    "substitution_management": [
        "Use substitutions to change one clear tactical theme and define each role.",
        "Make substitutions based only on fatigue without tactical instructions.",
        "Delay all substitutions even if the window shows tactical instability.",
    ],
    "error_suppression": [
        "Reduce central build-up risk, add local support, and move play away from the error zone.",
        "Continue risky central progression through the same error zone.",
        "Increase pressing immediately after turnovers without restoring team shape.",
    ],
}


def main() -> None:
    """Run reward-guided action selection."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    decisions = pd.read_csv(INPUT_PATH)
    ranker = joblib.load(MODEL_PATH)
    selected = select_actions(decisions, ranker)
    selected.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    build_summary(selected).to_csv(SUMMARY_PATH, index=False, encoding="utf-8")
    REPORT_PATH.write_text(build_report(selected), encoding="utf-8")
    print("Reward-guided policy selection complete")
    print(f"Decisions: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Report: {REPORT_PATH}")


def select_actions(decisions: pd.DataFrame, ranker: object) -> pd.DataFrame:
    """Rank candidate actions and select the best action."""
    rows = []
    for row in decisions.to_dict(orient="records"):
        family = str(row["action_family"])
        candidates = ALTERNATIVE_ACTIONS.get(family, [str(row["selected_action"])])
        scored = []
        for action in candidates:
            text = str(row["rlhf_state"]) + " ACTION: " + action
            reward_score = float(ranker.predict_proba([text])[0][1])
            scored.append((action, reward_score))
        scored = sorted(scored, key=lambda item: item[1], reverse=True)
        rows.append(
            {
                "team": row["team"],
                "match_id": row["match_id"],
                "time_window": row["time_window"],
                "rare_event_reason": row["rare_event_reason"],
                "action_family": family,
                "urgency": row["urgency"],
                "agent_confidence": row["confidence"],
                "selected_policy_action": scored[0][0],
                "selected_reward_score": round(scored[0][1], 4),
                "second_best_action": scored[1][0] if len(scored) > 1 else "",
                "second_best_reward_score": round(scored[1][1], 4) if len(scored) > 1 else "",
                "rejected_action": scored[-1][0],
                "rejected_reward_score": round(scored[-1][1], 4),
                "policy_selection_source": "simulated_expert_reward_ranker",
            }
        )
    return pd.DataFrame(rows)


def build_summary(selected: pd.DataFrame) -> pd.DataFrame:
    """Build policy summary."""
    family = selected.groupby("action_family").size().reset_index(name="rows")
    family.insert(0, "summary_type", "action_family")
    family = family.rename(columns={"action_family": "label"})
    urgency = selected.groupby("urgency").size().reset_index(name="rows")
    urgency.insert(0, "summary_type", "urgency")
    urgency = urgency.rename(columns={"urgency": "label"})
    metrics = pd.DataFrame(
        [
            {"summary_type": "metric", "label": "total_policy_decisions", "rows": len(selected)},
            {
                "summary_type": "metric",
                "label": "avg_selected_reward_score",
                "rows": round(float(selected["selected_reward_score"].mean()), 4),
            },
            {
                "summary_type": "metric",
                "label": "avg_rejected_reward_score",
                "rows": round(float(selected["rejected_reward_score"].mean()), 4),
            },
        ]
    )
    return pd.concat([metrics, family, urgency], ignore_index=True)


def build_report(selected: pd.DataFrame) -> str:
    """Build policy selector report."""
    return "\n".join(
        [
            "# Reward-Guided Policy Selector Report",
            "",
            "## Status",
            "",
            "- Generated multiple candidate coach actions for each rare-event state.",
            "- Used the simulated expert reward ranker to select the highest-scoring action.",
            "- Exported policy decisions for all rare-event windows.",
            "",
            "## Summary",
            "",
            f"- Policy decisions: {len(selected)}",
            f"- Average selected reward score: {selected['selected_reward_score'].mean():.4f}",
            f"- Average rejected reward score: {selected['rejected_reward_score'].mean():.4f}",
            "",
            "## Important Limitation",
            "",
            "The selector is ready as an engineering pipeline, but the reward model still uses simulated expert labels. "
            "Real expert review is required before publication claims.",
        ]
    )


if __name__ == "__main__":
    main()
