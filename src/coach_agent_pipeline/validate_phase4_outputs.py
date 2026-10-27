"""Validate Phase 4 simulated coach agent outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PHASE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PHASE_ROOT / "data"
OUTPUT_DIR = PHASE_ROOT / "outputs"


def main() -> None:
    """Run compact validation checks."""
    required_paths = [
        DATA_DIR / "coach_reaction_dataset_full.csv",
        DATA_DIR / "coach_reaction_dataset_rare_only.csv",
        DATA_DIR / "coach_reaction_reference_cases.csv",
        DATA_DIR / "coach_reaction_expert_review_sample.csv",
        DATA_DIR / "simulated_coach_agent_decisions.csv",
        DATA_DIR / "coach_reaction_rlhf_preference_seed.csv",
        DATA_DIR / "coach_reaction_expert_review_sample_simulated_filled.csv",
        DATA_DIR / "coach_reaction_rlhf_preference_reviewed_simulated.csv",
        DATA_DIR / "reward_guided_policy_decisions.csv",
        DATA_DIR / "reward_guided_policy_summary.csv",
        DATA_DIR / "simulated_coach_agent_training_metrics.json",
        DATA_DIR / "simulated_coach_agent_holdout_predictions.csv",
        DATA_DIR / "simulated_coach_agent_runtime_smoke_predictions.csv",
        DATA_DIR / "seed_reward_ranker_metrics.json",
        DATA_DIR / "simulated_expert_reward_ranker_metrics.json",
        PHASE_ROOT / "models" / "seed_reward_ranker.joblib",
        PHASE_ROOT / "models" / "simulated_expert_reward_ranker.joblib",
        PHASE_ROOT / "models" / "simulated_coach_agent_bundle.joblib",
        OUTPUT_DIR / "phase4_simulated_coach_agent_review_pack.xlsx",
        OUTPUT_DIR / "seed_reward_ranker_report.md",
        OUTPUT_DIR / "simulated_expert_reward_ranker_report.md",
        OUTPUT_DIR / "reward_guided_policy_selector_report.md",
        OUTPUT_DIR / "simulated_coach_agent_training_report.md",
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size == 0:
            raise RuntimeError(f"Empty output: {path}")

    full = pd.read_csv(DATA_DIR / "coach_reaction_dataset_full.csv")
    rare = pd.read_csv(DATA_DIR / "coach_reaction_dataset_rare_only.csv")
    review = pd.read_csv(DATA_DIR / "coach_reaction_expert_review_sample.csv")
    decisions = pd.read_csv(DATA_DIR / "simulated_coach_agent_decisions.csv")
    preferences = pd.read_csv(DATA_DIR / "coach_reaction_rlhf_preference_seed.csv")
    filled_review = pd.read_csv(DATA_DIR / "coach_reaction_expert_review_sample_simulated_filled.csv")
    reviewed_preferences = pd.read_csv(DATA_DIR / "coach_reaction_rlhf_preference_reviewed_simulated.csv")
    policy = pd.read_csv(DATA_DIR / "reward_guided_policy_decisions.csv")
    holdout = pd.read_csv(DATA_DIR / "simulated_coach_agent_holdout_predictions.csv")
    smoke = pd.read_csv(DATA_DIR / "simulated_coach_agent_runtime_smoke_predictions.csv")
    references = pd.read_csv(DATA_DIR / "coach_reaction_reference_cases.csv")

    assert len(full) == 63603
    assert len(rare) == 12364
    assert len(decisions) == len(rare)
    assert len(preferences) == len(rare)
    assert 150 <= len(review) <= 300
    assert len(filled_review) == len(review)
    assert references["evidence_source_url"].str.len().min() > 10
    assert {"coach_reaction", "reaction_type", "rlhf_reward_seed"}.issubset(full.columns)
    assert {"expert_acceptance", "expert_reward", "expert_correction"}.issubset(review.columns)
    assert {"chosen_action", "rejected_action", "needs_expert_review"}.issubset(preferences.columns)
    assert len(reviewed_preferences) == len(preferences)
    assert len(policy) == len(rare)
    assert len(holdout) > 1000
    assert len(smoke) == 25
    assert decisions["confidence"].between(0, 1).all()
    assert preferences["reward_seed_chosen"].gt(preferences["reward_seed_rejected"]).all()
    assert policy["selected_reward_score"].mean() > policy["rejected_reward_score"].mean()
    assert set(filled_review["review_status"]) == {"simulated_reviewed"}
    assert holdout["policy_action_correct"].mean() >= 0.95
    assert holdout["urgency_correct"].mean() >= 0.95
    assert smoke["selected_policy_action"].str.len().min() > 20
    reward_report = (OUTPUT_DIR / "seed_reward_ranker_report.md").read_text(encoding="utf-8")
    assert "not final RLHF evidence" in reward_report
    simulated_reward_report = (OUTPUT_DIR / "simulated_expert_reward_ranker_report.md").read_text(encoding="utf-8")
    assert "final PhD claims require real expert review" in simulated_reward_report
    agent_report = (OUTPUT_DIR / "simulated_coach_agent_training_report.md").read_text(encoding="utf-8")
    assert "real expert-reviewed labels are still required" in agent_report

    print("Phase 4 validation passed")
    print(f"Full rows: {len(full)}")
    print(f"Rare rows: {len(rare)}")
    print(f"Review sample rows: {len(review)}")
    print(f"Agent decisions: {len(decisions)}")
    print(f"RLHF preference rows: {len(preferences)}")


if __name__ == "__main__":
    main()
