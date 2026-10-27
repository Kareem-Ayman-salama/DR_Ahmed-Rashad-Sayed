"""Runtime API for the simulated coach agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PHASE_ROOT = Path(__file__).resolve().parent
SUPERVISOR_BUNDLE_PATH = PHASE_ROOT / "models" / "supervisor_approved_coach_agent_bundle.joblib"
SIMULATED_BUNDLE_PATH = PHASE_ROOT / "models" / "simulated_coach_agent_bundle.joblib"
DEFAULT_BUNDLE_PATH = SUPERVISOR_BUNDLE_PATH if SUPERVISOR_BUNDLE_PATH.exists() else SIMULATED_BUNDLE_PATH


@dataclass(frozen=True)
class AgentRecommendation:
    """Coach-agent recommendation returned at inference time."""

    selected_policy_action: str
    urgency: str
    policy_confidence: float
    urgency_confidence: float
    explanation: str
    source: str


class SimulatedCoachAgent:
    """Loadable simulated coach agent."""

    def __init__(self, bundle_path: Path = DEFAULT_BUNDLE_PATH) -> None:
        """Load model bundle.

        Args:
            bundle_path: Saved model bundle path.
        """
        self.bundle_path = bundle_path
        self.bundle: dict[str, Any] = joblib.load(bundle_path)
        self.action_model = self.bundle["action_model"]
        self.urgency_model = self.bundle["urgency_model"]
        self.feature_columns: list[str] = list(self.bundle["feature_columns"])

    def recommend(self, state: dict[str, Any]) -> AgentRecommendation:
        """Recommend a coach action for one match state.

        Args:
            state: Dict containing rare-event window features.

        Returns:
            Structured coach recommendation.
        """
        frame = self._make_frame(state)
        action_proba = self.action_model.predict_proba(frame[self.feature_columns])[0]
        urgency_proba = self.urgency_model.predict_proba(frame[self.feature_columns])[0]
        action_index = int(np.argmax(action_proba))
        urgency_index = int(np.argmax(urgency_proba))
        action = str(self.action_model.classes_[action_index])
        urgency = str(self.urgency_model.classes_[urgency_index])
        policy_confidence = round(float(action_proba[action_index]), 4)
        urgency_confidence = round(float(urgency_proba[urgency_index]), 4)
        return AgentRecommendation(
            selected_policy_action=action,
            urgency=urgency,
            policy_confidence=policy_confidence,
            urgency_confidence=urgency_confidence,
            explanation=self._explain(state, action, urgency),
            source=self.bundle_path.stem,
        )

    def recommend_many(self, states: pd.DataFrame) -> pd.DataFrame:
        """Recommend actions for many rows."""
        rows = []
        for row in states.to_dict(orient="records"):
            recommendation = self.recommend(row)
            output = {
                "team": row.get("team", ""),
                "match_id": row.get("match_id", ""),
                "time_window": row.get("time_window", ""),
                "rare_event_reason": row.get("rare_event_reason", ""),
                **recommendation.__dict__,
            }
            rows.append(output)
        return pd.DataFrame(rows)

    def _make_frame(self, state: dict[str, Any]) -> pd.DataFrame:
        """Build a single-row feature frame with expected columns."""
        row = {}
        for column in self.feature_columns:
            if column == "rare_event_reason":
                row[column] = str(state.get(column, "Unknown") or "Unknown")
            elif column.startswith("is_") or column.startswith("has_"):
                row[column] = str(state.get(column, False))
            else:
                row[column] = state.get(column, 0)
        return pd.DataFrame([row])

    @staticmethod
    def _explain(state: dict[str, Any], action: str, urgency: str) -> str:
        """Create a concise recommendation explanation."""
        reason = state.get("rare_event_reason", "rare event")
        shift = float(state.get("tactical_shift_score", 0.0) or 0.0)
        pass_success = float(state.get("pass_success_rate", 0.0) or 0.0)
        return (
            f"{urgency.title()} recommendation for {reason}. "
            f"Tactical shift={shift:.2f}, pass success={pass_success:.2f}. "
            f"Selected action: {action}"
        )


def main() -> None:
    """Run a small smoke inference from the rare-event dataset."""
    data_path = PHASE_ROOT / "data" / "coach_reaction_dataset_rare_only.csv"
    output_path = PHASE_ROOT / "data" / "supervisor_approved_coach_agent_runtime_smoke_predictions.csv"
    data = pd.read_csv(data_path).head(25)
    agent = SimulatedCoachAgent()
    predictions = agent.recommend_many(data)
    predictions.to_csv(output_path, index=False, encoding="utf-8")
    print("Runtime smoke predictions created")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
