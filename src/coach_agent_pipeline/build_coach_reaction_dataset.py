"""Build a simulated-coach reaction dataset from rare-event windows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ReactionRule:
    """Coach reaction rule grounded in an evidence case."""

    key: str
    trigger_terms: tuple[str, ...]
    priority: int
    coach_reaction: str
    reaction_type: str
    formation_adjustment: str
    pressing_adjustment: str
    defensive_adjustment: str
    possession_adjustment: str
    substitution_priority: str
    agent_objective: str
    rlhf_preference_prompt: str
    rlhf_reward_seed: float
    real_match_reference: str
    evidence_source_url: str
    evidence_summary: str
    evidence_confidence: str


PHASE_ROOT = Path(__file__).resolve().parent
INPUT_PATH = Path(
    os.getenv(
        "COACH_REACTION_INPUT_PATH",
        str(PHASE_ROOT / "data" / "Cell output 41 teamm.xlsx"),
    )
)
OUTPUT_DIR = PHASE_ROOT / "data"
FULL_OUTPUT_CSV = OUTPUT_DIR / "coach_reaction_dataset_full.csv"
RARE_OUTPUT_CSV = OUTPUT_DIR / "coach_reaction_dataset_rare_only.csv"
REFERENCE_OUTPUT_CSV = OUTPUT_DIR / "coach_reaction_reference_cases.csv"
SUMMARY_OUTPUT_CSV = OUTPUT_DIR / "coach_reaction_dataset_summary.csv"


RULES: tuple[ReactionRule, ...] = (
    ReactionRule(
        key="red_card",
        trigger_terms=("Red Card", "Second Yellow"),
        priority=100,
        coach_reaction=(
            "Immediately protect the result with a compact out-of-possession shape. "
            "Drop into a 4-4-1 or 5-3-1, stop isolated pressing, protect zone 14, "
            "and delay full-back runs until the team has stable possession."
        ),
        reaction_type="numerical_disadvantage_control",
        formation_adjustment="Compact 4-4-1 or 5-3-1 depending on available defenders.",
        pressing_adjustment="Reduce high press; press only on backward pass or poor first touch.",
        defensive_adjustment="Narrow midfield line, protect central lane, force play wide.",
        possession_adjustment="Use safer outlet passes and longer possession rest phases.",
        substitution_priority="Sacrifice one attacker/winger for defensive balance if needed.",
        agent_objective="Survive the next 5-10 minutes and prevent central penetration.",
        rlhf_preference_prompt=(
            "For a red-card state, prefer the action that preserves compactness and "
            "reduces central risk over one that keeps aggressive high pressing."
        ),
        rlhf_reward_seed=1.0,
        real_match_reference="Southampton vs Manchester United, Premier League, 2021; Arsenal vs Barcelona, UCL Final, 2006.",
        evidence_source_url=(
            "https://learning.coachesvoice.com/manchester-united-9-southampton-0-tactical-analysis/ | "
            "https://en.wikipedia.org/wiki/2006_UEFA_Champions_League_final"
        ),
        evidence_summary=(
            "Coaches' Voice reports Southampton's early red card forced a 4-4-1 and "
            "abandoned expansive plans. The 2006 UCL final records Arsenal substituting "
            "Pires for Almunia after Lehmann's red card."
        ),
        evidence_confidence="high",
    ),
    ReactionRule(
        key="injury",
        trigger_terms=("Injury",),
        priority=95,
        coach_reaction=(
            "Stabilize the match state after the injury. Reassign the injured player's "
            "zone, reduce tempo for several possessions, and choose the replacement "
            "based on structural balance before attacking ambition."
        ),
        reaction_type="injury_restructure",
        formation_adjustment="Keep the same base shape if possible; avoid two simultaneous structural changes.",
        pressing_adjustment="Temporarily reduce pressing triggers until distances are restored.",
        defensive_adjustment="Cover the injured player's channel with nearest midfielder/full-back.",
        possession_adjustment="Slow tempo and use secure circulation for the next phase.",
        substitution_priority="Like-for-like replacement unless chasing the game late.",
        agent_objective="Absorb disruption and restore spacing within one tactical window.",
        rlhf_preference_prompt=(
            "For an injury state, prefer the reaction that restores team spacing and role clarity "
            "before increasing attacking risk."
        ),
        rlhf_reward_seed=0.8,
        real_match_reference="Portugal vs France, UEFA Euro 2016 Final.",
        evidence_source_url="https://www.theguardian.com/football/live/2016/jul/10/portugal-v-france-euro-2016-final-live?page=with%3Ablock-5782a692e4b0eebd31586dc7",
        evidence_summary=(
            "Portugal lost Cristiano Ronaldo to injury in the first half, remained compact, "
            "and later won through substitute Eder."
        ),
        evidence_confidence="medium",
    ),
    ReactionRule(
        key="unexpected_goal",
        trigger_terms=("Unexpected Goal", "Penalty Goal", "Own Goal"),
        priority=90,
        coach_reaction=(
            "Treat the goal as a match-state shock. Reset emotionally and tactically, "
            "protect against the next transition, keep rest-defense numbers behind the ball, "
            "and only increase attacking risk after the team regains spacing."
        ),
        reaction_type="goal_state_reset",
        formation_adjustment="Keep base shape for 3-5 minutes; avoid chaotic immediate reshuffle.",
        pressing_adjustment="Counter-press only with support; avoid single-player pressure.",
        defensive_adjustment="Protect central counter lanes and second balls.",
        possession_adjustment="Use controlled possessions to reset rhythm after the goal.",
        substitution_priority="If chasing late, prepare direct attacking substitute; otherwise stabilize first.",
        agent_objective="Prevent a second concession in the next window and regain control.",
        rlhf_preference_prompt=(
            "After a goal shock, prefer a response that prevents immediate repeat danger "
            "before selecting a high-risk attacking change."
        ),
        rlhf_reward_seed=0.9,
        real_match_reference="Manchester United vs Bayern Munich, UCL Final, 1999.",
        evidence_source_url="https://www.uefa.com/uefachampionsleague/news/025e-0f8bbd7e5380-eb00803acfbb-1000--champions-league-classics-manchester-united-2-1-bayern/",
        evidence_summary=(
            "UEFA describes Ferguson using attacking substitutes Sheringham and Solskjaer "
            "before United's late comeback, showing match-state response through substitutions."
        ),
        evidence_confidence="medium",
    ),
    ReactionRule(
        key="formation_change",
        trigger_terms=("Formation Change",),
        priority=80,
        coach_reaction=(
            "Make the shape change explicit and conservative for the next phase. "
            "Communicate new pressing lanes, mark the weak-side cover rule, and avoid "
            "risky central passes while players adapt."
        ),
        reaction_type="shape_transition_control",
        formation_adjustment="Confirm the new base shape and individual lane responsibilities.",
        pressing_adjustment="Use simpler pressing triggers until the new shape is stable.",
        defensive_adjustment="Protect weak side and half-spaces during the transition.",
        possession_adjustment="Circulate through safe side before switching play.",
        substitution_priority="Use substitutions only if the current squad cannot fill the new role map.",
        agent_objective="Avoid conceding during the tactical transition window.",
        rlhf_preference_prompt=(
            "For a formation-change state, prefer clear role assignment and stable transition "
            "over complex multi-step tactical instructions."
        ),
        rlhf_reward_seed=0.75,
        real_match_reference="Argentina tactical switch to 3-5-2 against Netherlands, FIFA World Cup 2022.",
        evidence_source_url="https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/lionel-scaloni-argentina-tactics-world-cup",
        evidence_summary=(
            "FIFA notes Scaloni switched to a 3-5-2 to mirror the Dutch system, "
            "an example of a coach changing shape for the match context."
        ),
        evidence_confidence="medium",
    ),
    ReactionRule(
        key="substitution_wave",
        trigger_terms=("Substitutions",),
        priority=70,
        coach_reaction=(
            "Use the substitution wave to change one tactical theme, not everything at once. "
            "Refresh the highest-load role, define the new pressing trigger, and give the "
            "team one simple attacking route for the next window."
        ),
        reaction_type="substitution_management",
        formation_adjustment="Keep base structure unless the substitution is explicitly tactical.",
        pressing_adjustment="Refresh pressure around the ball with one clear trigger.",
        defensive_adjustment="Maintain rest-defense coverage while new players settle.",
        possession_adjustment="Use the substitute's strongest channel immediately but safely.",
        substitution_priority="Prioritize role clarity, physical freshness, and late-game specialty.",
        agent_objective="Convert substitutions into controlled tactical gain.",
        rlhf_preference_prompt=(
            "For a substitution cluster, prefer one coherent tactical objective over several "
            "competing instructions."
        ),
        rlhf_reward_seed=0.7,
        real_match_reference="England Women vs Germany, UEFA Women's Euro 2022 Final; Manchester United vs Bayern Munich, UCL Final, 1999.",
        evidence_source_url=(
            "https://www.skysports.com/football/news/11095/12663256/england-women-win-euro-2022-sarina-wiegmans-use-of-super-subs-could-be-game-changing-after-lionesses-triumph | "
            "https://www.uefa.com/uefachampionsleague/news/025e-0f8bbd7e5380-eb00803acfbb-1000--champions-league-classics-manchester-united-2-1-bayern/"
        ),
        evidence_summary=(
            "Sky Sports highlights Wiegman's effective use of substitutes in Euro 2022. "
            "UEFA records United's substitutes scoring decisive late goals in 1999."
        ),
        evidence_confidence="medium",
    ),
    ReactionRule(
        key="error_cluster",
        trigger_terms=("Error Cluster",),
        priority=60,
        coach_reaction=(
            "Reduce technical risk immediately. Stop risky central build-up, create a short "
            "support triangle near the ball, use the goalkeeper or pivot as a reset option, "
            "and force the next attack away from the error zone."
        ),
        reaction_type="error_suppression",
        formation_adjustment="No full formation change; add local support around the error zone.",
        pressing_adjustment="Do not press after a loose structure; recover shape first.",
        defensive_adjustment="Close central lane and protect the space behind the losing player.",
        possession_adjustment="Use short secure passes or clear to safe wide channel.",
        substitution_priority="Replace only if errors repeat or fatigue is visible.",
        agent_objective="Stop repeated turnovers from becoming a rare-event cascade.",
        rlhf_preference_prompt=(
            "For repeated errors, prefer a low-risk possession reset and local support over "
            "forcing forward play."
        ),
        rlhf_reward_seed=0.65,
        real_match_reference="Inter vs Barcelona tactical transition reference, UEFA Champions League 2010.",
        evidence_source_url="https://franciscomaldonado.substack.com/p/inter-milan-3-1-barcelona-2010-analysis",
        evidence_summary=(
            "The analysis describes Mourinho prioritising defensive compactness and "
            "direct transition routes, useful as evidence for avoiding risky central exposure."
        ),
        evidence_confidence="low",
    ),
)


NORMAL_REACTION = {
    "coach_reaction": (
        "No immediate structural intervention. Continue monitoring risk, keep the current "
        "shape stable, and prepare contingency instructions if the next window escalates."
    ),
    "reaction_type": "monitor",
    "formation_adjustment": "No change.",
    "pressing_adjustment": "Maintain current pressing rules.",
    "defensive_adjustment": "Maintain current defensive balance.",
    "possession_adjustment": "Maintain current possession plan.",
    "substitution_priority": "No substitution trigger.",
    "agent_objective": "Monitor without overreacting.",
    "rlhf_preference_prompt": "For normal states, prefer no unnecessary intervention.",
    "rlhf_reward_seed": 0.0,
    "real_match_reference": "Not applicable.",
    "evidence_source_url": "",
    "evidence_summary": "Normal windows are not mapped to real rare-event reactions.",
    "evidence_confidence": "not_applicable",
    "reaction_generation_method": "default_monitoring_rule",
}


def main() -> None:
    """Build enriched CSV outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = clean_text_frame(pd.read_excel(INPUT_PATH))
    enriched = add_reactions(df)
    rare_only = enriched[enriched["rare_event"].astype(bool)].copy()
    reference = pd.DataFrame([rule.__dict__ for rule in RULES])
    summary = build_summary(enriched)

    enriched.to_csv(FULL_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    rare_only.to_csv(RARE_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    reference.to_csv(REFERENCE_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("Coach reaction datasets created")
    print(f"Full: {FULL_OUTPUT_CSV}")
    print(f"Rare only: {RARE_OUTPUT_CSV}")
    print(f"Reference: {REFERENCE_OUTPUT_CSV}")
    print(f"Summary: {SUMMARY_OUTPUT_CSV}")


def clean_text_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Repair common mojibake text artifacts in object columns."""
    cleaned = df.copy()
    for column in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[column] = cleaned[column].map(clean_text_value)
    return cleaned


def clean_text_value(value: object) -> object:
    """Repair text that was decoded as Latin-1 instead of UTF-8."""
    if not isinstance(value, str):
        return value
    if "Ã" not in value and "Â" not in value:
        return value
    try:
        return value.encode("latin1").decode("utf-8")
    except UnicodeError:
        return value


def add_reactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add simulated coach reaction columns to the dataset.

    Args:
        df: Original tactical rare-event window dataset.

    Returns:
        Enriched dataset with coach-agent reaction and evidence columns.
    """
    enriched = df.copy()
    reaction_rows = [select_reaction(row) for row in enriched.to_dict(orient="records")]
    reaction_df = pd.DataFrame(reaction_rows)
    return pd.concat([enriched, reaction_df], axis=1)


def select_reaction(row: dict[str, Any]) -> dict[str, Any]:
    """Select the highest-priority reaction matching a row's rare-event reason."""
    if not bool(row.get("rare_event", False)):
        return NORMAL_REACTION.copy()

    reason = str(row.get("rare_event_reason", "") or "")
    matching_rules = [
        rule
        for rule in RULES
        if any(term.lower() in reason.lower() for term in rule.trigger_terms)
    ]
    if not matching_rules:
        reaction = NORMAL_REACTION.copy()
        reaction["reaction_type"] = "rare_event_unmapped"
        reaction["coach_reaction"] = (
            "Rare event detected, but no specific rule matched. Stabilize team shape, "
            "protect central spaces, and request expert review."
        )
        reaction["rlhf_reward_seed"] = 0.25
        reaction["reaction_generation_method"] = "fallback_rare_event_rule"
        return reaction

    rule = sorted(matching_rules, key=lambda item: item.priority, reverse=True)[0]
    reaction = {
        key: value
        for key, value in rule.__dict__.items()
        if key not in {"trigger_terms", "priority"}
    }
    reaction["reaction_generation_method"] = "evidence_backed_rule_mapping"
    if len(matching_rules) > 1:
        reaction["coach_reaction"] = (
            reaction["coach_reaction"]
            + " Additional combined-event note: handle the secondary event only after the primary risk is stable."
        )
    return reaction


def build_summary(enriched: pd.DataFrame) -> pd.DataFrame:
    """Create a compact summary table."""
    rows = [
        {"metric": "total_rows", "value": int(len(enriched))},
        {
            "metric": "rare_event_rows",
            "value": int(enriched["rare_event"].astype(bool).sum()),
        },
        {
            "metric": "normal_rows",
            "value": int((~enriched["rare_event"].astype(bool)).sum()),
        },
        {
            "metric": "unique_teams",
            "value": int(enriched["team"].nunique(dropna=True)),
        },
        {
            "metric": "unique_matches",
            "value": int(enriched["match_id"].nunique(dropna=True)),
        },
    ]
    reaction_counts = (
        enriched.groupby("reaction_type", dropna=False)
        .size()
        .reset_index(name="value")
        .rename(columns={"reaction_type": "metric"})
    )
    reaction_counts["metric"] = "reaction_type_" + reaction_counts["metric"].astype(str)
    return pd.concat([pd.DataFrame(rows), reaction_counts], ignore_index=True)


if __name__ == "__main__":
    main()
