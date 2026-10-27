"""Build a concise client-facing final summary report."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


PHASE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PHASE_ROOT / "data"
OUTPUT_DIR = PHASE_ROOT / "outputs"
VIDEO_DIR = OUTPUT_DIR / "video_demo"

METRICS_PATH = DATA_DIR / "supervisor_approved_coach_agent_training_metrics.json"
AUDIT_PATH = DATA_DIR / "coach_agent_overfitting_audit_metrics.json"
VIDEO_STREAM_PATH = DATA_DIR / "supervisor_approved_video_decision_stream.csv"
REPORT_MD = OUTPUT_DIR / "client_final_summary_report.md"
REPORT_PDF = OUTPUT_DIR / "client_final_summary_report.pdf"


def main() -> None:
    """Build client summary report in Markdown and PDF."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_json(METRICS_PATH)
    audit = load_json(AUDIT_PATH)
    video_stream = pd.read_csv(VIDEO_STREAM_PATH)
    REPORT_MD.write_text(build_markdown(metrics, audit, video_stream), encoding="utf-8")
    build_pdf(metrics, audit, video_stream, REPORT_PDF)
    print("Client final summary report created")
    print(f"Markdown: {REPORT_MD}")
    print(f"PDF: {REPORT_PDF}")


def load_json(path: Path) -> dict[str, object]:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_markdown(
    metrics: dict[str, object],
    audit: dict[str, object],
    video_stream: pd.DataFrame,
) -> str:
    """Build Markdown report."""
    video_path = VIDEO_DIR / "supervisor_approved_agent_side_panel_demo.mp4"
    return f"""
# Tactical Football AI - Client Final Summary

## Current Status

The project now has a clean supervisor-approved coach-agent stage built on top of the final rare-event dataset.

- Full match-window dataset: 63,603 rows.
- Rare-event dataset: 12,364 rows.
- Supervisor-reviewed sample: {metrics["supervisor_review_rows"]} rows.
- Supervisor acceptance rate: {metrics["supervisor_decision_acceptance_rate"]}.
- Final demo video: `{video_path}`.
- Video decision stream: `{VIDEO_STREAM_PATH}`.

## Trained Agent Outputs

The current agent returns:

- Tactical decision/action.
- Urgency level.
- Policy confidence.
- Explanation.
- Model source.

## Model Metrics

- Holdout rows: {metrics["holdout_rows"]}.
- Holdout matches: {metrics["holdout_matches"]}.
- Policy action accuracy: {metrics["policy_action_accuracy"]}.
- Policy action macro-F1: {metrics["policy_action_macro_f1"]}.
- Urgency accuracy: {metrics["urgency_accuracy"]}.
- Urgency macro-F1: {metrics["urgency_macro_f1"]}.

## Overfitting / Shortcut Audit

There is no evidence that the numbers were manipulated. However, the audit shows high label-shortcut risk.

- Full-feature policy accuracy: {audit["full_feature_policy_action_accuracy"]}.
- Text/rare-event-only policy accuracy: {audit["text_reason_only_policy_action_accuracy"]}.
- Numeric-only policy accuracy: {audit["numeric_only_policy_action_accuracy"]}.
- Shortcut risk level: {audit["shortcut_risk_level"]}.

Safe scientific wording:

> The trained agent accurately reproduces a supervisor-approved, reward-guided tactical decision policy over rare-event match windows.

Avoid claiming that the model independently predicts real coach decisions with perfect accuracy until larger independent expert validation is added.

## Final Demo Windows

{video_stream[["window_id", "start_time_seconds", "end_time_seconds", "rare_event_reason", "urgency", "selected_policy_action"]].to_csv(index=False)}

## Next Step Before Paper Writing

1. Send this delivery pack and final demo to the client/supervisor.
2. Confirm that the demo style and scientific wording are accepted.
3. Add larger expert validation or external validation if requested.
4. Start paper writing after this first phase is accepted.
""".strip() + "\n"


def build_pdf(
    metrics: dict[str, object],
    audit: dict[str, object],
    video_stream: pd.DataFrame,
    output_path: Path,
) -> None:
    """Build a compact PDF report."""
    with PdfPages(output_path) as pdf:
        add_text_page(
            pdf,
            "Tactical Football AI - Final Client Summary",
            [
                "Current stage: supervisor-approved coach-agent demo and delivery pack.",
                "Full dataset: 63,603 match-window rows.",
                "Rare-event dataset: 12,364 rows.",
                f"Supervisor-reviewed sample: {metrics['supervisor_review_rows']} rows.",
                f"Supervisor acceptance rate: {metrics['supervisor_decision_acceptance_rate']}.",
                "Final demo video: supervisor_approved_agent_side_panel_demo.mp4.",
                "The agent outputs tactical action, urgency, confidence, and explanation.",
            ],
        )
        add_text_page(
            pdf,
            "Model Results and Audit",
            [
                f"Policy action accuracy: {metrics['policy_action_accuracy']}.",
                f"Policy action macro-F1: {metrics['policy_action_macro_f1']}.",
                f"Urgency accuracy: {metrics['urgency_accuracy']}.",
                f"Urgency macro-F1: {metrics['urgency_macro_f1']}.",
                "",
                "Audit conclusion: no evidence of manipulated results, but label-shortcut risk is high.",
                f"Text/rare-event-only policy accuracy: {audit['text_reason_only_policy_action_accuracy']}.",
                f"Numeric-only policy accuracy: {audit['numeric_only_policy_action_accuracy']}.",
                "",
                "Safe claim: the agent reproduces a supervisor-approved, reward-guided tactical decision policy.",
            ],
        )
        add_table_page(pdf, video_stream)


def add_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    """Add a text-only page to the PDF."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    plt.axis("off")
    plt.text(0.08, 0.93, title, fontsize=18, weight="bold", va="top")
    y = 0.86
    for line in lines:
        if not line:
            y -= 0.035
            continue
        wrapped = textwrap.wrap(line, width=88)
        for segment in wrapped:
            plt.text(0.08, y, segment, fontsize=11, va="top")
            y -= 0.032
        y -= 0.012
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf: PdfPages, video_stream: pd.DataFrame) -> None:
    """Add the final video decision stream table to the PDF."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    plt.axis("off")
    plt.text(0.04, 0.95, "Final Demo Decision Windows", fontsize=16, weight="bold", va="top")
    display = video_stream[
        ["window_id", "start_time_seconds", "end_time_seconds", "rare_event_reason", "urgency"]
    ].copy()
    table = plt.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
