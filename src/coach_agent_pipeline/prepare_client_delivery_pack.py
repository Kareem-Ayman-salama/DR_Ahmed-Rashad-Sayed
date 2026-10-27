"""Prepare a clean client delivery pack for the tactical football AI project.

The pack collects the final notebook/data, Phase 3 video/demo artifacts, Phase 4
simulated coach-agent datasets, trained models, reports, and paper-ready figures.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
PHASE3_DIR = ROOT_DIR / "phase_3_cv_tracking_bridge"
PHASE4_DIR = ROOT_DIR / "phase_4_simulated_coach_agent"
DELIVERY_DIR = ROOT_DIR / "client_delivery_supervisor_approved_agent_20260813"


@dataclass(frozen=True)
class DeliveryFile:
    """File to copy into the delivery pack."""

    source: Path
    target_subdir: str
    purpose: str


def ensure_directories(base_dir: Path) -> None:
    """Create the delivery folder layout.

    Args:
        base_dir: Root directory of the delivery pack.
    """
    folders = [
        "00_read_me",
        "01_source_final_work",
        "02_processed_datasets",
        "03_trained_models",
        "04_reproducible_code",
        "05_reports_and_review",
        "06_paper_figures",
        "07_video_demo",
        "08_paper_assets",
    ]
    for folder in folders:
        (base_dir / folder).mkdir(parents=True, exist_ok=True)


def copy_if_exists(item: DeliveryFile, base_dir: Path) -> Path | None:
    """Copy one file if it exists.

    Args:
        item: Source, destination, and purpose metadata.
        base_dir: Root directory of the delivery pack.

    Returns:
        Destination path when copied, otherwise None.
    """
    if not item.source.exists():
        return None
    destination = base_dir / item.target_subdir / item.source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(item.source, destination)
    return destination


def build_file_plan() -> list[DeliveryFile]:
    """Build the list of files to include in the delivery pack."""
    source_files = [
        DeliveryFile(ROOT_DIR / "Cell output 41 teamm.xlsx", "01_source_final_work", "Original final extracted dataset workbook"),
        DeliveryFile(ROOT_DIR / "final.ipynb", "01_source_final_work", "Final notebook used for the data paper phase"),
        DeliveryFile(ROOT_DIR / "AI_AGENT_HANDOFF_FINAL.md", "01_source_final_work", "Hand-off note describing the final notebook/data stage"),
        DeliveryFile(ROOT_DIR / "relation_model_alerts_FINAL.csv", "01_source_final_work", "Earlier relation-model alert output from the final work"),
        DeliveryFile(ROOT_DIR / "relation_model_top_alerts_FINAL.csv", "01_source_final_work", "Earlier top tactical alert output from the final work"),
        DeliveryFile(ROOT_DIR / "relation_model_report_FINAL.csv", "01_source_final_work", "Earlier relation-model KPI report"),
        DeliveryFile(ROOT_DIR / "relation_model_tier_kpi_FINAL.csv", "01_source_final_work", "Earlier relation-model tier KPI report"),
        DeliveryFile(ROOT_DIR / "relation_model_config_FINAL.json", "01_source_final_work", "Earlier relation-model configuration"),
    ]

    phase4_data = [
        "coach_reaction_dataset_full.csv",
        "coach_reaction_dataset_rare_only.csv",
        "coach_reaction_dataset_summary.csv",
        "coach_reaction_reference_cases.csv",
        "coach_reaction_expert_review_sample.csv",
        "coach_reaction_expert_review_sample_simulated_filled.csv",
        "coach_reaction_expert_review_supervisor_accepted.csv",
        "coach_reaction_expert_review_supervisor_acceptance_summary.csv",
        "coach_reaction_rlhf_preference_seed.csv",
        "coach_reaction_rlhf_preference_reviewed_simulated.csv",
        "reward_guided_policy_decisions.csv",
        "reward_guided_policy_summary.csv",
        "simulated_coach_agent_decisions.csv",
        "simulated_coach_agent_decision_summary.csv",
        "simulated_coach_agent_holdout_predictions.csv",
        "simulated_coach_agent_runtime_smoke_predictions.csv",
        "simulated_coach_agent_training_metrics.json",
        "supervisor_approved_coach_agent_holdout_predictions.csv",
        "supervisor_approved_coach_agent_runtime_smoke_predictions.csv",
        "supervisor_approved_coach_agent_training_metrics.json",
        "supervisor_approved_video_decision_stream.csv",
        "seed_reward_ranker_metrics.json",
        "simulated_expert_reward_ranker_metrics.json",
        "coach_agent_overfitting_audit_metrics.json",
        "coach_agent_overfitting_audit_table.csv",
        "coach_agent_label_shortcut_table.csv",
    ]
    data_files = [
        DeliveryFile(PHASE4_DIR / "data" / filename, "02_processed_datasets", f"Phase 4 processed dataset/output: {filename}")
        for filename in phase4_data
    ]

    model_files = [
        DeliveryFile(PHASE4_DIR / "models" / "simulated_coach_agent_bundle.joblib", "03_trained_models", "Trained simulated coach-agent bundle"),
        DeliveryFile(PHASE4_DIR / "models" / "supervisor_approved_coach_agent_bundle.joblib", "03_trained_models", "Current supervisor-approved coach-agent bundle"),
        DeliveryFile(PHASE4_DIR / "models" / "simulated_expert_reward_ranker.joblib", "03_trained_models", "Trained reward ranker from simulated expert review"),
        DeliveryFile(PHASE4_DIR / "models" / "seed_reward_ranker.joblib", "03_trained_models", "Seed reward-ranker baseline"),
    ]

    code_files = [
        "build_coach_reaction_dataset.py",
        "build_expert_review_sample.py",
        "build_rlhf_preference_seed.py",
        "simulate_expert_review_and_reward.py",
        "train_reward_ranker_seed.py",
        "run_reward_guided_policy_selector.py",
        "train_simulated_coach_agent_model.py",
        "train_supervisor_approved_coach_agent_model.py",
        "coach_agent_runtime.py",
        "record_supervisor_review_acceptance.py",
        "audit_coach_agent_overfitting.py",
        "build_supervisor_approved_video_demo.py",
        "build_client_final_summary_report.py",
        "validate_phase4_outputs.py",
    ]
    code = [
        DeliveryFile(PHASE4_DIR / filename, "04_reproducible_code", f"Reproducible Phase 4 code: {filename}")
        for filename in code_files
    ]
    code.append(DeliveryFile(Path(__file__).resolve(), "04_reproducible_code", "Delivery-pack preparation script"))

    reports = [
        DeliveryFile(PHASE4_DIR / "README.md", "05_reports_and_review", "Phase 4 README"),
        DeliveryFile(PHASE4_DIR / "PHASE4_FINAL_STATUS.md", "05_reports_and_review", "Final Phase 4 status"),
        DeliveryFile(PHASE4_DIR / "outputs" / "phase4_simulated_coach_agent_review_pack.xlsx", "05_reports_and_review", "Review workbook for supervisor/client inspection"),
        DeliveryFile(PHASE4_DIR / "outputs" / "simulated_coach_agent_training_report.md", "05_reports_and_review", "Agent training report"),
        DeliveryFile(PHASE4_DIR / "outputs" / "supervisor_approved_coach_agent_training_report.md", "05_reports_and_review", "Supervisor-approved coach-agent training report"),
        DeliveryFile(PHASE4_DIR / "outputs" / "coach_agent_overfitting_audit_report.md", "05_reports_and_review", "Overfitting and label-shortcut audit report"),
        DeliveryFile(PHASE4_DIR / "outputs" / "client_final_summary_report.md", "05_reports_and_review", "Client final summary report in Markdown"),
        DeliveryFile(PHASE4_DIR / "outputs" / "client_final_summary_report.pdf", "05_reports_and_review", "Client final summary report in PDF"),
        DeliveryFile(PHASE4_DIR / "outputs" / "reward_guided_policy_selector_report.md", "05_reports_and_review", "Reward-guided policy selector report"),
        DeliveryFile(PHASE4_DIR / "outputs" / "supervisor_expert_review_acceptance_report.md", "05_reports_and_review", "Supervisor expert-review acceptance report"),
        DeliveryFile(PHASE4_DIR / "outputs" / "seed_reward_ranker_report.md", "05_reports_and_review", "Seed reward ranker report"),
        DeliveryFile(PHASE4_DIR / "outputs" / "simulated_expert_reward_ranker_report.md", "05_reports_and_review", "Simulated expert reward-ranker report"),
        DeliveryFile(PHASE3_DIR / "client_delivery" / "CLIENT_STATUS_SUMMARY.md", "05_reports_and_review", "Client video/status summary from Phase 3"),
        DeliveryFile(PHASE3_DIR / "MASTER_PUBLICATION_TODO.md", "05_reports_and_review", "Master publication todo from earlier phases"),
    ]

    phase3_figures = [
        DeliveryFile(PHASE3_DIR / "reports" / "figures" / "phase3_demo_risk_timeline.png", "06_paper_figures", "Phase 3 video risk timeline"),
        DeliveryFile(PHASE3_DIR / "reports" / "figures" / "phase3_demo_top_features.png", "06_paper_figures", "Phase 3 top video features"),
        DeliveryFile(PHASE3_DIR / "reports" / "figures" / "phase3_demo_top_moment_pitch.png", "06_paper_figures", "Phase 3 top moment pitch visualization"),
        DeliveryFile(PHASE4_DIR / "outputs" / "phase4_output_map_preview.png", "06_paper_figures", "Phase 4 output-map preview"),
        DeliveryFile(PHASE4_DIR / "outputs" / "simulated_coach_summary_preview.png", "06_paper_figures", "Phase 4 summary preview"),
    ]

    videos = [
        DeliveryFile(PHASE3_DIR / "client_delivery" / "client_demo_tactical_match_ai.mp4", "07_video_demo", "Client-ready tactical match AI demo video"),
        DeliveryFile(PHASE4_DIR / "outputs" / "video_demo" / "supervisor_approved_agent_side_panel_demo.mp4", "07_video_demo", "Final supervisor-approved coach-agent side-panel demo"),
        DeliveryFile(PHASE3_DIR / "reports" / "video_demo" / "real_bayern_final_roboflow_decision_demo.mp4", "07_video_demo", "Real broadcast decision demo"),
        DeliveryFile(PHASE3_DIR / "reports" / "video_demo" / "real_bayern_tracking_quality_side_panel_demo.mp4", "07_video_demo", "Side-panel tracking quality demo"),
        DeliveryFile(PHASE3_DIR / "reports" / "video_demo" / "referee_visibility_demo.mp4", "07_video_demo", "Referee visibility demo"),
    ]

    return source_files + data_files + model_files + code + reports + phase3_figures + videos


def save_bar_chart(
    counts: pd.Series,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str = "Count",
    color: str = "#2563eb",
) -> None:
    """Save a horizontal bar chart from a count series."""
    clean_counts = counts.dropna().sort_values(ascending=True)
    fig_height = max(4.5, min(9.0, 0.45 * len(clean_counts) + 1.5))
    plt.figure(figsize=(10, fig_height))
    clean_counts.plot(kind="barh", color=color)
    plt.title(title, fontsize=14, weight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_reward_comparison(decisions: pd.DataFrame, output_path: Path) -> None:
    """Save selected/rejected reward score comparison chart."""
    score_frame = pd.DataFrame(
        {
            "Selected policy": decisions["selected_reward_score"],
            "Second best": decisions["second_best_reward_score"],
            "Rejected policy": decisions["rejected_reward_score"],
        }
    )
    means = score_frame.mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(means.index, means.values, color=["#16a34a", "#f59e0b", "#dc2626"])
    plt.title("Reward-Guided Policy Ranking", fontsize=14, weight="bold")
    plt.ylabel("Mean reward score")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_metrics_chart(metrics_path: Path, output_path: Path) -> None:
    """Save a compact chart of holdout metrics."""
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    values = {
        "Policy action accuracy": metrics.get("policy_action_accuracy"),
        "Policy action macro-F1": metrics.get("policy_action_macro_f1"),
        "Urgency accuracy": metrics.get("urgency_accuracy"),
        "Urgency macro-F1": metrics.get("urgency_macro_f1"),
    }
    series = pd.Series(values).dropna()
    plt.figure(figsize=(9, 5))
    bars = plt.bar(series.index, series.values, color=["#0f766e", "#0ea5e9", "#7c3aed", "#db2777"])
    plt.ylim(0, 1.05)
    plt.title("Simulated Coach Agent Holdout Metrics", fontsize=14, weight="bold")
    plt.ylabel("Score")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.015, f"{height:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_supervisor_acceptance_chart(summary_path: Path, output_path: Path) -> None:
    """Save supervisor acceptance checkpoint chart."""
    summary = pd.read_csv(summary_path).iloc[0]
    values = pd.Series(
        {
            "Decision suitable": summary["decision_acceptance_rate"],
            "Urgency correct": summary["urgency_correct_rate"],
            "No better action needed": summary["no_better_decision_rate"],
        }
    )
    plt.figure(figsize=(8, 5))
    bars = plt.bar(values.index, values.values, color=["#16a34a", "#0ea5e9", "#7c3aed"])
    plt.ylim(0, 1.05)
    plt.title("Supervisor Review Acceptance Checkpoint", fontsize=14, weight="bold")
    plt.ylabel("Acceptance rate")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.015, f"{height:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_ablation_audit_chart(audit_path: Path, output_path: Path) -> None:
    """Save ablation audit chart."""
    audit = pd.read_csv(audit_path)
    x_labels = audit["variant"].tolist()
    x_positions = list(range(len(x_labels)))
    plt.figure(figsize=(10, 5.5))
    plt.plot(x_positions, audit["policy_action_accuracy"], marker="o", linewidth=2.0, label="Policy action accuracy")
    plt.plot(x_positions, audit["urgency_accuracy"], marker="o", linewidth=2.0, label="Urgency accuracy")
    plt.ylim(0, 1.05)
    plt.title("Overfitting Audit: Feature Ablation", fontsize=14, weight="bold")
    plt.ylabel("Accuracy")
    plt.xticks(x_positions, x_labels, rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_pipeline_diagram(output_path: Path) -> None:
    """Save a paper-ready pipeline overview diagram."""
    stages = [
        ("Final data", "63.6K match-window rows\\nrare-event features"),
        ("Coach labels", "Supervisor-approved\\nsimulated reactions"),
        ("RLHF seed", "preferred vs rejected\\naction pairs"),
        ("Reward model", "rank tactical\\nresponses"),
        ("Policy agent", "select action, urgency,\\nconfidence, explanation"),
        ("CV/live demo", "match video stream\\n+ side-panel decisions"),
    ]
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis("off")
    box_width = 0.145
    y = 0.55
    for index, (title, body) in enumerate(stages):
        x = 0.03 + index * 0.16
        rect = plt.Rectangle((x, y), box_width, 0.28, facecolor="#f8fafc", edgecolor="#334155", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + box_width / 2, y + 0.2, title, ha="center", va="center", fontsize=11, weight="bold")
        ax.text(x + box_width / 2, y + 0.09, body, ha="center", va="center", fontsize=9)
        if index < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + box_width + 0.02, y + 0.14),
                xytext=(x + box_width + 0.005, y + 0.14),
                arrowprops={"arrowstyle": "->", "color": "#334155", "lw": 1.5},
            )
    ax.text(0.5, 0.18, "Tactical Football Preparation AI: Data -> Coach Decision Agent -> Video Decision Assistant", ha="center", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def generate_figures(base_dir: Path) -> list[Path]:
    """Generate paper/presentation figures from Phase 4 outputs.

    Args:
        base_dir: Root directory of the delivery pack.

    Returns:
        List of generated figure paths.
    """
    output_dir = base_dir / "06_paper_figures"
    rare = pd.read_csv(PHASE4_DIR / "data" / "coach_reaction_dataset_rare_only.csv")
    agent = pd.read_csv(PHASE4_DIR / "data" / "simulated_coach_agent_decisions.csv")
    reward = pd.read_csv(PHASE4_DIR / "data" / "reward_guided_policy_decisions.csv")

    figures = [
        output_dir / "fig_01_rare_event_distribution.png",
        output_dir / "fig_02_coach_reaction_type_distribution.png",
        output_dir / "fig_03_agent_urgency_distribution.png",
        output_dir / "fig_04_reward_guided_policy_ranking.png",
        output_dir / "fig_05_holdout_metrics.png",
        output_dir / "fig_06_project_pipeline.png",
        output_dir / "fig_07_supervisor_review_acceptance.png",
        output_dir / "fig_08_overfitting_ablation_audit.png",
    ]

    save_bar_chart(
        rare["rare_event_reason"].value_counts().head(12),
        figures[0],
        "Rare Event Distribution",
        "Windows",
        "Rare event type",
        "#2563eb",
    )
    save_bar_chart(
        rare["reaction_type"].value_counts(),
        figures[1],
        "Coach Reaction Type Distribution",
        "Windows",
        "Reaction type",
        "#0f766e",
    )
    save_bar_chart(
        agent["urgency"].value_counts(),
        figures[2],
        "Agent Urgency Distribution",
        "Decisions",
        "Urgency",
        "#7c3aed",
    )
    save_reward_comparison(reward, figures[3])
    supervisor_metrics = PHASE4_DIR / "data" / "supervisor_approved_coach_agent_training_metrics.json"
    fallback_metrics = PHASE4_DIR / "data" / "simulated_coach_agent_training_metrics.json"
    save_metrics_chart(supervisor_metrics if supervisor_metrics.exists() else fallback_metrics, figures[4])
    save_pipeline_diagram(figures[5])
    save_supervisor_acceptance_chart(
        PHASE4_DIR / "data" / "coach_reaction_expert_review_supervisor_acceptance_summary.csv",
        figures[6],
    )
    save_ablation_audit_chart(PHASE4_DIR / "data" / "coach_agent_overfitting_audit_table.csv", figures[7])
    return figures


def write_markdown(path: Path, content: str) -> None:
    """Write a markdown file with UTF-8 encoding."""
    path.write_text(content.strip() + "\n", encoding="utf-8")


def write_client_readme(base_dir: Path) -> None:
    """Write client-facing README and run instructions."""
    readme = """
# Tactical Football AI - Client Delivery Pack

## What is included

This folder contains the current supervised delivery state of the project:

- Final source work from the earlier data-paper stage.
- Processed rare-event coach-reaction datasets.
- Supervisor-approved simulated coach-agent labels.
- RLHF-style preference data and reward-guided policy decisions.
- Trained coach-agent and reward-ranker model bundles.
- Reproducible Phase 4 Python code.
- Review workbook, reports, demo video files, and paper/presentation figures.

## Current validated result

- Full dataset rows: 63,603.
- Rare-event rows: 12,364.
- Supervisor-reviewed sample rows: 239.
- Supervisor acceptance rate on reviewed sample: 1.0000.
- Agent decision rows: 12,364.
- Holdout policy-action accuracy: 1.0000.
- Holdout policy-action macro-F1: 1.0000.
- Holdout urgency accuracy: 0.9948.
- Holdout urgency macro-F1: 0.9883.

Important wording: these metrics are for the supervisor-approved simulated coach-agent labels. The reviewed sample was accepted by the supervisor for decision suitability, urgency correctness, and no required better alternative. The overfitting audit shows high label-shortcut risk because rare-event categories strongly determine the selected policy action. Final paper claims should therefore separate supervisor-approved policy replication from independent multi-expert real-world generalization.

## Main folders

- `01_source_final_work`: original final data/notebook assets.
- `02_processed_datasets`: clean processed datasets and model outputs.
- `03_trained_models`: trained `.joblib` model bundles.
- `04_reproducible_code`: scripts needed to rebuild and validate the agent stage.
- `05_reports_and_review`: review workbook and technical reports.
- `06_paper_figures`: generated figures for paper/presentation use.
- `07_video_demo`: available match-video decision demos.

## Recommended next step

Use this pack for client/supervisor review, then move to real expert validation:

1. Ask the supervisor/coach to review a stratified sample of decisions.
2. Replace simulated filled labels with reviewed labels where available.
3. Retrain the reward ranker and coach-agent policy.
4. Evaluate on a held-out match/video segment.
5. Only after that, write the final paper results section.
"""

    instructions = """
# Run Instructions

## Validate Phase 4 outputs

From the main project folder:

```powershell
cd "D:\\Freelance work\\Rashad\\phase_4_simulated_coach_agent"
python validate_phase4_outputs.py
```

## Rebuild the simulated coach-agent model

```powershell
cd "D:\\Freelance work\\Rashad\\phase_4_simulated_coach_agent"
python train_simulated_coach_agent_model.py
```

## Run a runtime smoke test

```powershell
cd "D:\\Freelance work\\Rashad\\phase_4_simulated_coach_agent"
python coach_agent_runtime.py
```

## Regenerate this delivery pack

```powershell
cd "D:\\Freelance work\\Rashad"
python prepare_client_delivery_pack.py
```
"""

    captions = """
# Paper Figure Index

| Figure | File | Suggested caption |
|---|---|---|
| Fig. 1 | `fig_01_rare_event_distribution.png` | Distribution of rare tactical/disciplinary events used to trigger coach-agent decisions. |
| Fig. 2 | `fig_02_coach_reaction_type_distribution.png` | Distribution of simulated coach reaction types across rare-event windows. |
| Fig. 3 | `fig_03_agent_urgency_distribution.png` | Urgency profile assigned by the simulated coach agent for rare-event decisions. |
| Fig. 4 | `fig_04_reward_guided_policy_ranking.png` | Mean reward comparison between selected, second-best, and rejected tactical policies. |
| Fig. 5 | `fig_05_holdout_metrics.png` | Holdout evaluation metrics for the trained simulated coach-agent model. |
| Fig. 6 | `fig_06_project_pipeline.png` | End-to-end project pipeline from final match-window data to coach-agent decisions and video assistant. |
| Fig. 7 | `fig_07_supervisor_review_acceptance.png` | Supervisor acceptance checkpoint for decision suitability, urgency correctness, and no better required alternative. |
| Fig. 8 | `fig_08_overfitting_ablation_audit.png` | Feature ablation audit showing how much policy-action prediction depends on rare-event shortcuts versus numeric tactical features. |
"""

    arabic_summary = """
# ملخص جاهز للإرسال للعميل / الدكتور

وصلنا حالياً لنسخة منظمة من المشروع فيها:

- الداتا النهائية الخارجة من مرحلة `final` مع ملفات التقارير القديمة.
- داتا rare events بعدد 12,364 حالة من أصل 63,603 window.
- إضافة simulated coach reaction لكل rare event بناءً على قواعد تكتيكية ومراجع حالات حقيقية، وتم اعتبارها supervisor-approved بعد مراجعة الدكتور.
- تسجيل موافقة الدكتور على عينة مراجعة عددها 239 صف: القرار مناسب، الـ urgency صحيح، ولا يوجد قرار أفضل مطلوب.
- إضافة audit للـ overfitting والـ label shortcut: الأرقام ليست ملعوب فيها، لكن القرار مرتبط بقوة بنوع rare event، وده لازم يتذكر كـ limitation علمي.
- بناء RLHF-style preference data لاختيار القرار الأفضل مقابل قرار مرفوض.
- تدريب reward ranker ثم simulated coach agent يخرج:
  - القرار التكتيكي المقترح.
  - درجة الاستعجال.
  - الثقة.
  - السبب/الشرح.
  - القرار الذي يجب تجنبه.
- تجهيز موديلات مدربة وملفات تشغيل وتحقق.
- تجهيز فيديوهات demo وجرافات paper/presentation جاهزة للعرض.
- تجهيز فيديو نهائي باسم `supervisor_approved_agent_side_panel_demo.mp4` مربوط بقرارات الموديل المعتمد.
- تجهيز تقرير عميل نهائي PDF وMarkdown قبل مرحلة كتابة البيبر.

مهم في العرض العلمي: النتائج الحالية مبنية على supervisor-approved simulated labels، وليست ground-truth كاملة من مدربين حقيقيين لكل صف. لذلك المرحلة القادمة الأفضل هي مراجعة عينة أكبر من القرارات بواسطة خبير/مدرب، ثم إعادة التدريب والتقييم قبل كتابة الورقة النهائية.
"""

    todo = """
# Next Phase TODO

## P1 - Client/Supervisor Review

- عرض delivery pack الحالي على العميل/الدكتور.
- مراجعة `phase4_simulated_coach_agent_review_pack.xlsx`.
- تأكيد صياغة القرارات التكتيكية وهل هي مناسبة كـ simulated coach.

## P2 - Expert Validation

- اختيار عينة stratified من rare events.
- ملء قرار المدرب الحقيقي أو تعديل القرار المقترح.
- تسجيل هل القرار صحيح/مقبول/غير مقبول.
- استخدام المراجعة كـ expert ground truth.

## P3 - Retraining With Reviewed Labels

- إعادة تدريب reward ranker باستخدام المراجعات الحقيقية.
- إعادة تدريب coach-agent policy.
- مقارنة النتائج قبل وبعد expert review.

## P4 - Video + Agent Integration

- ربط فيديو المباراة بقرارات الـ agent على side panel.
- عرض high-risk windows والقرارات بالإنجليزي.
- حفظ decision stream كـ CSV مع الفيديو.

## P5 - External Validation

- تجربة الداتا/الموديل على مصدر خارجي مثل SkillCorner أو SoccerNet.
- توثيق الفرق بين Metrica/internal data والمصدر الخارجي.

## P6 - Paper Writing

- كتابة الورقة بعد اكتمال validation فقط.
- استخدام figures الموجودة في `06_paper_figures`.
- فصل نتائج simulated-supervision عن نتائج expert-ground-truth بوضوح.
"""

    write_markdown(base_dir / "00_read_me" / "CLIENT_HANDOFF_README.md", readme)
    write_markdown(base_dir / "00_read_me" / "CLIENT_SUMMARY_AR.md", arabic_summary)
    write_markdown(base_dir / "00_read_me" / "NEXT_PHASE_TODO.md", todo)
    write_markdown(base_dir / "00_read_me" / "RUN_INSTRUCTIONS.md", instructions)
    write_markdown(base_dir / "08_paper_assets" / "PAPER_FIGURE_INDEX.md", captions)


def write_manifest(base_dir: Path, copied_files: Iterable[Path], generated_figures: Iterable[Path]) -> None:
    """Write a CSV manifest for the delivery pack."""
    rows = []
    all_files = list(copied_files) + list(generated_figures)
    for path in sorted(set(all_files)):
        if not path.exists():
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(base_dir)),
                "size_bytes": path.stat().st_size,
                "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )

    manifest_path = base_dir / "00_read_me" / "DELIVERY_MANIFEST.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "last_modified"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Create the complete client delivery pack."""
    ensure_directories(DELIVERY_DIR)
    copied_files: list[Path] = []
    missing_files: list[Path] = []

    for item in build_file_plan():
        copied = copy_if_exists(item, DELIVERY_DIR)
        if copied is None:
            missing_files.append(item.source)
        else:
            copied_files.append(copied)

    generated_figures = generate_figures(DELIVERY_DIR)
    write_client_readme(DELIVERY_DIR)
    write_manifest(DELIVERY_DIR, copied_files, generated_figures)

    missing_report = "\n".join(f"- {path}" for path in missing_files) or "- None"
    write_markdown(
        DELIVERY_DIR / "00_read_me" / "MISSING_OPTIONAL_FILES.md",
        f"""
# Missing Optional Files

The delivery pack was created successfully. These optional files were not found and therefore were not copied:

{missing_report}
""",
    )

    print(f"Delivery pack created: {DELIVERY_DIR}")
    print(f"Copied files: {len(copied_files)}")
    print(f"Generated figures: {len(generated_figures)}")
    print(f"Missing optional files: {len(missing_files)}")


if __name__ == "__main__":
    main()
