"""Create reproducible analyses requested in the R1 editorial assessment."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
from rare_event_revision_pipeline import (  # noqa: E402
    PipelineConfig,
    best_f1_threshold,
    metric_bundle,
    prepare_design_matrices,
    split_by_match,
)


COMPONENTS: dict[str, tuple[str, int]] = {
    "Red Card": ("n_red_cards", 1),
    "Second Yellow": ("n_second_yellows", 1),
    "Substitution Burst": ("n_subs", 2),
    "Unexpected Goal": ("n_unexpected_goals", 1),
    "Own Goal": ("n_own_goals", 1),
    "Formation Change": ("n_formation_changes", 1),
    "Penalty Goal": ("n_penalty_goals", 1),
}

ANTICIPATION_FEATURES = [
    "event_count", "pass_count", "avg_pass_length", "pass_success_rate",
    "forward_rate", "long_pass_rate", "delta_pass_count",
    "delta_avg_pass_length", "delta_pass_success_rate", "delta_forward_rate",
    "delta_long_pass_rate", "tactical_shift_score", "n_error_events",
    "window_end_minute", "period_end", "is_partial_window",
    "crosses_period_boundary",
]

DETECTION_FEATURES = [
    "event_count", "pass_count", "avg_pass_length", "pass_success_rate",
    "forward_rate", "long_pass_rate", "delta_pass_count",
    "delta_avg_pass_length", "delta_pass_success_rate", "delta_forward_rate",
    "delta_long_pass_rate", "tactical_shift_score", "n_error_events",
    "window_end_minute", "period_end", "is_partial_window",
    "crosses_period_boundary",
]


@dataclass(frozen=True)
class Paths:
    """Local inputs and generated output paths."""

    data_dir: Path
    output_dir: Path
    config_path: Path
    dictionary_path: Path
    independent_reference_path: Path | None


def parse_args() -> Paths:
    """Parse command-line paths with workspace defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "output" / "pipeline")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "reviewer_r2_audit")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "reviewer_r2_config.json",
    )
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=ROOT / "docs" / "DATA_DICTIONARY.md",
    )
    parser.add_argument(
        "--independent-reference",
        type=Path,
        default=None,
        help="Optional local Football-Data.co.uk SP1.csv copy.",
    )
    args = parser.parse_args()
    return Paths(
        args.data_dir,
        args.output_dir,
        args.config,
        args.dictionary,
        args.independent_reference,
    )


def component_table(data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Count each primary-label component at window and match level."""
    match_count = data["match_id"].nunique()
    rows: list[dict[str, Any]] = []
    for name, (column, threshold) in COMPONENTS.items():
        selected = data[column].ge(threshold)
        rows.append({
            "window_size": window_size,
            "component": name,
            "positive_windows": int(selected.sum()),
            "window_prevalence": float(selected.mean()),
            "matches_with_component": int(data.loc[selected, "match_id"].nunique()),
            "windows_per_match": float(selected.sum() / match_count),
            "matches_with_component_rate": float(
                data.loc[selected, "match_id"].nunique() / match_count
            ),
        })
    return pd.DataFrame(rows)


def window_size_analysis(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit sensitivity by component and the primary-label definition."""
    component_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for size in (25, 50, 100):
        frame = pd.read_pickle(paths.data_dir / f".window_{size}.pkl")
        component_frames.append(component_table(frame, size))
        component_without_formation = pd.Series(False, index=frame.index)
        for name, (column, threshold) in COMPONENTS.items():
            if name != "Formation Change":
                component_without_formation |= frame[column].ge(threshold)
        valid_duration = frame.loc[
            ~frame["crosses_period_boundary"]
            & frame["window_start_minute"].notna()
            & frame["window_end_minute"].notna()
        ]
        duration = (
            valid_duration["window_end_minute"] * 60
            + valid_duration["window_end_second"]
            - valid_duration["window_start_minute"] * 60
            - valid_duration["window_start_second"]
        ).clip(lower=0)
        summary_rows.append({
            "window_size": size,
            "matches": int(frame["match_id"].nunique()),
            "windows": int(len(frame)),
            "primary_windows": int(frame["primary_rare_event"].sum()),
            "primary_rate": float(frame["primary_rare_event"].mean()),
            "without_formation_windows": int(component_without_formation.sum()),
            "without_formation_rate": float(component_without_formation.mean()),
            "formation_only_windows": int(
                (frame["primary_rare_event"] & ~component_without_formation).sum()
            ),
            "partial_window_rate": float(frame["is_partial_window"].mean()),
            "median_duration_minutes": float(duration.median() / 60),
            "duration_q1_minutes": float(duration.quantile(0.25) / 60),
            "duration_q3_minutes": float(duration.quantile(0.75) / 60),
        })
    components = pd.concat(component_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    components.to_csv(paths.output_dir / "component_window_sensitivity.csv", index=False)
    summary.to_csv(paths.output_dir / "window_size_and_label_sensitivity.csv", index=False)
    return components, summary


def audit_dataset(data: pd.DataFrame, paths: Paths) -> dict[str, Any]:
    """Create season, score-maximum, co-occurrence, and goal checks."""
    season = (
        data.groupby("season", dropna=False)
        .agg(matches=("match_id", "nunique"), windows=("match_id", "size"))
        .reset_index()
        .sort_values("season")
    )
    season.to_csv(paths.output_dir / "matches_by_season.csv", index=False)
    high = data.loc[data["tactical_shift_score"].idxmax()]
    score_fields = [
        "match_id", "season", "team", "opponent", "time_window",
        "event_count", "pass_count", "is_partial_window",
        "crosses_period_boundary", "period_start", "period_end",
        "delta_pass_count", "delta_avg_pass_length",
        "delta_pass_success_rate", "delta_forward_rate",
        "delta_long_pass_rate", "tactical_shift_score",
    ]
    pd.DataFrame([high[score_fields].to_dict()]).to_csv(
        paths.output_dir / "maximum_tactical_shift_score_window.csv", index=False
    )
    formation = data["n_formation_changes"].ge(1)
    with_sub = formation & data["n_subs"].ge(1)
    goal_total = int(data["n_goals"].sum())
    shot_goal_total = goal_total - int(data["n_own_goals_for"].sum())
    return {
        "matches": int(data["match_id"].nunique()),
        "seasons": int(data["season"].nunique()),
        "rows": int(len(data)),
        "formation_windows": int(formation.sum()),
        "formation_with_substitution_windows": int(with_sub.sum()),
        "formation_with_substitution_rate": float(with_sub.sum() / formation.sum()),
        "formation_only_primary_windows": int(
            data["primary_rare_event_reason"].fillna("").eq("Formation Change").sum()
        ),
        "n_goals_including_own_goal_for": goal_total,
        "shot_goals_only": shot_goal_total,
        "own_goal_for": int(data["n_own_goals_for"].sum()),
        "score_max": float(high["tactical_shift_score"]),
    }


def build_modeling_data(data: pd.DataFrame) -> pd.DataFrame:
    """Build labels for each event type in the next three team windows."""
    ordered = data.sort_values(
        ["match_id", "team_id", "time_window"], kind="stable"
    ).copy()
    grouped = ordered.groupby(["match_id", "team_id"], sort=False)
    future = [grouped["primary_rare_event"].shift(-i) for i in (1, 2, 3)]
    ordered["target"] = pd.concat(future, axis=1).max(axis=1, skipna=True)
    for name, (column, threshold) in COMPONENTS.items():
        label = ordered[column].ge(threshold)
        shifted = [
            label.groupby([ordered["match_id"], ordered["team_id"]]).shift(-i)
            for i in (1, 2, 3)
        ]
        ordered[f"future_{column}"] = pd.concat(shifted, axis=1).max(
            axis=1, skipna=True
        )
    count_from_end = grouped.cumcount(ascending=False)
    modeling = ordered.loc[count_from_end.ge(3)].copy()
    modeling["target"] = modeling["target"].astype(int)
    return modeling


def calibration_bins(labels: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    """Summarize predicted and observed prevalence in equal-frequency bins."""
    frame = pd.DataFrame({"observed": labels, "predicted": probabilities})
    frame["bin"] = pd.qcut(
        frame["predicted"].rank(method="first"), q=10, labels=False
    )
    return frame.groupby("bin").agg(
        windows=("observed", "size"),
        mean_probability=("predicted", "mean"),
        observed_rate=("observed", "mean"),
    ).reset_index()


def write_calibration_figure(
    uncalibrated: pd.DataFrame,
    calibrated: pd.DataFrame,
    destination: Path,
) -> None:
    """Draw a publication-ready reliability diagram from test-set bins."""
    image = Image.new("RGB", (1100, 760), "white")
    draw = ImageDraw.Draw(image)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = (
        ImageFont.truetype(str(font_path), 24)
        if font_path.exists()
        else ImageFont.load_default()
    )
    title_font = (
        ImageFont.truetype(str(font_path), 32)
        if font_path.exists() else ImageFont.load_default()
    )
    left, top, right, bottom = 120, 100, 980, 650
    draw.text(
        (left, 30),
        "Next-three-window anticipation calibration",
        fill="#17212F",
        font=title_font,
    )
    draw.line((left, top, left, bottom), fill="#4B5563", width=3)
    draw.line((left, bottom, right, bottom), fill="#4B5563", width=3)
    for value in np.linspace(0, 1, 6):
        x = left + value * (right - left)
        y = bottom - value * (bottom - top)
        draw.line((x, bottom, x, bottom + 8), fill="#4B5563", width=2)
        draw.line((left - 8, y, left, y), fill="#4B5563", width=2)
        draw.text((x - 18, bottom + 14), f"{value:.1f}", fill="#374151", font=font)
        draw.text((25, y - 13), f"{value:.1f}", fill="#374151", font=font)
    draw.line((left, bottom, right, top), fill="#9CA3AF", width=3)
    for table, color in ((uncalibrated, "#B45309"), (calibrated, "#0E7490")):
        points = [(
            left + float(row.mean_probability) * (right - left),
            bottom - float(row.observed_rate) * (bottom - top),
        ) for row in table.itertuples()]
        if len(points) >= 2:
            draw.line(points, fill=color, width=5)
        for x, y in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
    draw.text((left + 200, 705), "Mean predicted probability", fill="#17212F", font=font)
    draw.text((left + 10, 80), "Observed event rate", fill="#17212F", font=font)
    draw.line((600, 80, 635, 80), fill="#B45309", width=5)
    draw.text((645, 65), "Uncalibrated", fill="#17212F", font=font)
    draw.line((820, 80, 855, 80), fill="#0E7490", width=5)
    draw.text((865, 65), "Platt", fill="#17212F", font=font)
    image.save(destination)


def predict_probability(design: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply a stable sigmoid transform to linear model scores."""
    logits = np.clip(design @ weights, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-logits))


def fit_newton_logistic(
    design: np.ndarray,
    labels: np.ndarray,
    balanced: bool,
    regularization: float = 0.01,
    iterations: int = 40,
) -> np.ndarray:
    """Fit a compact logistic model by regularized Newton iterations."""
    if labels.min() == labels.max():
        raise ValueError("Both target classes are required")
    weights = np.zeros(design.shape[1], dtype=float)
    sample_weights = np.ones(len(labels), dtype=float)
    if balanced:
        positives = int(labels.sum())
        negatives = len(labels) - positives
        sample_weights = np.where(
            labels == 1, len(labels) / (2 * positives),
            len(labels) / (2 * negatives),
        )
    penalty = np.eye(design.shape[1]) * regularization
    penalty[0, 0] = 0.0
    for _ in range(iterations):
        probabilities = predict_probability(design, weights)
        gradient = (
            design.T @ ((probabilities - labels) * sample_weights)
            / len(labels) + penalty @ weights
        )
        curvature = probabilities * (1 - probabilities) * sample_weights
        hessian = (
            design.T @ (design * curvature[:, None]) / len(labels)
            + penalty + np.eye(design.shape[1]) * 1e-8
        )
        step = np.linalg.solve(hessian, gradient)
        weights -= np.clip(step, -2.0, 2.0)
        if np.max(np.abs(step)) < 1e-7:
            break
    return weights


def expanded_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add nonlinear and time interaction terms to the base feature set."""
    expanded = frame[ANTICIPATION_FEATURES].astype(float).copy()
    for column in (
        "pass_count", "avg_pass_length", "pass_success_rate",
        "forward_rate", "long_pass_rate", "tactical_shift_score",
        "window_end_minute",
    ):
        expanded[f"{column}_squared"] = expanded[column] ** 2
    for column in (
        "pass_count", "pass_success_rate", "forward_rate",
        "tactical_shift_score", "n_error_events",
    ):
        expanded[f"{column}_by_time"] = (
            expanded[column] * expanded["window_end_minute"]
        )
    return expanded


def fit_and_score(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    target: str,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    test_index: np.ndarray,
    model_name: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Fit a balanced classifier and evaluate an untouched match test set."""
    design, _ = prepare_design_matrices(features.astype(float), train_index)
    labels = frame[target].to_numpy(dtype=int)
    weights = fit_newton_logistic(design[train_index], labels[train_index], True)
    validation_probability = predict_probability(design[validation_index], weights)
    test_probability = predict_probability(design[test_index], weights)
    threshold = best_f1_threshold(labels[validation_index], validation_probability)
    result = {
        "model": model_name,
        "task": target,
        "threshold": threshold,
        "test_prevalence": float(labels[test_index].mean()),
        **metric_bundle(labels[test_index], test_probability, threshold),
    }
    return result, validation_probability, test_probability


def benchmark(data: pd.DataFrame, paths: Paths) -> dict[str, Any]:
    """Run anticipation, time-only, calibration, category and detection checks."""
    frame = build_modeling_data(data)
    train_index, validation_index, test_index = split_by_match(frame, 20260908)
    partition = np.full(len(frame), "train", dtype=object)
    partition[validation_index] = "validation"
    partition[test_index] = "test"
    split_table = pd.DataFrame({"match_id": frame["match_id"], "split": partition})
    split_table = split_table.drop_duplicates().sort_values(["split", "match_id"])
    split_table.to_csv(paths.output_dir / "fixed_match_split_assignments.csv", index=False)
    if split_table["match_id"].duplicated().any():
        raise AssertionError("A match appears in more than one partition")

    metric_rows: list[dict[str, Any]] = []
    outputs: dict[str, np.ndarray] = {}
    for name, features, target in [
        ("balanced_logistic", frame[ANTICIPATION_FEATURES], "target"),
        ("time_period_only", frame[["window_end_minute", "period_end"]], "target"),
        ("expanded_logistic", expanded_features(frame), "target"),
    ]:
        result, validation_probability, test_probability = fit_and_score(
            frame, features, target, train_index, validation_index,
            test_index, name,
        )
        metric_rows.append(result)
        outputs[name] = test_probability
        if name == "balanced_logistic":
            raw_validation = np.log(
                np.clip(validation_probability, 1e-8, 1 - 1e-8)
                / np.clip(1 - validation_probability, 1e-8, 1 - 1e-8)
            )
            raw_test = np.log(
                np.clip(test_probability, 1e-8, 1 - 1e-8)
                / np.clip(1 - test_probability, 1e-8, 1 - 1e-8)
            )
            calibration_design = np.column_stack([
                np.ones(len(raw_validation)), raw_validation
            ])
            calibration_weights = fit_newton_logistic(
                calibration_design,
                frame["target"].iloc[validation_index].to_numpy(),
                balanced=False, regularization=1e-5,
            )
            calibrated_validation = predict_probability(
                calibration_design, calibration_weights
            )
            calibrated_test = predict_probability(
                np.column_stack([np.ones(len(raw_test)), raw_test]),
                calibration_weights,
            )
            calibration_threshold = best_f1_threshold(
                frame["target"].iloc[validation_index].to_numpy(),
                calibrated_validation,
            )
            metric_rows.append({
                "model": "balanced_logistic_platt_calibrated",
                "task": "target",
                "threshold": calibration_threshold,
                "test_prevalence": float(frame["target"].iloc[test_index].mean()),
                **metric_bundle(
                    frame["target"].iloc[test_index].to_numpy(),
                    calibrated_test,
                    calibration_threshold,
                ),
            })
            outputs["balanced_logistic_platt_calibrated"] = calibrated_test
            uncalibrated_bins = calibration_bins(
                frame["target"].iloc[test_index].to_numpy(), test_probability
            )
            calibrated_bins = calibration_bins(
                frame["target"].iloc[test_index].to_numpy(), calibrated_test
            )
            uncalibrated_bins.to_csv(
                paths.output_dir / "calibration_curve_uncalibrated.csv", index=False
            )
            calibrated_bins.to_csv(
                paths.output_dir / "calibration_curve_platt.csv", index=False
            )
            write_calibration_figure(
                uncalibrated_bins, calibrated_bins,
                paths.output_dir / "calibration_curve.png",
            )
    train_prevalence = float(frame["target"].iloc[train_index].mean())
    test_labels = frame["target"].iloc[test_index].to_numpy()
    test_prevalence = float(test_labels.mean())
    metric_rows.append({
        "model": "training_prevalence_baseline", "task": "target",
        "threshold": np.nan, "test_prevalence": test_prevalence,
        "brier_score": float(np.mean((test_labels - train_prevalence) ** 2)),
    })
    metric_rows.append({
        "model": "test_prevalence_oracle", "task": "target",
        "threshold": np.nan, "test_prevalence": test_prevalence,
        "brier_score": test_prevalence * (1 - test_prevalence),
    })
    pd.DataFrame(metric_rows).to_csv(paths.output_dir / "benchmark_metrics.csv", index=False)
    prediction = frame.iloc[test_index][["match_id", "team_id", "time_window", "target"]].copy()
    for name, values in outputs.items():
        prediction[f"probability_{name}"] = values
    prediction.to_csv(paths.output_dir / "test_predictions.csv", index=False)

    category_rows: list[dict[str, Any]] = []
    risk = outputs["expanded_logistic"]
    threshold = next(
        row["threshold"] for row in metric_rows if row["model"] == "expanded_logistic"
    )
    for name, (column, _) in COMPONENTS.items():
        label = frame[f"future_{column}"].iloc[test_index].to_numpy(dtype=int)
        positives = int(label.sum())
        if positives == 0:
            continue
        scores = metric_bundle(label, risk, threshold)
        category_rows.append({
            "component": name,
            "test_positive_windows": positives,
            "interpretation": "general primary-event alert against this component",
            **scores,
        })
    pd.DataFrame(category_rows).to_csv(
        paths.output_dir / "per_component_alert_performance.csv", index=False
    )

    detection = data.copy()
    detect_train, detect_valid, detect_test = split_by_match(detection, 20260908)
    detection["target"] = detection["primary_rare_event"].astype(int)
    detection_result, _, _ = fit_and_score(
        detection, detection[DETECTION_FEATURES], "target",
        detect_train, detect_valid, detect_test, "balanced_logistic",
    )
    detection_result["model"] = "retrospective_detection_logistic"
    pd.DataFrame([detection_result]).to_csv(
        paths.output_dir / "detection_baseline_metrics.csv", index=False
    )
    return {
        "anticipation_rows": int(len(frame)),
        "split_match_counts": split_table["split"].value_counts().to_dict(),
        "test_prevalence": test_prevalence,
        "benchmark_metrics": metric_rows,
        "detection_baseline": detection_result,
    }


def write_dictionary(paths: Paths, data: pd.DataFrame) -> None:
    """Expand the existing 56-column dictionary with units and missing semantics."""
    lines = paths.dictionary_path.read_text(encoding="utf-8").splitlines()
    definitions: dict[str, tuple[str, str]] = {}
    for line in lines:
        if line.startswith("| `"):
            parts = [part.strip() for part in line.strip("|").split("|")]
            if len(parts) >= 3:
                definitions[parts[0].strip("`")] = (parts[1], parts[2])
    if set(definitions) != set(data.columns):
        raise ValueError("Dictionary fields do not match the exported dataset")
    context = {
        "competition_id", "competition", "season_id", "season", "match_id",
        "match_date", "team_id", "team", "opponent_id", "opponent",
        "home_away", "time_window", "start_event_index", "end_event_index",
    }
    detection_only = {
        "n_goals", "n_own_goals_for", "n_own_goals_against", "n_own_goals",
        "n_red_cards", "n_second_yellows", "n_subs", "n_injuries",
        "n_unexpected_goals", "n_penalty_goals", "n_formation_changes",
        "n_starting_xi_events", "has_formation_change", "is_error_cluster",
    }
    target = {
        "primary_rare_event", "primary_rare_event_reason",
        "broad_rare_event", "broad_rare_event_reason",
    }
    rates = {"pass_success_rate", "forward_rate", "long_pass_rate",
             "delta_pass_success_rate", "delta_forward_rate", "delta_long_pass_rate"}
    pitch = {"avg_pass_length", "delta_avg_pass_length"}
    counts = {column for column in data.columns if column.startswith("n_")}
    rows: list[dict[str, Any]] = []
    for column in data.columns:
        dtype, definition = definitions[column]
        if column in {
            "window_start_minute", "window_end_minute",
            "window_start_second", "window_end_second",
        }:
            dtype = "float (whole-valued)"
        if column in {"primary_rare_event_reason", "broad_rare_event_reason"}:
            definition = definition.replace("Semicolon-separated", "Comma-separated")
        if column in target:
            eligibility = "Target or target explanation; never a predictor"
        elif column in context:
            eligibility = "Context or identifier; withheld from reported models"
        elif column in detection_only:
            eligibility = "Retrospective label-rule input; excluded from model predictors"
        else:
            eligibility = "Both at completed-window boundary"
        if column in rates:
            unit = "proportion, or proportion-point change"
        elif column in pitch:
            unit = "StatsBomb pitch units"
        elif column == "tactical_shift_score":
            unit = "mixed-unit sum (not physically interpretable)"
        elif column in counts or column in {"event_count", "pass_count", "delta_pass_count"}:
            unit = "events per window"
        elif "minute" in column:
            unit = "match minutes"
        elif "second" in column:
            unit = "seconds within minute"
        elif column == "match_date":
            unit = "calendar date"
        elif column in {"period_start", "period_end"}:
            unit = "period number"
        else:
            unit = "category, index, ID, or boolean"
        if column in {"avg_pass_length", "pass_success_rate", "forward_rate", "long_pass_rate"}:
            missing = "Undefined when the window contains no valid passes"
        elif (
            column.startswith("delta_") and column != "delta_available"
        ) or column == "tactical_shift_score":
            missing = "No preceding team window or unavailable pass aggregate"
        elif column.endswith("reason"):
            missing = "Blank when the associated label is false"
        else:
            missing = "No missing values in this release"
        rows.append({
            "name": column, "type": dtype, "unit": unit,
            "definition": definition, "missing_value_meaning": missing,
            "model_eligibility": eligibility,
            "observed_missing": int(data[column].isna().sum()),
        })
    table = pd.DataFrame(rows)
    table.to_csv(paths.output_dir / "complete_56_column_dictionary.csv", index=False)
    table[["name", "model_eligibility"]].to_csv(
        paths.output_dir / "feature_eligibility.csv", index=False
    )


def independent_validation(data: pd.DataFrame, paths: Paths) -> dict[str, Any]:
    """Compare 2015/16 match scores and red cards with Football-Data.co.uk."""
    source_url = "https://www.football-data.co.uk/mmz4281/1516/SP1.csv"
    reference_source: str | Path = (
        paths.independent_reference_path
        if paths.independent_reference_path is not None
        else source_url
    )
    reference = pd.read_csv(reference_source, encoding="latin1")
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HR", "AR"}
    if not required.issubset(reference):
        raise ValueError("Independent source schema changed")
    name_map = {
        "Athletic Club": "Ath Bilbao",
        "Atlético Madrid": "Ath Madrid",
        "Celta Vigo": "Celta",
        "Espanyol": "Espanol",
        "Levante UD": "Levante",
        "Málaga": "Malaga",
        "RC Deportivo La Coruña": "La Coruna",
        "Rayo Vallecano": "Vallecano",
        "Real Betis": "Betis",
        "Real Sociedad": "Sociedad",
        "Sporting Gijón": "Sp Gijon",
    }
    subset = data.loc[data["season"].eq("2015/2016")].copy()
    grouped = subset.groupby(["match_id", "match_date", "home_away"], as_index=False).agg(
        team=("team", "first"),
        goals=("n_goals", "sum"),
        red_cards=("n_red_cards", "sum"),
        second_yellows=("n_second_yellows", "sum"),
    )
    home = grouped.loc[grouped["home_away"].eq("home")].drop(columns="home_away")
    away = grouped.loc[grouped["home_away"].eq("away")].drop(columns="home_away")
    home = home.rename(columns={
        "team": "HomeTeam", "goals": "source_home_goals",
        "red_cards": "source_home_red", "second_yellows": "source_home_second_yellow",
    })
    away = away.rename(columns={
        "team": "AwayTeam", "goals": "source_away_goals",
        "red_cards": "source_away_red", "second_yellows": "source_away_second_yellow",
    })
    ours = home.merge(away, on=["match_id", "match_date"], validate="one_to_one")
    ours["HomeTeam"] = ours["HomeTeam"].replace(name_map)
    ours["AwayTeam"] = ours["AwayTeam"].replace(name_map)
    reference["match_date"] = pd.to_datetime(
        reference["Date"], dayfirst=True, format="%d/%m/%y"
    ).dt.strftime("%Y-%m-%d")
    joined = ours.merge(
        reference[list(required) + ["match_date"]],
        on=["match_date", "HomeTeam", "AwayTeam"],
        how="left", validate="one_to_one", indicator=True,
    )
    joined["goal_match"] = (
        joined["source_home_goals"].eq(joined["FTHG"])
        & joined["source_away_goals"].eq(joined["FTAG"])
    )
    joined["card_match"] = (
        (joined["source_home_red"] + joined["source_home_second_yellow"]).eq(joined["HR"])
        & (joined["source_away_red"] + joined["source_away_second_yellow"]).eq(joined["AR"])
    )
    joined.to_csv(paths.output_dir / "independent_match_validation_all_2015_16.csv", index=False)
    sample = joined.sample(n=min(50, len(joined)), random_state=20260908)
    sample.sort_values("match_id").to_csv(
        paths.output_dir / "independent_match_validation_sample.csv", index=False
    )
    return {
        "source": source_url,
        "reference_input": str(reference_source),
        "2015_16_matches": int(len(joined)),
        "matched_to_reference": int(joined["_merge"].eq("both").sum()),
        "independent_goal_matches": int(joined["goal_match"].sum()),
        "independent_card_matches": int(joined["card_match"].sum()),
        "sample_size": int(len(sample)),
        "sample_goal_matches": int(sample["goal_match"].sum()),
        "sample_card_matches": int(sample["card_match"].sum()),
    }


def expert_review_sample(data: pd.DataFrame, paths: Paths) -> dict[str, int]:
    """Prepare blinded expert review records without assigning expert labels."""
    rng = np.random.default_rng(20260908)
    samples: list[pd.DataFrame] = []
    for name, column in [
        ("Formation Change", "n_formation_changes"),
        ("Unexpected Goal", "n_unexpected_goals"),
    ]:
        candidates = data.loc[data[column].ge(1)].copy()
        selected_index = rng.choice(
            candidates.index.to_numpy(), size=min(30, len(candidates)), replace=False
        )
        selected = candidates.loc[selected_index, [
            "match_id", "season", "match_date", "team", "opponent",
            "time_window", "start_event_index", "end_event_index",
            "period_start", "period_end", "window_start_minute",
            "window_end_minute",
        ]].copy()
        selected.insert(0, "event_category", name)
        samples.append(selected)
    sheet = pd.concat(samples, ignore_index=True).sample(
        frac=1, random_state=20260908
    ).reset_index(drop=True)
    sheet.insert(0, "blind_case_id", [f"R2-{i:03d}" for i in range(1, len(sheet) + 1)])
    for column in (
        "expert_1_confirmed", "expert_1_notes", "expert_2_confirmed",
        "expert_2_notes", "adjudicated_label",
    ):
        sheet[column] = ""
    sheet.to_csv(paths.output_dir / "expert_review_blank_sample.csv", index=False)
    return {"formation_change_cases": 30, "unexpected_goal_cases": 30}


def main() -> None:
    """Generate all reproducible reviewer analysis artifacts."""
    paths = parse_args()
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(paths.data_dir / "football_team_window_dataset_v2.csv")
    config = PipelineConfig.from_json(paths.config_path)
    if len(data.columns) != 56 or data.empty:
        raise ValueError("Expected a non-empty, corrected 56-column dataset")
    if data["match_id"].nunique() < 578:
        raise ValueError("The match audit regressed below the prior release")
    components, sensitivity = window_size_analysis(paths)
    audit = audit_dataset(data, paths)
    write_dictionary(paths, data)
    models = benchmark(data, paths)
    validation = independent_validation(data, paths)
    expert_sample = expert_review_sample(data, paths)
    summary = {
        "config": {"seed": config.random_seed},
        "dataset": audit,
        "window_sensitivity": sensitivity.to_dict(orient="records"),
        "component_rates": components.to_dict(orient="records"),
        "models": models,
        "independent_validation": validation,
        "expert_review_sample": expert_sample,
    }
    (paths.output_dir / "reviewer_r2_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps(
        {"dataset": audit, "models": models, "validation": validation},
        indent=2, default=str,
    ))


if __name__ == "__main__":
    main()
