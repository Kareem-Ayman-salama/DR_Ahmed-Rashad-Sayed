"""Rebuild the football rare-event dataset for the editor revision.

The pipeline downloads the official StatsBomb Open Data JSON files, preserves
source metadata, creates leakage-aware team windows, and exports corrected
statistics used by the revised manuscript.
"""

from __future__ import annotations

import argparse
import http.client
import json
import logging
import math
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for source acquisition and dataset construction."""

    competition_id: int
    competition_name: str
    window_sizes: tuple[int, ...]
    primary_window_size: int
    long_pass_threshold: float
    unexpected_goal_xg_threshold: float
    substitution_burst_threshold: int
    error_cluster_threshold: int
    download_workers: int
    use_cached_events_only: bool
    random_seed: int
    bootstrap_resamples: int
    source_base_url: str

    @classmethod
    def from_json(cls, path: Path) -> "PipelineConfig":
        """Load pipeline configuration from a JSON file.

        Args:
            path: Configuration file path.

        Returns:
            Parsed immutable configuration.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["window_sizes"] = tuple(int(value) for value in payload["window_sizes"])
        return cls(**payload)


@dataclass(frozen=True)
class SourceRecord:
    """Metadata for one StatsBomb match."""

    match_id: int
    season_id: int
    season_name: str
    home_team_id: int
    home_team: str
    away_team_id: int
    away_team: str
    home_score: int
    away_score: int
    match_date: str


def configure_logging() -> None:
    """Configure concise progress logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def request_json(url: str, timeout_seconds: int = 120) -> Any:
    """Fetch and decode a JSON document with retry handling.

    Args:
        url: Remote JSON URL.
        timeout_seconds: Per-request timeout.

    Returns:
        Decoded JSON payload.

    Raises:
        RuntimeError: If all download attempts fail.
    """
    headers = {"User-Agent": "football-rare-event-editor-revision/2.0"}
    error: Exception | None = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (
            urllib.error.URLError,
            http.client.IncompleteRead,
            TimeoutError,
            ValueError,
            OSError,
        ) as exc:
            error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Unable to download JSON after retries: {url}") from error


def load_or_download_json(url: str, destination: Path) -> Any:
    """Load cached JSON or download it atomically.

    Args:
        url: Remote JSON URL.
        destination: Local cache path.

    Returns:
        Decoded JSON payload.
    """
    if destination.exists():
        return json.loads(destination.read_text(encoding="utf-8"))
    payload = request_json(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(destination)
    return payload


def collect_source_records(config: PipelineConfig, cache_dir: Path) -> list[SourceRecord]:
    """Download competition and match metadata for the configured competition.

    Args:
        config: Pipeline configuration.
        cache_dir: Local source cache.

    Returns:
        Deduplicated match records sorted by date and match identifier.
    """
    competitions_url = f"{config.source_base_url}/competitions.json"
    competitions = load_or_download_json(
        competitions_url,
        cache_dir / "competitions.json",
    )
    seasons = sorted(
        (
            item
            for item in competitions
            if int(item["competition_id"]) == config.competition_id
        ),
        key=lambda item: str(item["season_name"]),
    )
    if not seasons:
        raise ValueError(f"No seasons found for competition {config.competition_id}")

    records: dict[int, SourceRecord] = {}
    for season in seasons:
        season_id = int(season["season_id"])
        season_name = str(season["season_name"])
        matches_url = (
            f"{config.source_base_url}/matches/"
            f"{config.competition_id}/{season_id}.json"
        )
        matches = load_or_download_json(
            matches_url,
            cache_dir / "matches" / str(config.competition_id) / f"{season_id}.json",
        )
        for match in matches:
            match_id = int(match["match_id"])
            home = match["home_team"]
            away = match["away_team"]
            records[match_id] = SourceRecord(
                match_id=match_id,
                season_id=season_id,
                season_name=season_name,
                home_team_id=int(home["home_team_id"]),
                home_team=str(home["home_team_name"]),
                away_team_id=int(away["away_team_id"]),
                away_team=str(away["away_team_name"]),
                home_score=int(match["home_score"]),
                away_score=int(match["away_score"]),
                match_date=str(match.get("match_date", "")),
            )
    return sorted(records.values(), key=lambda item: (item.match_date, item.match_id))


def download_event_files(
    records: Sequence[SourceRecord],
    config: PipelineConfig,
    cache_dir: Path,
) -> None:
    """Download all event files required by the selected matches.

    Args:
        records: Match metadata records.
        config: Pipeline configuration.
        cache_dir: Local source cache.
    """
    events_dir = cache_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    pending = [record for record in records if not (events_dir / f"{record.match_id}.json").exists()]
    LOGGER.info("Event files cached=%d pending=%d", len(records) - len(pending), len(pending))
    if config.use_cached_events_only:
        LOGGER.warning(
            "Cached-only mode enabled; %d unavailable event files will be excluded",
            len(pending),
        )
        return

    def download(record: SourceRecord) -> int:
        url = f"{config.source_base_url}/events/{record.match_id}.json"
        load_or_download_json(url, events_dir / f"{record.match_id}.json")
        return record.match_id

    with ThreadPoolExecutor(max_workers=config.download_workers) as executor:
        futures = {executor.submit(download, record): record for record in pending}
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 50 == 0 or completed == len(pending):
                LOGGER.info("Downloaded %d/%d pending event files", completed, len(pending))


def nested_name(event: dict[str, Any], key: str, nested_key: str | None = None) -> str | None:
    """Return a nested StatsBomb name field when present.

    Args:
        event: Raw event object.
        key: Top-level object key.
        nested_key: Optional child key before the name field.

    Returns:
        String value or ``None``.
    """
    value: Any = event.get(key)
    if nested_key is not None and isinstance(value, dict):
        value = value.get(nested_key)
    if isinstance(value, dict):
        name = value.get("name")
        return str(name) if name is not None else None
    return None


def timestamp_seconds(value: Any) -> float:
    """Convert a StatsBomb timestamp string to seconds within a period.

    Args:
        value: Timestamp value in ``HH:MM:SS.sss`` format.

    Returns:
        Seconds elapsed within the period.
    """
    try:
        hours, minutes, seconds = str(value).split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (TypeError, ValueError):
        return math.nan


def flatten_events(payload: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Flatten raw StatsBomb events into analysis columns.

    Args:
        payload: Raw match event objects.

    Returns:
        Event-level dataframe.
    """
    rows: list[dict[str, Any]] = []
    for event in payload:
        event_type = event.get("type") or {}
        team = event.get("team") or {}
        shot = event.get("shot") or {}
        pass_data = event.get("pass") or {}
        foul = event.get("foul_committed") or {}
        bad_behaviour = event.get("bad_behaviour") or {}
        location = event.get("location") or []
        pass_end = pass_data.get("end_location") or []
        row = {
            "event_id": str(event.get("id", "")),
            "event_index": int(event.get("index", -1)),
            "period": int(event.get("period", 0)),
            "timestamp": str(event.get("timestamp", "")),
            "period_seconds": timestamp_seconds(event.get("timestamp")),
            "minute": float(event.get("minute", math.nan)),
            "second": float(event.get("second", math.nan)),
            "type_id": int(event_type.get("id", -1)),
            "type": str(event_type.get("name", "")),
            "team_id": int(team.get("id", -1)) if team else -1,
            "team": str(team.get("name", "")) if team else "",
            "possession": int(event.get("possession", -1)),
            "x": float(location[0]) if len(location) >= 2 else math.nan,
            "y": float(location[1]) if len(location) >= 2 else math.nan,
            "end_x": float(pass_end[0]) if len(pass_end) >= 2 else math.nan,
            "end_y": float(pass_end[1]) if len(pass_end) >= 2 else math.nan,
            "pass_outcome": nested_name(event, "pass", "outcome"),
            "shot_outcome": (
                str(shot.get("outcome", {}).get("name"))
                if isinstance(shot.get("outcome"), dict)
                else None
            ),
            "shot_type": (
                str(shot.get("type", {}).get("name"))
                if isinstance(shot.get("type"), dict)
                else None
            ),
            "shot_xg": pd.to_numeric(shot.get("statsbomb_xg"), errors="coerce"),
            "foul_card": (
                str(foul.get("card", {}).get("name"))
                if isinstance(foul.get("card"), dict)
                else None
            ),
            "bad_behaviour_card": (
                str(bad_behaviour.get("card", {}).get("name"))
                if isinstance(bad_behaviour.get("card"), dict)
                else None
            ),
            "formation": pd.to_numeric(
                (event.get("tactics") or {}).get("formation"),
                errors="coerce",
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def card_matches(frame: pd.DataFrame, value: str) -> pd.Series:
    """Match a card label across both StatsBomb card fields.

    Args:
        frame: Event dataframe.
        value: Exact card label.

    Returns:
        Boolean event mask.
    """
    foul = frame["foul_card"].fillna("").astype(str)
    bad = frame["bad_behaviour_card"].fillna("").astype(str)
    return foul.eq(value) | bad.eq(value)


def label_event_rows(frame: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Add pass, event-count, and rare-event indicators.

    Args:
        frame: Flattened event dataframe.
        config: Pipeline configuration.

    Returns:
        Copy with derived event-level variables.
    """
    data = frame.copy()
    data["is_pass"] = data["type"].eq("Pass")
    data["pass_length"] = np.sqrt(
        (data["end_x"] - data["x"]) ** 2 + (data["end_y"] - data["y"]) ** 2
    )
    data["pass_success"] = data["is_pass"] & data["pass_outcome"].isna()
    data["forward_pass"] = data["is_pass"] & data["end_x"].gt(data["x"])
    data["long_pass"] = data["is_pass"] & data["pass_length"].gt(config.long_pass_threshold)
    data["is_goal"] = data["type"].eq("Shot") & data["shot_outcome"].eq("Goal")
    data["is_own_goal_for"] = data["type"].eq("Own Goal For")
    data["is_own_goal_against"] = data["type"].eq("Own Goal Against")
    data["is_red_card"] = card_matches(data, "Red Card")
    data["is_second_yellow"] = card_matches(data, "Second Yellow")
    data["is_substitution"] = data["type"].eq("Substitution")
    data["is_injury"] = data["type"].eq("Injury Stoppage")
    data["is_formation_change"] = data["type"].eq("Tactical Shift")
    data["is_starting_xi"] = data["type"].eq("Starting XI")
    data["is_penalty_goal"] = data["is_goal"] & data["shot_type"].eq("Penalty")
    data["is_unexpected_goal"] = (
        data["is_goal"]
        & data["shot_xg"].lt(config.unexpected_goal_xg_threshold)
    )
    data["is_error_event"] = data["type"].isin(["Error", "Miscontrol", "Dispossessed"])
    return data


def aggregate_team_windows(
    events: pd.DataFrame,
    record: SourceRecord,
    team_id: int,
    team_name: str,
    opponent_id: int,
    opponent_name: str,
    home_away: str,
    window_size: int,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Aggregate one team-match event stream into fixed event-count windows.

    Args:
        events: Match-level event dataframe.
        record: Match metadata.
        team_id: Numeric team identifier.
        team_name: Team name.
        opponent_id: Numeric opponent identifier.
        opponent_name: Opponent name.
        home_away: ``home`` or ``away``.
        window_size: Number of team events per window.
        config: Pipeline configuration.

    Returns:
        Team-window dataframe.
    """
    team_events = events.loc[events["team_id"].eq(team_id)].copy()
    team_events = team_events.sort_values(
        ["period", "event_index", "period_seconds"],
        kind="stable",
    ).reset_index(drop=True)
    if team_events.empty:
        return pd.DataFrame()
    team_events["time_window"] = np.arange(len(team_events), dtype=int) // window_size

    rows: list[dict[str, Any]] = []
    for time_window, group in team_events.groupby("time_window", sort=True):
        passes = group.loc[group["is_pass"]]
        valid_passes = passes.loc[passes["pass_length"].notna()]
        event_count = int(len(group))
        pass_count = int(len(passes))
        error_count = int(group["is_error_event"].sum())
        formation_count = int(group["is_formation_change"].sum())
        starting_xi_count = int(group["is_starting_xi"].sum())
        n_own_for = int(group["is_own_goal_for"].sum())
        n_own_against = int(group["is_own_goal_against"].sum())

        primary_flags = {
            "Red Card": int(group["is_red_card"].sum()) >= 1,
            "Second Yellow": int(group["is_second_yellow"].sum()) >= 1,
            "Substitution Burst": int(group["is_substitution"].sum())
            >= config.substitution_burst_threshold,
            "Unexpected Goal": int(group["is_unexpected_goal"].sum()) >= 1,
            "Own Goal": (n_own_for + n_own_against) >= 1,
            "Formation Change": formation_count >= 1,
            "Penalty Goal": int(group["is_penalty_goal"].sum()) >= 1,
        }
        primary_reasons = [name for name, active in primary_flags.items() if active]
        is_error_cluster = error_count >= config.error_cluster_threshold
        broad_reasons = primary_reasons + (["Error Cluster"] if is_error_cluster else [])
        row = {
            "competition_id": config.competition_id,
            "competition": config.competition_name,
            "season_id": record.season_id,
            "season": record.season_name,
            "match_id": record.match_id,
            "match_date": record.match_date,
            "team_id": team_id,
            "team": team_name,
            "opponent_id": opponent_id,
            "opponent": opponent_name,
            "home_away": home_away,
            "time_window": int(time_window),
            "event_count": event_count,
            "is_partial_window": event_count < window_size,
            "start_event_index": int(group["event_index"].iloc[0]),
            "end_event_index": int(group["event_index"].iloc[-1]),
            "period_start": int(group["period"].iloc[0]),
            "period_end": int(group["period"].iloc[-1]),
            "window_start_minute": float(group["minute"].iloc[0]),
            "window_end_minute": float(group["minute"].iloc[-1]),
            "window_start_second": float(group["second"].iloc[0]),
            "window_end_second": float(group["second"].iloc[-1]),
            "crosses_period_boundary": int(group["period"].nunique()) > 1,
            "pass_count": pass_count,
            "has_pass_data": pass_count > 0,
            "avg_pass_length": float(valid_passes["pass_length"].mean())
            if not valid_passes.empty
            else math.nan,
            "pass_success_rate": float(passes["pass_success"].mean())
            if not passes.empty
            else math.nan,
            "forward_rate": float(passes["forward_pass"].mean())
            if not passes.empty
            else math.nan,
            "long_pass_rate": float(passes["long_pass"].mean())
            if not passes.empty
            else math.nan,
            "n_goals": int(group["is_goal"].sum()) + n_own_for,
            "n_own_goals_for": n_own_for,
            "n_own_goals_against": n_own_against,
            "n_own_goals": n_own_for + n_own_against,
            "n_red_cards": int(group["is_red_card"].sum()),
            "n_second_yellows": int(group["is_second_yellow"].sum()),
            "n_subs": int(group["is_substitution"].sum()),
            "n_injuries": int(group["is_injury"].sum()),
            "n_unexpected_goals": int(group["is_unexpected_goal"].sum()),
            "n_penalty_goals": int(group["is_penalty_goal"].sum()),
            "n_error_events": error_count,
            "n_formation_changes": formation_count,
            "n_starting_xi_events": starting_xi_count,
            "has_formation_change": formation_count > 0,
            "is_error_cluster": is_error_cluster,
            "primary_rare_event": bool(primary_reasons),
            "primary_rare_event_reason": ", ".join(primary_reasons) if primary_reasons else "None",
            "broad_rare_event": bool(broad_reasons),
            "broad_rare_event_reason": ", ".join(broad_reasons) if broad_reasons else "None",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_window_dataset(
    records: Sequence[SourceRecord],
    config: PipelineConfig,
    cache_dir: Path,
    window_size: int,
) -> pd.DataFrame:
    """Build all team windows for one event-count window size.

    Args:
        records: Match metadata.
        config: Pipeline configuration.
        cache_dir: Local JSON cache.
        window_size: Team-event window size.

    Returns:
        Combined team-window dataframe.
    """
    frames: list[pd.DataFrame] = []
    for index, record in enumerate(records, start=1):
        payload = json.loads(
            (cache_dir / "events" / f"{record.match_id}.json").read_text(encoding="utf-8")
        )
        events = label_event_rows(flatten_events(payload), config)
        frames.append(
            aggregate_team_windows(
                events,
                record,
                record.home_team_id,
                record.home_team,
                record.away_team_id,
                record.away_team,
                "home",
                window_size,
                config,
            )
        )
        frames.append(
            aggregate_team_windows(
                events,
                record,
                record.away_team_id,
                record.away_team,
                record.home_team_id,
                record.home_team,
                "away",
                window_size,
                config,
            )
        )
        if index % 100 == 0 or index == len(records):
            LOGGER.info("Aggregated window=%d matches=%d/%d", window_size, index, len(records))
    data = pd.concat(frames, ignore_index=True)
    return data.sort_values(["match_id", "team_id", "time_window"], kind="stable").reset_index(drop=True)


def add_delta_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add within-team-match deltas without crossing stream boundaries.

    First-window deltas and deltas involving unavailable pass aggregates remain
    missing. This preserves the distinction between no pass data and a numeric
    value of zero.

    Args:
        data: Primary team-window dataframe.

    Returns:
        Copy with delta variables and tactical-shift score.
    """
    result = data.copy()
    group_columns = ["match_id", "team_id"]
    features = [
        "pass_count",
        "avg_pass_length",
        "pass_success_rate",
        "forward_rate",
        "long_pass_rate",
    ]
    result["has_previous_window"] = result.groupby(group_columns).cumcount().gt(0)
    for feature in features:
        result[f"delta_{feature}"] = (
            result.groupby(group_columns, sort=False)[feature].diff().abs()
        )
    delta_columns = [f"delta_{feature}" for feature in features]
    result["delta_available"] = result[delta_columns].notna().all(axis=1)
    result["tactical_shift_score"] = result[delta_columns].sum(axis=1, min_count=len(delta_columns))
    return result


def validation_summary(
    records: Sequence[SourceRecord],
    config: PipelineConfig,
    cache_dir: Path,
) -> dict[str, Any]:
    """Validate goal extraction against match-level score metadata.

    Args:
        records: Match metadata.
        config: Pipeline configuration.
        cache_dir: Local JSON cache.

    Returns:
        Validation counts and mismatch examples.
    """
    mismatches: list[dict[str, Any]] = []
    for record in records:
        payload = json.loads(
            (cache_dir / "events" / f"{record.match_id}.json").read_text(encoding="utf-8")
        )
        events = label_event_rows(flatten_events(payload), config)
        shots = events.loc[events["is_goal"]].groupby("team_id").size().to_dict()
        own_for = events.loc[events["is_own_goal_for"]].groupby("team_id").size().to_dict()
        home_detected = int(shots.get(record.home_team_id, 0) + own_for.get(record.home_team_id, 0))
        away_detected = int(shots.get(record.away_team_id, 0) + own_for.get(record.away_team_id, 0))
        if home_detected != record.home_score or away_detected != record.away_score:
            mismatches.append(
                {
                    "match_id": record.match_id,
                    "home_expected": record.home_score,
                    "home_detected": home_detected,
                    "away_expected": record.away_score,
                    "away_detected": away_detected,
                }
            )
    return {
        "matches_checked": len(records),
        "matches_matching_score_metadata": len(records) - len(mismatches),
        "score_match_rate": (len(records) - len(mismatches)) / len(records),
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:20],
    }


def split_by_match(
    data: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic 70/10/20 grouped row indices by match.

    Args:
        data: Modeling dataframe.
        seed: Random seed.

    Returns:
        Train, validation, and test row-index arrays.
    """
    rng = np.random.default_rng(seed)
    match_ids = data["match_id"].drop_duplicates().to_numpy().copy()
    rng.shuffle(match_ids)
    test_count = max(1, int(round(len(match_ids) * 0.20)))
    validation_count = max(1, int(round(len(match_ids) * 0.10)))
    test_matches = set(match_ids[:test_count].tolist())
    validation_matches = set(
        match_ids[test_count : test_count + validation_count].tolist()
    )
    match_values = data["match_id"].to_numpy()
    test = np.flatnonzero(np.isin(match_values, list(test_matches)))
    validation = np.flatnonzero(np.isin(match_values, list(validation_matches)))
    train = np.flatnonzero(~np.isin(match_values, list(test_matches | validation_matches)))
    return train, validation, test


def binary_roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Calculate ROC-AUC from average probability ranks.

    Args:
        y_true: Binary labels.
        probabilities: Positive-class probabilities.

    Returns:
        ROC-AUC value.
    """
    labels = np.asarray(y_true, dtype=int)
    ranks = pd.Series(probabilities).rank(method="average").to_numpy()
    positives = labels == 1
    positive_count = int(positives.sum())
    negative_count = int((~positives).sum())
    if positive_count == 0 or negative_count == 0:
        return math.nan
    numerator = ranks[positives].sum() - positive_count * (positive_count + 1) / 2
    return float(numerator / (positive_count * negative_count))


def binary_average_precision(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Calculate non-interpolated average precision.

    Args:
        y_true: Binary labels.
        probabilities: Positive-class probabilities.

    Returns:
        Average precision value.
    """
    labels = np.asarray(y_true, dtype=int)
    order = np.argsort(-np.asarray(probabilities), kind="stable")
    sorted_labels = labels[order]
    positive_count = int(sorted_labels.sum())
    if positive_count == 0:
        return math.nan
    cumulative_positives = np.cumsum(sorted_labels)
    precision = cumulative_positives / np.arange(1, len(sorted_labels) + 1)
    return float(precision[sorted_labels == 1].sum() / positive_count)


def fit_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    iterations: int = 1200,
    learning_rate: float = 0.03,
    l2_penalty: float = 0.01,
) -> np.ndarray:
    """Fit a class-balanced logistic regression with batch gradient descent.

    Args:
        x_train: Standardized training design matrix with bias column.
        y_train: Binary training labels.
        seed: Reproducibility seed.
        iterations: Optimization iterations.
        learning_rate: Gradient step size.
        l2_penalty: L2 regularization coefficient.

    Returns:
        Fitted coefficient vector.
    """
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=x_train.shape[1])
    positive_count = max(1, int(y_train.sum()))
    negative_count = max(1, int((1 - y_train).sum()))
    sample_weights = np.where(
        y_train == 1,
        len(y_train) / (2 * positive_count),
        len(y_train) / (2 * negative_count),
    )
    for _ in range(iterations):
        logits = np.clip(x_train @ weights, -35.0, 35.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        residual = (probabilities - y_train) * sample_weights
        gradient = (x_train.T @ residual) / len(y_train)
        gradient[1:] += l2_penalty * weights[1:]
        weights -= learning_rate * gradient
    return weights


def prepare_design_matrices(
    frame: pd.DataFrame,
    train_index: np.ndarray,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Median-impute and standardize features using training statistics.

    Args:
        frame: Numeric feature dataframe.
        train_index: Training row indices.

    Returns:
        Design matrix with a bias column and reusable transform metadata.
    """
    values = frame.to_numpy(dtype=float)
    training = values[train_index]
    medians = np.nanmedian(training, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    imputed = np.where(np.isnan(values), medians, values)
    means = imputed[train_index].mean(axis=0)
    scales = imputed[train_index].std(axis=0)
    scales = np.where(scales < 1e-12, 1.0, scales)
    standardized = (imputed - means) / scales
    design = np.column_stack([np.ones(len(standardized)), standardized])
    metadata = {
        "medians": medians.tolist(),
        "means": means.tolist(),
        "scales": scales.tolist(),
    }
    return design, metadata


def best_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Select an F1-maximizing validation threshold.

    Args:
        y_true: Binary validation labels.
        probabilities: Positive-class probabilities.

    Returns:
        Selected threshold.
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = []
    for threshold in thresholds:
        predictions = probabilities >= threshold
        true_positive = int(((predictions == 1) & (y_true == 1)).sum())
        false_positive = int(((predictions == 1) & (y_true == 0)).sum())
        false_negative = int(((predictions == 0) & (y_true == 1)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append((2 * true_positive / denominator) if denominator else 0.0)
    return float(thresholds[int(np.argmax(scores))])


def metric_bundle(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calculate imbalance-aware classification metrics.

    Args:
        y_true: Binary labels.
        probabilities: Positive-class probabilities.
        threshold: Classification threshold.

    Returns:
        Metric dictionary.
    """
    predictions = probabilities >= threshold
    true_positive = int(((predictions == 1) & (y_true == 1)).sum())
    false_positive = int(((predictions == 1) & (y_true == 0)).sum())
    false_negative = int(((predictions == 0) & (y_true == 1)).sum())
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "roc_auc": binary_roc_auc(y_true, probabilities),
        "pr_auc": binary_average_precision(y_true, probabilities),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "brier_score": float(np.mean((probabilities - y_true) ** 2)),
    }


def bootstrap_match_ci(
    predictions: pd.DataFrame,
    resamples: int,
    seed: int,
) -> dict[str, list[float]]:
    """Estimate match-level bootstrap intervals for ROC-AUC and PR-AUC.

    Args:
        predictions: Test predictions with match identifiers.
        resamples: Number of match-level resamples.
        seed: Random seed.

    Returns:
        Two-sided 95% confidence intervals.
    """
    rng = np.random.default_rng(seed)
    match_ids = predictions["match_id"].drop_duplicates().to_numpy()
    roc_values: list[float] = []
    pr_values: list[float] = []
    grouped = {key: value for key, value in predictions.groupby("match_id")}
    for _ in range(resamples):
        sampled = rng.choice(match_ids, size=len(match_ids), replace=True)
        sample = pd.concat([grouped[int(match_id)] for match_id in sampled], ignore_index=True)
        if sample["target"].nunique() < 2:
            continue
        roc_values.append(binary_roc_auc(sample["target"].to_numpy(), sample["probability"].to_numpy()))
        pr_values.append(binary_average_precision(sample["target"].to_numpy(), sample["probability"].to_numpy()))
    return {
        "roc_auc_95_ci": [float(value) for value in np.quantile(roc_values, [0.025, 0.975])],
        "pr_auc_95_ci": [float(value) for value in np.quantile(pr_values, [0.025, 0.975])],
    }


def run_anticipation_benchmark(data: pd.DataFrame, config: PipelineConfig) -> dict[str, Any]:
    """Run a leakage-aware next-three-window benchmark.

    Same-window rare-event counts and label indicators are excluded from the
    feature set. The target uses future windows from the same team-match stream.

    Args:
        data: Corrected primary-window dataset.
        config: Pipeline configuration.

    Returns:
        Benchmark metrics and split metadata.
    """
    ordered = data.sort_values(["match_id", "team_id", "time_window"], kind="stable").copy()
    groups = ordered.groupby(["match_id", "team_id"], sort=False)["primary_rare_event"]
    future_labels = [groups.shift(-offset) for offset in (1, 2, 3)]
    ordered["target"] = pd.concat(future_labels, axis=1).max(axis=1, skipna=True)
    future_counts = ordered.groupby(["match_id", "team_id"], sort=False).cumcount(ascending=False)
    modeling = ordered.loc[future_counts.ge(3)].copy()
    modeling["target"] = modeling["target"].astype(int)

    feature_columns = [
        "event_count",
        "pass_count",
        "avg_pass_length",
        "pass_success_rate",
        "forward_rate",
        "long_pass_rate",
        "delta_pass_count",
        "delta_avg_pass_length",
        "delta_pass_success_rate",
        "delta_forward_rate",
        "delta_long_pass_rate",
        "tactical_shift_score",
        "n_error_events",
        "window_end_minute",
        "period_end",
        "is_partial_window",
        "crosses_period_boundary",
    ]
    train_index, validation_index, test_index = split_by_match(modeling, config.random_seed)
    x = modeling[feature_columns].astype(float)
    y = modeling["target"].to_numpy(dtype=int)
    design, transform_metadata = prepare_design_matrices(x, train_index)
    weights = fit_logistic_regression(
        design[train_index],
        y[train_index],
        config.random_seed,
    )
    validation_logits = np.clip(design[validation_index] @ weights, -35.0, 35.0)
    validation_probabilities = 1.0 / (1.0 + np.exp(-validation_logits))
    threshold = best_f1_threshold(y[validation_index], validation_probabilities)
    test_logits = np.clip(design[test_index] @ weights, -35.0, 35.0)
    test_probabilities = 1.0 / (1.0 + np.exp(-test_logits))
    metrics = metric_bundle(y[test_index], test_probabilities, threshold)
    prediction_frame = modeling.iloc[test_index][["match_id", "team_id", "time_window"]].copy()
    prediction_frame["target"] = y[test_index]
    prediction_frame["probability"] = test_probabilities
    confidence_intervals = bootstrap_match_ci(
        prediction_frame,
        config.bootstrap_resamples,
        config.random_seed,
    )
    return {
        "task": "Anticipate a primary rare event in the next three team windows",
        "model": "Class-balanced logistic regression",
        "feature_policy": "Same-window rare labels and component counts excluded",
        "feature_columns": feature_columns,
        "threshold_selected_on_validation": threshold,
        "train_rows": int(len(train_index)),
        "validation_rows": int(len(validation_index)),
        "test_rows": int(len(test_index)),
        "train_matches": int(modeling.iloc[train_index]["match_id"].nunique()),
        "validation_matches": int(modeling.iloc[validation_index]["match_id"].nunique()),
        "test_matches": int(modeling.iloc[test_index]["match_id"].nunique()),
        "test_prevalence": float(y[test_index].mean()),
        "preprocessing": transform_metadata,
        **metrics,
        **confidence_intervals,
    }


def dataset_summary(
    data: pd.DataFrame,
    records: Sequence[SourceRecord],
    window_size: int,
) -> dict[str, Any]:
    """Create the manuscript summary for a window dataset.

    Args:
        data: Team-window dataframe.
        records: Match metadata.
        window_size: Event-count window size.

    Returns:
        Summary dictionary.
    """
    barcelona_match_ids = {
        record.match_id
        for record in records
        if "barcelona" in record.home_team.lower() or "barcelona" in record.away_team.lower()
    }
    barcelona_team_mask = data["team"].str.contains("Barcelona", case=False, na=False)
    barcelona_match_mask = data["match_id"].isin(barcelona_match_ids)
    primary = data["primary_rare_event"].astype(bool)
    broad = data["broad_rare_event"].astype(bool)
    error_only = data["is_error_cluster"].astype(bool) & ~primary
    return {
        "window_size": window_size,
        "rows": int(len(data)),
        "columns": int(data.shape[1]),
        "matches": int(data["match_id"].nunique()),
        "teams": int(data["team_id"].nunique()),
        "primary_rare_windows": int(primary.sum()),
        "primary_rare_rate": float(primary.mean()),
        "broad_rare_windows": int(broad.sum()),
        "broad_rare_rate": float(broad.mean()),
        "error_cluster_windows": int(data["is_error_cluster"].sum()),
        "error_cluster_only_windows": int(error_only.sum()),
        "zero_pass_windows": int(data["pass_count"].eq(0).sum()),
        "partial_windows": int(data["is_partial_window"].sum()),
        "own_goal_for_events": int(data["n_own_goals_for"].sum()),
        "own_goal_against_events": int(data["n_own_goals_against"].sum()),
        "formation_change_windows": int(data["has_formation_change"].sum()),
        "formation_change_time_window_zero": int(
            (data["has_formation_change"] & data["time_window"].eq(0)).sum()
        ),
        "starting_xi_events_in_formation_count": 0,
        "barcelona_matches": int(len(barcelona_match_ids)),
        "barcelona_match_share": float(len(barcelona_match_ids) / len(records)),
        "windows_from_matches_involving_barcelona": int(barcelona_match_mask.sum()),
        "window_share_from_matches_involving_barcelona": float(barcelona_match_mask.mean()),
        "barcelona_team_windows": int(barcelona_team_mask.sum()),
        "barcelona_team_window_share": float(barcelona_team_mask.mean()),
    }


def write_outputs(
    primary_data: pd.DataFrame,
    sensitivity: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write corrected data and manuscript statistics.

    Args:
        primary_data: Corrected 50-event dataset.
        sensitivity: Window-size sensitivity table.
        summary: Full analysis summary.
        output_dir: Destination directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_data.to_csv(output_dir / "football_team_window_dataset_v2.csv", index=False)
    try:
        primary_data.to_parquet(output_dir / "football_team_window_dataset_v2.parquet", index=False)
    except (ImportError, ValueError) as exc:
        LOGGER.warning("Parquet export skipped: %s", exc)
    sensitivity.to_csv(output_dir / "window_size_sensitivity.csv", index=False)
    (output_dir / "editor_revision_statistics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the complete editor-revision pipeline."""
    configure_logging()
    args = parse_args()
    config = PipelineConfig.from_json(args.config)
    records = collect_source_records(config, args.cache_dir)
    LOGGER.info("Selected %d matches across %d seasons", len(records), len({r.season_id for r in records}))
    download_event_files(records, config, args.cache_dir)
    if config.use_cached_events_only:
        available_records = [
            record
            for record in records
            if (args.cache_dir / "events" / f"{record.match_id}.json").exists()
        ]
        LOGGER.info(
            "Using %d/%d matches with locally available event files",
            len(available_records),
            len(records),
        )
        records = available_records
        if not records:
            raise ValueError("No cached event files matched the selected competition")

    summaries: list[dict[str, Any]] = []
    primary_data: pd.DataFrame | None = None
    for window_size in config.window_sizes:
        cache_path = args.output_dir / f".window_{window_size}.pkl"
        if cache_path.exists():
            LOGGER.info("Loading cached window dataset %s", cache_path)
            data = pd.read_pickle(cache_path)
        else:
            data = build_window_dataset(records, config, args.cache_dir, window_size)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            data.to_pickle(cache_path)
        if window_size == config.primary_window_size:
            primary_data = add_delta_features(data)
            summary_data = primary_data
        else:
            summary_data = data
        summaries.append(dataset_summary(summary_data, records, window_size))

    if primary_data is None:
        raise ValueError("Primary window size was not included in window_sizes")

    sensitivity = pd.DataFrame(summaries)[
        [
            "window_size",
            "rows",
            "primary_rare_windows",
            "primary_rare_rate",
            "broad_rare_windows",
            "broad_rare_rate",
            "error_cluster_windows",
        ]
    ]
    validation = validation_summary(records, config, args.cache_dir)
    benchmark = run_anticipation_benchmark(primary_data, config)
    primary_summary = next(
        item for item in summaries if item["window_size"] == config.primary_window_size
    )
    season_substitution = (
        primary_data.groupby("season", sort=False)
        .agg(
            windows=("match_id", "size"),
            substitution_burst_windows=(
                "n_subs",
                lambda values: int((values >= config.substitution_burst_threshold).sum()),
            ),
        )
        .reset_index()
    )
    season_substitution["substitution_burst_rate"] = (
        season_substitution["substitution_burst_windows"] / season_substitution["windows"]
    )
    summary = {
        "config": asdict(config),
        "source": {
            "provider": "StatsBomb Open Data",
            "source_url": "https://github.com/statsbomb/open-data",
            "downloaded_match_count": len(records),
            "season_count": len({record.season_id for record in records}),
        },
        "primary_dataset": primary_summary,
        "window_sensitivity": summaries,
        "goal_validation": validation,
        "anticipation_benchmark": benchmark,
        "substitution_burst_by_season": season_substitution.to_dict(orient="records"),
    }
    write_outputs(primary_data, sensitivity, summary, args.output_dir)
    LOGGER.info("Revision outputs written to %s", args.output_dir)


if __name__ == "__main__":
    main()
