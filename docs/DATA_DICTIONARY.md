# Data dictionary

Each row represents one team in one match during a fixed 50-event team stream window. Empty numeric pass aggregates and deltas are intentional missing values, not zeros.

| Field | Type | Definition |
|---|---|---|
| `competition_id` | integer | StatsBomb competition identifier. |
| `competition` | string | Competition name. |
| `season_id` | integer | StatsBomb season identifier. |
| `season` | string | Season label. |
| `match_id` | integer | StatsBomb match identifier. |
| `match_date` | date | Match date from match metadata. |
| `team_id` | integer | Focal-team identifier. |
| `team` | string | Focal-team name. |
| `opponent_id` | integer | Opponent identifier. |
| `opponent` | string | Opponent name. |
| `home_away` | category | `home` or `away` for the focal team. |
| `time_window` | integer | Zero-based event-window index within the team-match stream. |
| `event_count` | integer | Number of focal-team events in the window. |
| `is_partial_window` | boolean | True when the final window contains fewer than 50 events. |
| `start_event_index` | integer | StatsBomb event index at the start of the window. |
| `end_event_index` | integer | StatsBomb event index at the end of the window. |
| `period_start` | integer | Match period of the first event. |
| `period_end` | integer | Match period of the last event. |
| `window_start_minute` | integer | Recorded minute of the first event. |
| `window_end_minute` | integer | Recorded minute of the last event. |
| `window_start_second` | integer | Recorded second of the first event. |
| `window_end_second` | integer | Recorded second of the last event. |
| `crosses_period_boundary` | boolean | True when the window spans more than one match period. |
| `pass_count` | integer | Number of passes in the window. |
| `has_pass_data` | boolean | True when at least one pass is available for aggregate computation. |
| `avg_pass_length` | float, nullable | Mean Euclidean pass length in StatsBomb pitch units. |
| `pass_success_rate` | float, nullable | Share of passes with a missing pass-outcome field. |
| `forward_rate` | float, nullable | Share of passes whose end x-coordinate exceeds the start x-coordinate. |
| `long_pass_rate` | float, nullable | Share of passes longer than 30 pitch units. |
| `n_goals` | integer | Focal-team goals, including Own Goal For events. |
| `n_own_goals_for` | integer | Own Goal For events in the focal-team stream. |
| `n_own_goals_against` | integer | Own Goal Against events in the focal-team stream. |
| `n_own_goals` | integer | Sum of Own Goal For and Own Goal Against records. |
| `n_red_cards` | integer | Straight red cards found in supported card fields. |
| `n_second_yellows` | integer | Second-yellow cards found in supported card fields. |
| `n_subs` | integer | Substitution events. |
| `n_injuries` | integer | Injury stoppage events. |
| `n_unexpected_goals` | integer | Goals with StatsBomb shot xG below 0.1. |
| `n_penalty_goals` | integer | Goals from shots whose type is Penalty. |
| `n_error_events` | integer | Sum of Error, Miscontrol, and Dispossessed events. |
| `n_formation_changes` | integer | Tactical Shift events; Starting XI is excluded. |
| `n_starting_xi_events` | integer | Starting XI records retained for audit only. |
| `has_formation_change` | boolean | True when at least one Tactical Shift occurs. |
| `is_error_cluster` | boolean | True when at least two error-type events occur. |
| `primary_rare_event` | boolean | Strict label excluding Error Cluster. |
| `primary_rare_event_reason` | string, nullable | Semicolon-separated strict-label components. |
| `broad_rare_event` | boolean | Primary label OR Error Cluster. |
| `broad_rare_event_reason` | string, nullable | Semicolon-separated broad-label components. |
| `has_previous_window` | boolean | True when a preceding window exists in the same team-match stream. |
| `delta_pass_count` | float, nullable | Absolute pass-count change from the preceding team-match window. |
| `delta_avg_pass_length` | float, nullable | Absolute mean-pass-length change from the preceding window. |
| `delta_pass_success_rate` | float, nullable | Absolute pass-success-rate change from the preceding window. |
| `delta_forward_rate` | float, nullable | Absolute forward-pass-rate change from the preceding window. |
| `delta_long_pass_rate` | float, nullable | Absolute long-pass-rate change from the preceding window. |
| `delta_available` | boolean | True when all required current and preceding pass aggregates exist. |
| `tactical_shift_score` | float, nullable | Sum of the five absolute delta features. |
