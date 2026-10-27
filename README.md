# Football Rare Event Revision Code

This repository contains the reproducible processing and audit code accompanying the revised study **A Source Audited Team Window Dataset for Rare Event Analysis in Football**.

## Public research materials

- Dataset and associated study materials: https://doi.org/10.17632/yjrh9kfdbp.2
- StatsBomb Open Data source: https://github.com/hudl/open-data
- Metrica Sports sample data: https://github.com/metrica-sports/sample-data
- SkillCorner Open Data: https://github.com/SkillCorner/opendata
- Roboflow Sports components: https://github.com/roboflow/sports

External materials remain subject to the respective providers' usage terms.

## Repository contents

- `src/rare_event_revision_pipeline.py`: downloads the official StatsBomb event files, constructs leakage-aware team windows, calculates derived features, and writes the corrected analysis outputs.
- `src/reviewer_r2_audit.py`: reproduces the sensitivity, validation, benchmark, calibration, split, and data-dictionary audits used in the revision.
- `src/coach_agent_pipeline/`: builds the tactical-response enrichment tables, coach-reaction outputs, reward/preference seeds, reward-guided policy selection, supervisor-approved model artifacts, overfitting audits, video presentation outputs, and final validation reports.
- `config/reviewer_r2_config.json`: deterministic pipeline configuration and random seed.
- `docs/DATA_DICTIONARY.md`: source dictionary used to validate and expand the exported 56-column schema.

## Environment

Python 3.11 or later is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Run the processing pipeline

```bash
python src/rare_event_revision_pipeline.py \
  --config config/reviewer_r2_config.json \
  --cache-dir .cache/statsbomb \
  --output-dir output/pipeline
```

The command downloads source event JSON files from the URL declared in the configuration unless valid cached files are already present.

## Run the revision audit

```bash
python src/reviewer_r2_audit.py \
  --data-dir output/pipeline \
  --output-dir output/reviewer_r2_audit \
  --config config/reviewer_r2_config.json \
  --dictionary docs/DATA_DICTIONARY.md
```

No independent multi-expert annotation labels were collected. Consequently, this repository does not include a separate multi-rater annotation dataset or inter-rater agreement statistics.

The scripts whose names contain `simulated` generate synthetic review or preference artifacts for pipeline testing. Those outputs are not independent human-expert annotations and must not be interpreted as inter-rater validation.

## Coach-agent and reward-guided pipeline

The coach-agent scripts are retained in their reproducible execution order under `src/coach_agent_pipeline`. They read and write the local `data`, `models`, and `outputs` directories beneath that folder. Set `COACH_REACTION_INPUT_PATH` to the deposited Excel source before running the enrichment stage. Set `COACH_AGENT_INPUT_VIDEO` when generating the optional video demonstration.

```bash
python src/coach_agent_pipeline/build_coach_reaction_dataset.py
python src/coach_agent_pipeline/build_expert_review_sample.py
python src/coach_agent_pipeline/build_rlhf_preference_seed.py
python src/coach_agent_pipeline/train_reward_ranker_seed.py
python src/coach_agent_pipeline/train_simulated_coach_agent_model.py
python src/coach_agent_pipeline/simulate_expert_review_and_reward.py
python src/coach_agent_pipeline/run_reward_guided_policy_selector.py
python src/coach_agent_pipeline/train_supervisor_approved_coach_agent_model.py
python src/coach_agent_pipeline/audit_coach_agent_overfitting.py
python src/coach_agent_pipeline/validate_phase4_outputs.py
```

## Reproducibility

The configuration fixes the random seed at `20260908`. Generated data, caches, and audit outputs are ignored by Git and should be reproduced locally from the public source data and cited research deposit.
