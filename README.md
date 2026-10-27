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

## Reproducibility

The configuration fixes the random seed at `20260908`. Generated data, caches, and audit outputs are ignored by Git and should be reproduced locally from the public source data and cited research deposit.
