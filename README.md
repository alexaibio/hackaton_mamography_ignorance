# Mammography ignorance analysis toolset

## Installing package with UV

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Python >= 3.10

### Setup

```bash
git clone <repo-url>
cd hackaton_mamography_ignorance

# create .venv and install the project with all dependencies
uv sync

# copy the environment template, then fill in the values
cp .env_example .env
```


### Environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `LOG_LEVEL` | no | Defaults to `INFO` |
| `LOG_DIR` | yes | Where `logger.py` writes rotating log files |
| `DATA_FOLDER` | yes | Root of `raw/`, `interim/`, `processed/` |
| `MODELS_FOLDER` | yes | Where trained CatBoost models are saved |
| `FHIR_BASE_URL` | for `fetch` | Base URL of the FHIR server |
| `FHIR_TOKEN` | for `fetch` | Bearer token |


### Running commands

You can run Python with two options:

- through `uv run` — calling `python` directly bypasses the virtual
environment:

```bash
uv run list-risk-patients --top 50
uv run jupyter lab
uv run pytest
```
- through activation environment with `source .venv/bin/activate `


## Installing a test FHIR clinical dataset Synthea
Do it outside the main project folder

```
  git clone https://github.com/synthetichealth/synthea
  cd synthea
  ./run_synthea -p 50 Massachusetts
```



## The flow

First, provide access to FHIR API, 
- either by running a test instance in docker `docker compose -f docker/compose.yaml up -d`
- or by providing .env credentials `FHIR_BASE_UR`

You can test it either by
- `curl -s http://localhost:8080/fhir/metadata | jq '.software.version'`
- `curl -s "$FHIR_BASE_URL/.well-known/smart-configuration" | jq`

Then each command can be ran as
- `uv run mammo fetch` or
- or after `source .venv/bin/activate` you can do `mammo fetch`

Each stage is a CLI subcommand

```text
mammo fetch --force             →  data/raw/*.ndjson
mammo flatten                   →  data/interim/{patients,comms,appointments}.csv
        ↓
   notebook 01_explore.ipynb
        ↓
mammo build_train_set  →  data/processed/invitations.csv
mammo train            →  models/model.cbm + models/metrics.json
        ↓
   notebook 02_model.ipynb          ← calibration, lift, SHAP plots
        ↓
mammo segment          →  data/processed/segments.csv + reports/segments.md
```