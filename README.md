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


## Test data
### FHIR local docker 
First, provide access to FHIR API, 
- either by running a test instance in docker `docker compose -f docker/compose.yaml up -d`
- or by providing .env credentials `FHIR_BASE_UR`

You can test it either by
- `curl -s http://localhost:8080/fhir/metadata | jq '.software.version'`
- `curl -s "$FHIR_BASE_URL/.well-known/smart-configuration" | jq`


### Installing a test FHIR clinical dataset Synthea
Do it outside the main project folder

Use the pre-generated sample data — no Java, no Gradle, no build step.

```bash
  # 1. download the FHIR R4 sample data (~90 MB zipped) and unpack
  curl -L -o synthea_r4.zip \
    https://raw.githubusercontent.com/synthetichealth/synthea-sample-data/main/downloads/synthea_sample_data_fhir_r4_nov2021.zip
  unzip -q synthea_r4.zip -d synthea_r4

  # 2. locate the bundles (the zip layout varies) and check their type
  BUNDLES=$(dirname "$(find synthea_r4 -name '*.json' | head -1)")
  jq -r '.type' "$BUNDLES"/*.json | sort -u
```

The bundles are `batch` and `transaction`, so they are POSTable as they are.
Upload to the local FHIR server and verify:

```bash
  # 3. upload
  for f in "$BUNDLES"/*.json; do
    curl -s -X POST http://localhost:8080/fhir \
      -H "Content-Type: application/fhir+json" --data-binary @"$f" > /dev/null
  done

  # 4. count what landed
  curl -s "http://localhost:8080/fhir/Patient?_summary=count" | jq '.total'
```

Then pull it into the project with `mammo fetch --force`.

Two limits of this dataset: the geography is US (Massachusetts), not Czech, and it
contains no `Communication` resources — no invitations, channels or reminders. The
invitation layer has to be generated separately.



## The flow



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


## The UX for end customer
It is done with streamlit app.
