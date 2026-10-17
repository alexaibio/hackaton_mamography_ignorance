# Mammography ignorance analysis toolset

## Intro
This package provides a set of tools to analyse VZP insurance company data about patients
who were invited for a mammography screening but did not show up.

The tools are presented as a simple web interface (streamlit) with the following pages:
- `Home` page - general statistics: how many invitations we have, how they are distributed across locations, age bands and so on
- `Risk model` page - to run the model training and to analyse which of its features are strong predictors
- `Patient risk` page - to calculate the risk of not reacting to an invitation for a particular patient profile
- `Interventions` page - a list of patients who did not attend, with proposed interventions to improve attendance. Emails are generated as part of the interventions
- `Refusal profile` page - a portrait of the women who never attend: for the most important features of the trained model, the typical value of each one, plus a radar chart of the group against the whole population
- `Responder profile` page - the mirror image, the women who do attend reliably. This is the behaviour an intervention is trying to produce, so the same features are shown the same way
- `Refusal/responder comparison` page - one radar chart putting the two groups directly against each other. Every patient is in exactly one of them: never screened after any invitation, or screened after at least one


## AI components

- **Risk model** — a gradient boosted decision tree (CatBoost) predicting whether a
  screening is billed within the chosen horizon after an invitation. Risk of ignoring is
  one minus that probability
- **Three parallel models, one per horizon** (`3m`, `6m`, `12m`). The differences between
  them segment a patient by *response speed*: a prompt responder, a late responder who
  attends but needs a timed reminder, or someone unlikely at any horizon for whom a
  reminder will not work and the route has to be her doctor. Two women with the same
  12-month risk can need different interventions
- **Feature importance** (`PredictionValuesChange` and `LossFunctionChange`) to see what
  the model leans on across the cohort, and which features carry nothing and can be
  dropped
- **SHAP values for per-patient explanation** — the top drivers for one specific woman,
  with the direction of each contribution. This is what makes an individual intervention
  possible rather than a generic campaign
- **Counterfactual search over actionable features** — a greedy optimiser scores modified
  copies of a patient's record and returns the sequence of changes that lowers predicted
  risk most, with the risk after each step. The search is restricted to features an
  intervention can realistically move; age, region, insurance tenure and screening
  history are held fixed
- **Model-driven triage** — the whole refusal cohort is scored and ranked, so the output
  is an ordered worklist. The decision threshold converts risk into a contact list of
  whatever size outreach can handle
- **LLM (OpenAI API)** — three personalised reminders for the patient, escalating in
  concreteness rather than in pressure, plus one email each to her gynaecologist and
  general practitioner. The whole patient record is given to the model so the text can be
  specific


## Technical implementation details
- The data is provided by the VZP insurance company and should be placed in the `data/raw_csv` folder
- All functionality is available as a web application (streamlit) - `app/`
- AI: the risk model is a decision tree trained on 15 features; two features are excluded because they are redundant: `invitation_sequence`, `months_since_last_screen`
  - the model has three targets, `screening 12m`, `6m` and `3m`, in order to provide a time perspective
  - feature importance is calculated in order to understand which features are the most important and to propose an intervention
- LLM features: based on the most important features, three personalised reminder emails are composed, as well as an email to the gynaecologist and the general practitioner, in order to facilitate attendance. An OpenAI API token must be entered
- In order to compare refusal and non-refusal patient profiles, a radar plot is composed


## Installing with UV

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



### Running commands

You can run Python in two ways:

- through `uv run` — calling `python` directly bypasses the virtual
environment:

```bash
uv run mammo csv_import
uv run jupyter lab
```
- or by activating the environment with `source .venv/bin/activate`


## The UX for the end customer
It is done with a streamlit app.

UX: `uv run streamlit run app/Home.py`

Or in docker, with no local python needed: `docker compose -f docker/docker-compose-streamlit.yaml up --build`

It opens on http://localhost:8501.




# Leftover functionality
Some functionality is not used in this application.
Initially the data was supposed to come via a FHIR API, so a docker with a test FHIR server was created.
The data now comes from a csv file, so it is not used.

## Test data
### FHIR local docker
First, provide access to a FHIR API,
- either by running a test instance in docker `docker compose -f docker/docker-compose-fhir.yaml up -d`
- or by providing .env credentials `FHIR_BASE_URL`

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



### The flow

Each command can be run as
- `uv run mammo fetch`, or
- after `source .venv/bin/activate` you can do `mammo fetch`

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



