# AutoML Platform Suite

> AutoML & ML Platforms portfolio project — independent open-source implementation.
> This is an original, from-scratch build. It is not affiliated with, and does not
> contain any code, prompts, data, or business logic from, any employer or client.

![status](https://img.shields.io/badge/status-planned-lightgrey)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## Combines

This is a flagship suite that will combine 3 related AutoML/ML-platform repos
into one Django (or FastAPI) web app with all 3 features selectable from a
single dashboard UI — the same combined-suite pattern already used in this
portfolio for `video-analytics-suite`, `medical-imaging-suite`, and
`trading-ai-suite`:

- [`automl-web-platform`](../automl-web-platform/) — upload a dataset, auto-preprocess, train multiple models, compare metrics, download the best model
- [`automl-experiment-framework`](../automl-experiment-framework/) — search-space-driven experimentation with Optuna hyperparameter tuning and MLflow tracking, leaderboard of runs
- [`ml-evaluation-dashboard`](../ml-evaluation-dashboard/) — unified metrics schema and dashboard across classification/regression/CV/LLM model types

The 3 originals will get an archived banner + `status-archived` badge and move
to `E:\Projects\portfolio-archived-repos\` once this suite reaches feature
parity with each of them (matching how the video/medical/trading originals
were handled) — no code or git history is deleted, only relocated.

## 1. Problem

Each of the 3 source repos solves one slice of "get a dataset to a trained,
evaluated model" (train it, tune it, or judge it) but started as an isolated
scaffold with its own UI plan. Anyone using this portfolio's AutoML tooling
has to context-switch between 3 separate apps to upload a dataset, run a
tuned search, and then read a dashboard of the result — for what is really
one linear workflow.

## 2. Architecture

```text
Dashboard (pick a feature) -> Upload Dataset -> [Quick multi-model train | Search-space + Optuna tuning]
    -> Job (Celery/background) -> Tracked run (MLflow) -> Unified Metrics Schema -> Evaluation Dashboard / Leaderboard
```

A shared `automl_core` layer (data upload/validation, the unified metrics
schema, a run/job registry) sits underneath 3 feature apps — one per source
repo — registered the same way `medical-imaging-suite`'s
`BaseImagingTask` / `@register_task` pattern registers its 5 imaging tasks, so
adding a 4th AutoML feature later is a registry entry, not a rewrite.

## 3. Technology Stack

- Python, Django 5.x (feature-picker web app + job/run history, matching the
  other suites), Celery + Redis for background training/tuning jobs
- scikit-learn, LightGBM (multi-model training)
- Optuna (hyperparameter search), MLflow (experiment tracking)
- Pandas, Plotly/Matplotlib (metrics dashboard)
- PostgreSQL in production, SQLite for local/dev (`DATABASE_URL` override)

## 4. Feature List

- **Train** (from `automl-web-platform`): dataset upload, target column
  selection, task type (classification/regression), automated preprocessing,
  multi-model training, comparison table, trained-model download
- **Tune** (from `automl-experiment-framework`): YAML-driven preprocessing +
  model search spaces, Optuna hyperparameter optimization, MLflow-tracked
  runs, leaderboard
- **Evaluate** (from `ml-evaluation-dashboard`): unified metrics schema
  ingestion adapters for classification/regression/CV/LLM result types, one
  dashboard across all of them
- Shared: per-user job history, background job execution, one dataset upload
  path reused across all 3 features

## 5. Implementation Plan

1. Phase 1: `automl_core` shared app (dataset upload/validation, job model,
   unified metrics schema) + Django project skeleton with the dashboard shell
2. Phase 2: Port `automl-web-platform`'s upload → preprocess → multi-model
   train → compare flow into a feature app on top of `automl_core`
3. Phase 3: Port `automl-experiment-framework`'s search-space + Optuna +
   MLflow flow into a second feature app
4. Phase 4: Port `ml-evaluation-dashboard`'s ingestion adapters + dashboard
   views into a third feature app, wired to read both other features' runs
5. Phase 5: Archive the 3 original repos (banner + badge, move to
   `portfolio-archived-repos`) once parity is confirmed

## Task Tracking

Work will be broken into phase-tagged user stories tracked as GitHub Issues,
not in this file. Implement Phase 1 issues first (later phases depend on it).
When you start one, add label `status:in-progress`. When you finish, close it
referencing the commit (e.g. `git commit -m "... Closes #4"`) and push.

## 6. Repository Structure

```text
automl-platform-suite/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── .env.example
├── docker/
├── docs/
│   ├── architecture.md
│   └── evaluation.md
├── src/
├── tests/
├── configs/
├── scripts/
├── notebooks/
├── examples/
├── assets/
└── .github/
    └── workflows/
```

## 7. Setup

```bash
git clone <this-repo-url>
cd automl-platform-suite
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt   # or: pip install -e .
cp .env.example .env              # fill in API keys / config
```

## 8. Dataset

Document which public dataset(s) or synthetic data generators are used here.
No proprietary, employer-owned, or client-identifiable data is used in this
project.

## 9. Training / Execution

```bash
# Once Phase 1 lands:
python manage.py migrate
python manage.py runserver   # open http://127.0.0.1:8000/ and pick a feature
```

## 10. Evaluation

Document evaluation metrics and how to reproduce them here (see
`docs/evaluation.md`).

## 11. Results

_To be filled in as the implementation progresses — screenshots, metrics
tables, and sample outputs go here._

## 12. API

_If this project exposes an API, document the main endpoints here (or link to
auto-generated OpenAPI docs, e.g. `/docs` for FastAPI)._

## 13. Docker

```bash
docker build -t automl-platform-suite .
docker run -p 8000:8000 automl-platform-suite
```

## 14. Tests

```bash
pytest tests/
```

## 15. Limitations

- This is a from-scratch, independent recreation built for portfolio purposes.
- Performance numbers, once added, are based on public datasets and are not
  representative of any production system's real-world results.
- Scaffold stage: no code has been ported from the 3 source repos yet — see
  §5 Implementation Plan.

## 16. Future Work

- Port each source repo's existing Phase-1 code (where present) rather than
  rewriting from scratch.
- Expand evaluation coverage and add CI-based regression checks.
- Track open items as GitHub Issues.

## 17. Disclosure

This repository is an **independent open-source recreation inspired by the
kind of production systems I have worked on professionally**. It contains no
employer or client source code, prompts, datasets, credentials, architecture
diagrams, or business logic. All code, data, and documentation here are
original or built on publicly available datasets and open-source tools.

---
_Last updated: 2026-09-12_
