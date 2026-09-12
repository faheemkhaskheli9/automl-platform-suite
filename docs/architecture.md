# Architecture Notes: AutoML Platform Suite

## Pipeline

```text
Dashboard (pick a feature) -> Upload Dataset -> [Quick multi-model train | Search-space + Optuna tuning]
    -> Job (Celery/background) -> Tracked run (MLflow) -> Unified Metrics Schema -> Evaluation Dashboard / Leaderboard
```

## Components

- `automl_core` — shared dataset upload/validation, job model, unified
  metrics schema (reused from `ml-evaluation-dashboard`'s Phase-1 design)
- `train` feature app — ported from `automl-web-platform`: preprocessing,
  multi-model training, comparison, model export
- `tune` feature app — ported from `automl-experiment-framework`: YAML search
  spaces, Optuna HPO loop, MLflow tracking
- `evaluate` feature app — ported from `ml-evaluation-dashboard`: ingestion
  adapters per model type, unified dashboard reading both other apps' runs
- `jobs` — per-user job/run history, background execution (Celery + Redis).
  Implemented as `automl_core.models.Job` + `automl_core.tasks.submit_job`
  (Phase 1): a feature app calls `submit_job(owner=..., feature=...,
  dotted_path=...)` and gets a `Job` row back; the shared `run_job` Celery
  task drives `queued -> running -> succeeded/failed` and always re-raises
  on failure so a broken task is never silently marked anything but failed.
  This sandbox has no Redis broker, so `CELERY_TASK_ALWAYS_EAGER=1` (the
  default here) runs `.delay()` synchronously in-process against the same
  task code a real worker would run — flip it off with a real
  `CELERY_BROKER_URL` for an actual deployment.

## Design Notes

- Registry pattern for feature apps (mirrors `medical-imaging-suite`'s
  `BaseImagingTask` / `@register_task`) so a 4th AutoML feature is a registry
  entry, not a rewrite.
- Keep provider/model choices swappable behind interfaces (see
  `multi-llm-router` for the general pattern).
- Prefer configuration-driven pipelines (YAML/JSON in `configs/`) over
  hardcoded parameters so experiments stay reproducible.
- Port `automl-experiment-framework`'s existing `src/search_space.py` /
  `src/model_search_space.py` / `pipeline_builder.py` rather than
  re-implementing the search-space design.
