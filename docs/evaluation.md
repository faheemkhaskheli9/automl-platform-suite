# Evaluation Notes: AutoML Platform Suite

## Metrics

- **Train** feature: accuracy/F1 (classification) or MAE/RMSE (regression) per
  candidate model, comparison table
- **Tune** feature: best objective value found, number of trials, trial
  convergence curve (from MLflow-tracked runs)
- **Evaluate** feature: unified metric records across classification,
  regression, CV, and LLM-judge scores — one schema, per `ml-evaluation-dashboard`'s
  Phase-1 design

## Reproducing Results

```bash
python -m src.evaluate --config configs/eval.yaml
```

## Result Log

| Date | Config | Metric | Value | Notes |
|------|--------|--------|-------|-------|
|      |        |        |       |       |
