# Skill: End-to-End Verification

Load when: verifying the full pipeline works end-to-end.

## E2E test scripts

```bash
# Full pipeline with synthetic data (no Kaggle dependency)
cd backend && python3 tests/integration/run_e2e_tests.py
# Takes 8-10 minutes, runs complete pipeline to notebook generation

# Kaggle e2e (requires credentials)
cd backend && python3 run_kaggle_e2e.py
```

## What to verify

### Pipeline completes
- Job status reaches COMPLETED (not FAILED or stuck)
- Notebook file is generated at state.notebook_path
- No unhandled exceptions in logs

### Agent outputs populated
- state.data_profile is not None
- state.treatment_effects has at least 1 result
- state.sensitivity_results has at least 1 result
- state.critique_history has at least 1 entry

### Notebook is valid
- File is valid JSON (nbformat)
- Contains executable code cells
- At least 10 cells generated
- Introduction and conclusions sections present

### API endpoints work
```bash
# Create job
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{"kaggle_url": "...", "treatment_variable": "...", "outcome_variable": "..."}'

# Check status
curl http://localhost:8000/jobs/{job_id}/status

# Download notebook
curl -O http://localhost:8000/jobs/{job_id}/notebook
```

## Known issues to watch

1. Critique agent may auto-finalize without converging
2. Pipeline may take >10 minutes on large datasets
3. Some causal methods may fail on specific data shapes (this is expected)
4. Notebook cells may reference variables from failed agents (cells should handle None)

## Output format

```
## E2E Verification — [date]

Pipeline: COMPLETED / FAILED at [stage]
Duration: Xs
Agents completed: X/13
Treatment effects: X methods ran
Sensitivity checks: X ran
Notebook: generated / missing
Notebook cells: X total
```
