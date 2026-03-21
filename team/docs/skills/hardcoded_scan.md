# Skill: Hardcoded Values Scan

Load when: finding magic numbers, strings, and thresholds that should be in config.

## What counts as hardcoded

### Magic numbers
```python
if n_samples < 30:          # Why 30? Should be configurable
timeout = 300                # Should be in settings
max_retries = 3              # Should be in settings
```

### Magic strings
```python
model = "gemini-2.0-flash"   # Should come from settings
provider = "vertex"           # Should come from settings
```

### Threshold values in agent logic
```python
if corr > 0.05:              # Confounder threshold — should be configurable
if p_value < 0.05:           # Significance level — should be configurable
if overlap < 0.5:            # PS overlap threshold — should be configurable
```

## What is NOT hardcoded (don't flag)

- Constants that are definitional (pi, e, standard z-scores like 1.96)
- JSON schema definitions (these ARE the spec, not config)
- Tool names and descriptions (these ARE the interface)
- Status enum values (COMPLETED, FAILED — these are the protocol)
- Log event names ("agent_completed" — these are observability keys)
- Test assertions (expected values in tests are correct)

## Where extracted values should go

All extracted values go to: backend/src/config/settings.py as Pydantic fields with env var overrides.

```python
# In settings.py
confounder_correlation_threshold: float = 0.03
significance_level: float = 0.05
ps_overlap_critical_threshold: float = 0.5
max_trace_output_len: int = 1000
max_traces: int = 50
```

## Output format

```markdown
## Hardcoded Values Report

| File | Line | Value | Context | Recommended Config Name |
|------|------|-------|---------|------------------------|
| sensitivity_analyst.py | 467 | 0.05 | E-value SE threshold | `evalue_se_threshold` |
| ps_diagnostics.py | 532 | 0.5 | Overlap critical | `ps_overlap_critical` |
```
