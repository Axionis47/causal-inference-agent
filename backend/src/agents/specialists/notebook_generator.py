"""Notebook Generator Agent - Creates reproducible Jupyter notebooks.

Generates notebooks that faithfully report the work of each pipeline agent:
Data Profiler → Domain Knowledge → Data Repair → EDA → Causal Discovery
→ PS Diagnostics → Effect Estimator → Sensitivity Analyst → Critique Agent.

Each section presents the agent's actual findings from AnalysisState,
with visualizations built from pipeline-computed data (not recomputed).
"""

import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from src.agents.base import AnalysisState, BaseAgent, JobStatus
from src.agents.registry import register_agent
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


@register_agent("notebook_generator")
class NotebookGeneratorAgent(BaseAgent):
    """Agent that generates reproducible Jupyter notebooks reporting pipeline results.

    Each notebook section maps 1:1 to a pipeline agent and presents
    that agent's actual outputs, reasoning, and findings.
    """

    AGENT_NAME = "notebook_generator"

    SYSTEM_PROMPT = """You are an expert at creating clear, reproducible Jupyter notebooks
for causal inference analysis.

When generating narratives:
1. Be specific to the dataset - never write generic placeholder text
2. Explain the choice of methods in context of the data characteristics
3. Present results with proper interpretation and caveats
4. Discuss limitations honestly
5. Write in clear academic style accessible to data scientists"""

    TOOLS = []  # No tools needed - uses LLM for narrative only

    # ─────────────────────────── Main execute ───────────────────────────

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Generate the analysis notebook reporting all pipeline findings."""
        self.logger.info("notebook_generation_start", job_id=state.job_id)
        state.status = JobStatus.GENERATING_NOTEBOOK
        start_time = time.time()

        try:
            nb = new_notebook()
            nb.metadata = {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.11"},
            }

            cells: list = []

            # 1. Title & Introduction
            cells.extend(await self._generate_introduction(state))

            # 2. Domain Knowledge (if agent ran)
            if state.domain_knowledge:
                cells.extend(self._generate_domain_knowledge(state))

            # 3. Setup & Imports
            cells.extend(self._generate_setup_cells())

            # 4. Data Loading
            cells.extend(self._generate_data_loading(state))

            # 5. Data Profile (profiler agent findings)
            if state.data_profile:
                cells.extend(self._generate_data_profile_report(state))

            # 6. Data Repairs (data repair agent)
            if state.data_repairs:
                cells.extend(self._generate_data_repairs(state))

            # 7. EDA (EDA agent findings)
            cells.extend(self._generate_eda_report(state))

            # 8. Causal Structure (discovery agent)
            if state.proposed_dag:
                cells.extend(self._generate_causal_structure(state))

            # 9. Confounder Analysis
            if state.confounder_discovery or (
                state.data_profile and state.data_profile.potential_confounders
            ):
                cells.extend(self._generate_confounder_analysis(state))

            # 10. PS Diagnostics (propensity score diagnostics agent)
            if state.ps_diagnostics:
                cells.extend(self._generate_ps_diagnostics(state))

            # 11. Treatment Effect Estimation (effect estimator)
            cells.extend(self._generate_treatment_effects(state))

            # 12. Sensitivity Analysis (sensitivity analyst)
            if state.sensitivity_results:
                cells.extend(self._generate_sensitivity(state))

            # 13. Analysis Quality & Critique
            if state.critique_history:
                cells.extend(self._generate_critique_section(state))

            # 14. Conclusions
            cells.extend(await self._generate_conclusions(state))

            nb.cells = cells
            notebook_path = self._save_notebook(nb, state.job_id)
            state.notebook_path = notebook_path

            duration_ms = int((time.time() - start_time) * 1000)
            trace = self.create_trace(
                action="notebook_generated",
                reasoning="Generated pipeline report notebook",
                outputs={"path": notebook_path, "n_cells": len(cells)},
                duration_ms=duration_ms,
            )
            state.add_trace(trace)
            self.logger.info(
                "notebook_generation_complete",
                path=notebook_path,
                n_cells=len(cells),
            )

        except Exception as e:
            self.logger.error("notebook_generation_failed", error=str(e))
            state.mark_failed(f"Notebook generation failed: {str(e)}", self.AGENT_NAME)

        return state

    # ─────────────────────── Helper: LLM Narrative ──────────────────────

    async def _generate_llm_narrative(
        self, section: str, context: dict[str, Any]
    ) -> str:
        """Generate LLM-driven narrative for a notebook section.

        Falls back to empty string on any failure so notebook generation
        never breaks due to LLM issues.
        """
        context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
        prompt = f"""Generate a clear, concise narrative for the "{section}" section
of a causal inference analysis notebook.

Context:
{context_str}

Write in clear academic style. Be specific to this dataset.
Output markdown only, no code fences. 2-3 paragraphs maximum."""

        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_instruction=self.SYSTEM_PROMPT,
            )
            # Handle different LLM backends:
            # - Claude returns a dict with content blocks
            # - Gemini returns an object with .text
            if isinstance(response, dict):
                content = response.get("content", [])
                text_parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                text = "\n".join(text_parts)
            else:
                text = response.text
            # Strip leading markdown headings — the caller adds its own
            lines = text.lstrip().split("\n")
            while lines and lines[0].startswith("#"):
                lines.pop(0)
            return "\n".join(lines).strip()
        except Exception as e:
            self.logger.warning(
                "llm_narrative_failed", section=section, error=str(e)
            )
            return ""

    # ─────────────────────── Helper: Dedup Effects ──────────────────────

    def _deduplicate_effects(self, effects: list) -> list:
        """Deduplicate treatment effects by (method, treatment, outcome), case-insensitive."""
        seen: dict[tuple, Any] = {}
        for effect in effects:
            key = (
                effect.method.lower().strip(),
                getattr(effect, "treatment", ""),
                getattr(effect, "outcome", ""),
            )
            seen[key] = effect
        return list(seen.values())

    def _deduplicate_sensitivity(self, results: list) -> list:
        """Deduplicate sensitivity results by method name, keeping last occurrence."""
        seen: dict[str, Any] = {}
        for result in results:
            seen[result.method] = result
        return list(seen.values())

    # ─────────────────────────── 1. Introduction ────────────────────────

    async def _generate_introduction(self, state: AnalysisState) -> list:
        """Generate title and introduction."""
        cells = []

        title = "# Causal Inference Analysis Report\n\n"
        title += f"**Dataset**: {state.dataset_info.name or state.dataset_info.url}\n\n"
        title += f"**Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        title += f"**Job ID**: {state.job_id}\n"
        cells.append(new_markdown_cell(title))

        # Static intro
        intro = "## Introduction\n\n"
        intro += "This notebook reports the results of an automated causal inference analysis "
        intro += f"estimating the effect of **{state.treatment_variable}** on **{state.outcome_variable}**.\n\n"

        if state.data_profile:
            intro += f"The dataset contains **{state.data_profile.n_samples:,}** observations "
            intro += f"and **{state.data_profile.n_features}** variables.\n\n"

        # Pipeline overview
        intro += "### Pipeline Steps Completed\n\n"
        steps = []
        if state.data_profile:
            steps.append("Data Profiling")
        if state.domain_knowledge:
            steps.append("Domain Knowledge Extraction")
        if state.data_repairs:
            n_repairs = len(state.data_repairs)
            steps.append(f"Data Repair ({n_repairs} repair{'s' if n_repairs != 1 else ''})")
        if state.eda_result:
            steps.append("Exploratory Data Analysis")
        if state.proposed_dag:
            steps.append(f"Causal Discovery ({state.proposed_dag.discovery_method})")
        if state.ps_diagnostics:
            quality = state.ps_diagnostics.get("model_quality", "unknown")
            steps.append(f"Propensity Score Diagnostics (quality: {quality})")
        if state.treatment_effects:
            methods = [e.method for e in self._deduplicate_effects(state.treatment_effects)]
            steps.append(f"Treatment Effect Estimation ({', '.join(methods)})")
        if state.sensitivity_results:
            steps.append(f"Sensitivity Analysis ({len(state.sensitivity_results)} tests)")
        if state.critique_history:
            steps.append(f"Quality Review ({state.critique_history[-1].decision.value})")

        for i, step in enumerate(steps, 1):
            intro += f"{i}. {step}\n"

        cells.append(new_markdown_cell(intro))

        # LLM-generated context paragraph
        llm_context = {
            "dataset": state.dataset_info.name or state.dataset_info.url or "unknown",
            "treatment": state.treatment_variable,
            "outcome": state.outcome_variable,
            "n_samples": state.data_profile.n_samples if state.data_profile else "unknown",
            "n_methods": len(self._deduplicate_effects(state.treatment_effects)),
        }
        if state.domain_knowledge and state.domain_knowledge.get("hypotheses"):
            top = state.domain_knowledge["hypotheses"][0]
            if isinstance(top, dict):
                llm_context["key_hypothesis"] = top.get("claim", str(top))

        llm_intro = await self._generate_llm_narrative("introduction", llm_context)
        if llm_intro:
            cells.append(new_markdown_cell(llm_intro))

        return cells

    # ─────────────────── 2. Domain Knowledge ────────────────────────────

    def _generate_domain_knowledge(self, state: AnalysisState) -> list:
        """Report domain knowledge agent findings."""
        cells = []
        dk = state.domain_knowledge

        md = "## Domain Knowledge & Causal Framework\n\n"
        md += "The domain knowledge agent extracted the following understanding from dataset metadata.\n\n"

        # Hypotheses
        hypotheses = dk.get("hypotheses", [])
        if hypotheses:
            md += "### Causal Hypotheses\n\n"
            md += "| # | Hypothesis | Confidence | Evidence |\n"
            md += "|---|-----------|------------|----------|\n"
            for i, h in enumerate(hypotheses, 1):
                if isinstance(h, dict):
                    claim = h.get("claim", h.get("hypothesis", str(h)))
                    conf = h.get("confidence", "unknown")
                    evidence = h.get("evidence", "")
                    md += f"| {i} | {claim} | {conf} | {evidence[:80]} |\n"
                else:
                    md += f"| {i} | {h} | - | - |\n"
            md += "\n"

        # Temporal ordering
        temporal = dk.get("temporal_understanding")
        if temporal:
            md += "### Temporal Ordering\n\n"
            if isinstance(temporal, dict):
                for key, val in temporal.items():
                    md += f"- **{key}**: {val}\n"
            else:
                md += f"{temporal}\n"
            md += "\n"

        # Immutable variables
        immutable = dk.get("immutable_vars", [])
        if immutable:
            md += "### Immutable Variables\n\n"
            md += "These variables cannot be caused by the treatment and are safe to condition on:\n\n"
            for var in immutable:
                if isinstance(var, dict):
                    md += f"- **{var.get('name', var.get('variable', str(var)))}**: {var.get('reason', '')}\n"
                else:
                    md += f"- {var}\n"
            md += "\n"

        # Uncertainties
        uncertainties = dk.get("uncertainties", [])
        if uncertainties:
            md += "### Uncertainties & Limitations\n\n"
            for u in uncertainties:
                if isinstance(u, dict):
                    md += f"- {u.get('issue', u.get('description', str(u)))}\n"
                else:
                    md += f"- {u}\n"
            md += "\n"

        cells.append(new_markdown_cell(md))
        return cells

    # ──────────────────── 3. Setup & Imports ────────────────────────────

    def _generate_setup_cells(self) -> list:
        """Generate setup and import cells."""
        cells = []
        cells.append(
            new_markdown_cell("## Setup\n\nImport required libraries for visualizations.")
        )

        imports = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
print("Setup complete!")"""
        cells.append(new_code_cell(imports))
        return cells

    # ──────────────────── 4. Data Loading ───────────────────────────────

    def _generate_data_loading(self, state: AnalysisState) -> list:
        """Generate data loading cells using actual pipeline path."""
        cells = []
        cells.append(new_markdown_cell("## Data Loading\n\nLoad the dataset used in the analysis."))

        data_path = state.dataset_info.local_path
        if not data_path and hasattr(state, "dataframe_path") and state.dataframe_path:
            data_path = state.dataframe_path

        if data_path:
            if str(data_path).endswith(".parquet"):
                read_call = f'df = pd.read_parquet("{data_path}")'
            else:
                read_call = f'df = pd.read_csv("{data_path}")'

            load_code = f'''# Dataset path from pipeline
DATA_PATH = "{data_path}"
{read_call}

print(f"Dataset shape: {{df.shape}}")
print(f"Columns: {{list(df.columns)}}")
df.head()'''
        else:
            url = state.dataset_info.url or "unknown"
            load_code = f'''# NOTE: Update DATA_PATH to your local copy of the dataset.
# Original source: {url}
DATA_PATH = "UPDATE_THIS_PATH.csv"

df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {{df.shape}}")
print(f"Columns: {{list(df.columns)}}")
df.head()'''

        cells.append(new_code_cell(load_code))
        return cells

    # ──────────────────── 5. Data Profile Report ────────────────────────

    def _generate_data_profile_report(self, state: AnalysisState) -> list:
        """Report data profiler agent findings."""
        cells = []
        profile = state.data_profile
        if not profile:
            return cells

        md = "## Data Profile\n\n"
        md += "*Findings from the Data Profiler agent.*\n\n"

        # Overview
        md += "| Property | Value |\n"
        md += "|----------|-------|\n"
        md += f"| Samples | {profile.n_samples:,} |\n"
        md += f"| Features | {profile.n_features} |\n"
        md += f"| Has Time Dimension | {'Yes' if profile.has_time_dimension else 'No'} |\n"
        if profile.time_column:
            md += f"| Time Column | {profile.time_column} |\n"
        md += "\n"

        # Feature types
        if profile.feature_types:
            md += "### Feature Types\n\n"
            md += "| Feature | Type |\n|---------|------|\n"
            for feat, ftype in sorted(profile.feature_types.items()):
                md += f"| {feat} | {ftype} |\n"
            md += "\n"

        # Missing values
        if profile.missing_values:
            has_missing = {k: v for k, v in profile.missing_values.items() if v > 0}
            if has_missing:
                md += "### Missing Values\n\n"
                md += "| Feature | Missing Count | % Missing |\n"
                md += "|---------|--------------|----------|\n"
                for feat, count in sorted(has_missing.items(), key=lambda x: -x[1]):
                    pct = 100 * count / profile.n_samples if profile.n_samples > 0 else 0
                    md += f"| {feat} | {count} | {pct:.1f}% |\n"
                md += "\n"
            else:
                md += "**No missing values detected.**\n\n"

        # Causal candidates
        md += "### Variable Roles (Identified by Profiler)\n\n"
        md += f"- **Treatment**: {state.treatment_variable}\n"
        md += f"- **Outcome**: {state.outcome_variable}\n"
        if profile.treatment_candidates:
            md += f"- **Treatment candidates**: {', '.join(profile.treatment_candidates)}\n"
        if profile.outcome_candidates:
            md += f"- **Outcome candidates**: {', '.join(profile.outcome_candidates)}\n"
        if profile.potential_confounders:
            md += f"- **Potential confounders**: {', '.join(profile.potential_confounders)}\n"
        if profile.potential_instruments:
            md += f"- **Potential instruments**: {', '.join(profile.potential_instruments)}\n"
        if profile.discontinuity_candidates:
            md += f"- **Discontinuity candidates**: {', '.join(profile.discontinuity_candidates)}\n"
        md += "\n"

        cells.append(new_markdown_cell(md))
        return cells

    # ──────────────────── 6. Data Repairs ──────────────────────────────

    def _generate_data_repairs(self, state: AnalysisState) -> list:
        """Report data repair agent findings — what preprocessing was applied."""
        cells = []
        repairs = state.data_repairs
        if not repairs:
            return cells

        md = "## Data Preprocessing & Repairs\n\n"
        md += "*Findings from the Data Repair agent. "
        md += "These repairs were applied before causal analysis.*\n\n"

        # Repairs summary table
        md += "### Repairs Applied\n\n"
        md += "| # | Type | Strategy | Columns |\n"
        md += "|---|------|----------|---------|\n"
        for i, repair in enumerate(repairs, 1):
            rtype = repair.get("type", "unknown")
            strategy = repair.get("strategy", "unknown")
            columns = repair.get("columns", [])
            col_str = ", ".join(columns[:5])
            if len(columns) > 5:
                col_str += f" (+{len(columns) - 5} more)"
            md += f"| {i} | {rtype} | {strategy} | {col_str} |\n"
        md += "\n"

        # Detail each repair type
        missing_repairs = [r for r in repairs if r.get("type") == "missing"]
        outlier_repairs = [r for r in repairs if r.get("type") == "outliers"]
        collinearity_repairs = [r for r in repairs if r.get("type") == "collinearity"]

        if missing_repairs:
            md += "### Missing Value Handling\n\n"
            for r in missing_repairs:
                strategy = r.get("strategy", "unknown")
                columns = r.get("columns", [])
                md += f"- **Strategy**: {strategy}\n"
                md += f"- **Columns**: {', '.join(columns)}\n"
                if r.get("rows_dropped"):
                    md += f"- **Rows dropped**: {r['rows_dropped']:,}\n"
                if r.get("before") is not None and r.get("after") is not None:
                    md += f"- **Missing values**: {r['before']:,} → {r['after']:,}\n"
                md += "\n"

        if outlier_repairs:
            md += "### Outlier Treatment\n\n"
            for r in outlier_repairs:
                strategy = r.get("strategy", "unknown")
                columns = r.get("columns", [])
                md += f"- **Strategy**: {strategy}\n"
                md += f"- **Columns**: {', '.join(columns)}\n\n"

        if collinearity_repairs:
            md += "### Collinearity Resolution\n\n"
            for r in collinearity_repairs:
                strategy = r.get("strategy", "unknown")
                columns = r.get("columns", [])
                md += f"- **Strategy**: {strategy}\n"
                md += f"- **Columns removed/adjusted**: {', '.join(columns)}\n\n"

        # Quality assessment and cautions (from finalize_repairs)
        # These may be stored in the last repair entry or via add_agent_result
        # Check if any repair dict contains these fields
        quality = None
        summary_lines = None
        cautions = None
        for r in repairs:
            if "quality_assessment" in r:
                quality = r["quality_assessment"]
            if "repairs_summary" in r:
                summary_lines = r["repairs_summary"]
            if "cautions" in r:
                cautions = r["cautions"]

        if quality:
            md += f"### Data Quality Assessment\n\n{quality}\n\n"

        if cautions:
            md += "### Cautions\n\n"
            md += "*These caveats may affect the validity of downstream results.*\n\n"
            for c in cautions:
                md += f"- {c}\n"
            md += "\n"

        cells.append(new_markdown_cell(md))
        return cells

    # ──────────────────── 7. EDA Report ─────────────────────────────────

    def _generate_eda_report(self, state: AnalysisState) -> list:
        """Report EDA agent findings with visualizations from pipeline data."""
        cells = []

        cells.append(new_markdown_cell(
            "## Exploratory Data Analysis\n\n*Findings from the EDA agent.*"
        ))

        eda = state.eda_result

        # Quality summary
        if eda:
            quality_md = "### Data Quality\n\n"
            quality_md += f"**Overall Quality Score**: {eda.data_quality_score:.1f}/100\n\n"
            if eda.data_quality_issues:
                quality_md += "**Issues Found:**\n"
                for issue in eda.data_quality_issues:
                    quality_md += f"- {issue}\n"
            else:
                quality_md += "No significant data quality issues detected.\n"
            quality_md += "\n"
            cells.append(new_markdown_cell(quality_md))

        # High correlations
        if eda and eda.high_correlations:
            corr_md = "### High Correlations (|r| > 0.7)\n\n"
            corr_md += "| Variable 1 | Variable 2 | Correlation |\n"
            corr_md += "|-----------|-----------|------------|\n"
            for hc in eda.high_correlations[:15]:
                if isinstance(hc, dict):
                    v1 = hc.get("var1", hc.get("column1", "?"))
                    v2 = hc.get("var2", hc.get("column2", "?"))
                    r = hc.get("correlation", hc.get("r", "?"))
                    r_str = f"{r:.3f}" if isinstance(r, (int, float)) else str(r)
                    corr_md += f"| {v1} | {v2} | {r_str} |\n"
            corr_md += "\n"
            cells.append(new_markdown_cell(corr_md))

        # Multicollinearity
        if eda and (eda.vif_scores or eda.multicollinearity_warnings):
            vif_md = "### Multicollinearity\n\n"
            if eda.multicollinearity_warnings:
                for w in eda.multicollinearity_warnings:
                    vif_md += f"- {w}\n"
                vif_md += "\n"
            if eda.vif_scores:
                vif_md += "| Variable | VIF |\n|----------|-----|\n"
                for var, vif in sorted(
                    eda.vif_scores.items(), key=lambda x: -x[1]
                )[:10]:
                    severity = " **SEVERE**" if vif > 10 else " (moderate)" if vif > 5 else ""
                    vif_md += f"| {var} | {vif:.2f}{severity} |\n"
                vif_md += "\n"
            cells.append(new_markdown_cell(vif_md))

        # Covariate balance
        if eda and eda.covariate_balance:
            bal_md = "### Covariate Balance\n\n"
            if eda.balance_summary:
                bal_md += f"{eda.balance_summary}\n\n"
            bal_md += "| Covariate | Treated Mean | Control Mean | SMD | Balanced |\n"
            bal_md += "|-----------|-------------|-------------|-----|----------|\n"
            for cov, vals in eda.covariate_balance.items():
                if isinstance(vals, dict):
                    t_mean = vals.get("treated_mean", vals.get("mean_treated", "?"))
                    c_mean = vals.get("control_mean", vals.get("mean_control", "?"))
                    smd = vals.get("smd", vals.get("std_mean_diff", "?"))
                    balanced = vals.get("is_balanced", vals.get("balanced", "?"))
                    t_str = f"{t_mean:.3f}" if isinstance(t_mean, (int, float)) else str(t_mean)
                    c_str = f"{c_mean:.3f}" if isinstance(c_mean, (int, float)) else str(c_mean)
                    s_str = f"{smd:.3f}" if isinstance(smd, (int, float)) else str(smd)
                    b_str = "Yes" if balanced else "No"
                    bal_md += f"| {cov} | {t_str} | {c_str} | {s_str} | {b_str} |\n"
            bal_md += "\n"
            cells.append(new_markdown_cell(bal_md))

        # EDA summary findings
        if eda and eda.summary_table:
            if isinstance(eda.summary_table, dict):
                findings = eda.summary_table.get("key_findings", [])
                recs = eda.summary_table.get("recommendations", [])
                readiness = eda.summary_table.get("causal_readiness", "")

                if findings or recs or readiness:
                    summary_md = "### EDA Summary\n\n"
                    if findings:
                        summary_md += "**Key Findings:**\n"
                        for f in findings:
                            summary_md += f"- {f}\n"
                        summary_md += "\n"
                    if recs:
                        summary_md += "**Recommendations:**\n"
                        for r in recs:
                            summary_md += f"- {r}\n"
                        summary_md += "\n"
                    if readiness:
                        summary_md += f"**Causal Readiness**: {readiness}\n\n"
                    cells.append(new_markdown_cell(summary_md))

        # Visualization: Distribution plots for treatment & outcome
        cells.append(new_markdown_cell("### Distribution Visualizations"))
        dist_code = f'''# Distribution plots for treatment and outcome
treatment_var = "{state.treatment_variable}"
outcome_var = "{state.outcome_variable}"

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Treatment distribution
ax = axes[0]
if df[treatment_var].nunique() <= 10:
    df[treatment_var].value_counts().sort_index().plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
else:
    df[treatment_var].hist(bins=30, ax=ax, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_title(f'{{treatment_var}} (Treatment)')
ax.set_xlabel(treatment_var)
ax.set_ylabel('Count')

# Outcome distribution
ax = axes[1]
df[outcome_var].hist(bins=30, ax=ax, color='coral', alpha=0.7, edgecolor='black')
ax.set_title(f'{{outcome_var}} (Outcome)')
ax.set_xlabel(outcome_var)
ax.set_ylabel('Count')

plt.tight_layout()
plt.show()'''
        cells.append(new_code_cell(dist_code))

        # Correlation heatmap from pipeline data
        if eda and eda.correlation_matrix:
            cells.append(new_markdown_cell("### Correlation Heatmap"))
            corr_data = json.dumps(eda.correlation_matrix)
            heatmap_code = f'''# Correlation matrix from pipeline
import json
corr_data = json.loads('{corr_data}')
corr_df = pd.DataFrame(corr_data)

# Limit display to 15 variables
if len(corr_df.columns) > 15:
    cols = corr_df.columns[:15]
    corr_df = corr_df.loc[cols, cols]

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title('Correlation Matrix (from EDA agent)')
plt.tight_layout()
plt.show()'''
            cells.append(new_code_cell(heatmap_code))

        return cells

    # ──────────────────── 8. Causal Structure ───────────────────────────

    def _generate_causal_structure(self, state: AnalysisState) -> list:
        """Report causal discovery agent findings."""
        cells = []
        dag = state.proposed_dag

        md = "## Causal Structure\n\n"
        md += f"*Discovered by the Causal Discovery agent using **{dag.discovery_method}**.*\n\n"
        md += "The graph below represents the discovered causal relationships.\n"
        md += "**Green** = treatment, **Red** = outcome, **Blue** = other variables.\n"
        cells.append(new_markdown_cell(md))

        # DAG visualization code
        edges_json = json.dumps([(e.source, e.target) for e in dag.edges])
        nodes_json = json.dumps(dag.nodes)

        dag_code = f'''# Causal graph from pipeline
import networkx as nx

G = nx.DiGraph()
G.add_nodes_from({nodes_json})
G.add_edges_from({edges_json})

fig, ax = plt.subplots(figsize=(12, 8))
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

treatment_node = "{state.treatment_variable}"
outcome_node = "{state.outcome_variable}"

node_colors = []
for node in G.nodes():
    if node == treatment_node:
        node_colors.append('lightgreen')
    elif node == outcome_node:
        node_colors.append('lightcoral')
    else:
        node_colors.append('lightblue')

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.9, ax=ax)
nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20,
                       connectionstyle="arc3,rad=0.1", ax=ax)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
ax.set_title('Discovered Causal Graph')
ax.axis('off')
plt.tight_layout()
plt.show()

print(f"Nodes: {{len(G.nodes())}}, Edges: {{len(G.edges())}}")'''
        cells.append(new_code_cell(dag_code))

        # DAG interpretation
        if dag.interpretation:
            cells.append(new_markdown_cell(
                f"### Causal Graph Interpretation\n\n{dag.interpretation}\n"
            ))

        # Edge details
        if dag.edges:
            edge_md = "### Edge Details\n\n"
            edge_md += "| Source | Target | Type | Confidence |\n"
            edge_md += "|--------|--------|------|------------|\n"
            for e in dag.edges:
                conf_str = f"{e.confidence:.2f}" if isinstance(e.confidence, (int, float)) else str(e.confidence)
                edge_md += f"| {e.source} | {e.target} | {e.edge_type} | {conf_str} |\n"
            edge_md += "\n"
            cells.append(new_markdown_cell(edge_md))

        return cells

    # ──────────────────── 9. Confounder Analysis ────────────────────────

    def _generate_confounder_analysis(self, state: AnalysisState) -> list:
        """Report confounder identification from the pipeline."""
        cells = []

        md = "## Confounder Analysis\n\n"
        md += "*Variables identified as potential confounders by the pipeline.*\n\n"

        # From confounder_discovery if available
        if state.confounder_discovery:
            cd = state.confounder_discovery
            ranked = cd.get("ranked_confounders", [])
            if ranked:
                md += "### Ranked Confounders\n\n"
                if isinstance(ranked[0], dict):
                    md += "| Rank | Variable | Evidence |\n|------|----------|----------|\n"
                    for i, c in enumerate(ranked, 1):
                        name = c.get("variable", c.get("name", str(c)))
                        evidence = c.get("evidence", c.get("reason", "-"))
                        md += f"| {i} | {name} | {evidence} |\n"
                else:
                    md += "| Rank | Variable |\n|------|----------|\n"
                    for i, c in enumerate(ranked, 1):
                        md += f"| {i} | {c} |\n"
                md += "\n"

            strategy = cd.get("adjustment_strategy", "")
            if strategy:
                md += f"**Adjustment Strategy**: {strategy}\n\n"

        # From data profile
        elif state.data_profile and state.data_profile.potential_confounders:
            confounders = state.data_profile.potential_confounders
            md += "### Potential Confounders (from Data Profile)\n\n"
            for c in confounders:
                ftype = state.data_profile.feature_types.get(c, "unknown") if state.data_profile.feature_types else "unknown"
                md += f"- **{c}** ({ftype})\n"
            md += "\n"

        # Analyzed pairs
        if state.analyzed_pairs:
            md += "### Analyzed Treatment-Outcome Pairs\n\n"
            md += "| Treatment | Outcome | Priority | Rationale |\n"
            md += "|-----------|---------|----------|-----------|\n"
            for pair in state.analyzed_pairs:
                md += f"| {pair.treatment} | {pair.outcome} | {pair.priority} | {pair.rationale} |\n"
            md += "\n"

        cells.append(new_markdown_cell(md))
        return cells

    # ──────────────────── 10. Propensity Score Diagnostics ─────────────

    def _generate_ps_diagnostics(self, state: AnalysisState) -> list:
        """Report propensity score diagnostics agent findings."""
        cells = []
        ps = state.ps_diagnostics
        if not ps:
            return cells

        md = "## Propensity Score Diagnostics\n\n"
        md += "*Assessment from the PS Diagnostics agent — validates whether "
        md += "propensity-score-based estimation methods are reliable for this data.*\n\n"

        # Model quality badge
        quality = ps.get("model_quality", "unknown")
        proceed = ps.get("proceed_with_analysis", True)
        recommended = ps.get("recommended_method", "unknown")

        md += "### Summary\n\n"
        md += "| Diagnostic | Result |\n"
        md += "|-----------|--------|\n"
        md += f"| PS Model Quality | **{quality.replace('_', ' ').title()}** |\n"
        md += f"| Proceed with Analysis | **{'Yes' if proceed else 'No'}** |\n"
        md += f"| Recommended Method | **{recommended.upper().replace('_', ' ')}** |\n"

        # Trimming bounds
        trimming = ps.get("trimming_bounds")
        if trimming:
            md += f"| PS Trimming Bounds | [{trimming[0]:.3f}, {trimming[1]:.3f}] |\n"
        else:
            md += "| PS Trimming Bounds | None (no trimming needed) |\n"
        md += "\n"

        # Warnings
        warnings_list = ps.get("warnings", [])
        if warnings_list:
            md += "### Warnings\n\n"
            for w in warnings_list:
                md += f"- {w}\n"
            md += "\n"

        # Reasoning
        reasoning = ps.get("reasoning", "")
        if reasoning:
            md += "### Diagnostic Reasoning\n\n"
            md += f"{reasoning}\n\n"

        # Not-proceed warning box
        if not proceed:
            md += "> **WARNING**: The PS diagnostics agent recommends **not proceeding** "
            md += "with propensity-score-based methods for this dataset. Treatment effect "
            md += "estimates using IPW or matching should be interpreted with extreme caution.\n\n"

        cells.append(new_markdown_cell(md))

        # PS distribution visualization code (if we have the data)
        cells.append(new_markdown_cell(
            "### Propensity Score Distribution\n\n"
            "Visualize the estimated propensity scores by treatment group to assess overlap."
        ))

        trimming_code = ""
        if trimming:
            trimming_code = (
                f"\n    # Apply trimming bounds from PS diagnostics\n"
                f"    trim_lower, trim_upper = {trimming[0]}, {trimming[1]}\n"
                f"    mask = (ps_scores >= trim_lower) & (ps_scores <= trim_upper)\n"
                f"    print(f\"Trimming: {{{{(~mask).sum()}}}} observations outside "
                f"[{trimming[0]:.3f}, {trimming[1]:.3f}]\")\n"
            )

        ps_code = f'''# Propensity score estimation and overlap check
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

treatment_var = "{state.treatment_variable}"
outcome_var = "{state.outcome_variable}"

# Get covariates (all numeric columns except treatment and outcome)
covariates = [c for c in df.select_dtypes(include=[np.number]).columns
              if c != treatment_var and c != outcome_var]

if covariates:
    X = df[covariates].fillna(df[covariates].median())
    T = df[treatment_var].values

    # Binarize treatment if continuous
    if len(np.unique(T)) > 2:
        T = (T > np.median(T)).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, T)
    ps_scores = lr.predict_proba(X_scaled)[:, 1]
{trimming_code}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution by group
    ax = axes[0]
    ax.hist(ps_scores[T == 0], bins=30, alpha=0.6, label="Control", color="steelblue", edgecolor="black")
    ax.hist(ps_scores[T == 1], bins=30, alpha=0.6, label="Treated", color="coral", edgecolor="black")
    ax.set_xlabel("Propensity Score")
    ax.set_ylabel("Count")
    ax.set_title("PS Distribution by Treatment Group")
    ax.legend()

    # Overlap assessment
    ax = axes[1]
    ax.boxplot([ps_scores[T == 0], ps_scores[T == 1]], labels=["Control", "Treated"])
    ax.set_ylabel("Propensity Score")
    ax.set_title("PS Overlap Assessment")

    plt.tight_layout()
    plt.show()

    # Print diagnostics
    print(f"PS Model Quality: {quality}")
    print(f"Recommended Method: {recommended}")
    print(f"Overlap: ({{ps_scores[T == 1].min():.3f}}-{{ps_scores[T == 1].max():.3f}}) treated, "
          f"({{ps_scores[T == 0].min():.3f}}-{{ps_scores[T == 0].max():.3f}}) control")
else:
    print("No numeric covariates available for PS estimation.")'''
        cells.append(new_code_cell(ps_code))

        return cells

    # ──────────────────── 11. Treatment Effects ──────────────────────────

    def _generate_treatment_effects(self, state: AnalysisState) -> list:
        """Report treatment effect estimation results."""
        cells = []

        effects = self._deduplicate_effects(state.treatment_effects)

        md = "## Treatment Effect Estimation\n\n"
        md += "*Results from the Effect Estimator agent.*\n\n"
        md += f"**Treatment**: {state.treatment_variable}\n"
        md += f"**Outcome**: {state.outcome_variable}\n"
        md += f"**Methods applied**: {len(effects)}\n\n"
        cells.append(new_markdown_cell(md))

        # Results table
        if effects:
            table_md = "### Results Summary\n\n"
            table_md += "| Method | Estimand | Estimate | Std Error | 95% CI | p-value |\n"
            table_md += "|--------|----------|----------|-----------|--------|--------|\n"
            for e in effects:
                pval = f"{e.p_value:.4f}" if e.p_value is not None else "N/A"
                table_md += (
                    f"| {e.method} | {e.estimand} | {e.estimate:.4f} | "
                    f"{e.std_error:.4f} | [{e.ci_lower:.4f}, {e.ci_upper:.4f}] | {pval} |\n"
                )
            table_md += "\n"
            cells.append(new_markdown_cell(table_md))

        # Per-method details
        for e in effects:
            if e.details or e.assumptions_tested:
                detail_md = f"#### {e.method} Details\n\n"
                if e.assumptions_tested:
                    detail_md += "**Assumptions tested:**\n"
                    for a in e.assumptions_tested:
                        detail_md += f"- {a}\n"
                    detail_md += "\n"
                if e.details:
                    detail_md += "**Diagnostics:**\n"
                    for k, v in e.details.items():
                        if isinstance(v, float):
                            detail_md += f"- {k}: {v:.4f}\n"
                        elif not isinstance(v, (list, dict)):
                            detail_md += f"- {k}: {v}\n"
                    detail_md += "\n"
                cells.append(new_markdown_cell(detail_md))

        # Forest plot
        if effects:
            cells.append(new_markdown_cell("### Treatment Effect Comparison"))
            results_json = json.dumps([
                {
                    "method": e.method,
                    "estimate": e.estimate,
                    "ci_lower": e.ci_lower,
                    "ci_upper": e.ci_upper,
                }
                for e in effects
            ])

            plot_code = f'''# Forest plot of treatment effect estimates
import json
results = json.loads('{results_json}')

fig, ax = plt.subplots(figsize=(10, max(4, len(results) * 1.2)))

methods = [r['method'] for r in results]
estimates = [r['estimate'] for r in results]
ci_lower = [r['ci_lower'] for r in results]
ci_upper = [r['ci_upper'] for r in results]

y_pos = list(range(len(methods)))
xerr_lower = [e - l for e, l in zip(estimates, ci_lower)]
xerr_upper = [u - e for e, u in zip(estimates, ci_upper)]

# Traditional forest plot: point estimates with CI whiskers
ax.errorbar(estimates, y_pos, xerr=[xerr_lower, xerr_upper],
            fmt='o', color='steelblue', markersize=8, capsize=6,
            elinewidth=2, markeredgewidth=2)
ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Zero effect')
ax.set_yticks(y_pos)
ax.set_yticklabels(methods)
ax.set_xlabel('Treatment Effect Estimate')
ax.set_title('Forest Plot: Treatment Effect Estimates Across Methods')
ax.legend()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()'''
            cells.append(new_code_cell(plot_code))

        # Verification OLS code (reproducibility check)
        if effects:
            # Get numeric covariates only
            numeric_confounders: list[str] = []
            if state.data_profile and state.data_profile.potential_confounders:
                numeric_types = {"numeric", "binary", "ordinal"}
                ft = state.data_profile.feature_types or {}
                numeric_confounders = [
                    c
                    for c in state.data_profile.potential_confounders[:10]
                    if ft.get(c) in numeric_types
                    and c != state.treatment_variable
                    and c != state.outcome_variable
                ]

            cells.append(new_markdown_cell(
                "### Verification: OLS Regression\n\n"
                "Run this cell to independently verify the OLS estimate."
            ))

            covariates_json = json.dumps(numeric_confounders)
            verify_code = f'''# Verification: OLS regression
import statsmodels.api as sm

treatment_var = "{state.treatment_variable}"
outcome_var = "{state.outcome_variable}"
covariates = {covariates_json}

# Filter to numeric covariates only (safety check)
covariates = [c for c in covariates if c in df.columns
              and pd.api.types.is_numeric_dtype(df[c])]

all_cols = [treatment_var, outcome_var] + covariates
df_clean = df[all_cols].dropna()

T = df_clean[treatment_var].values.astype(float)
Y = df_clean[outcome_var].values.astype(float)

# Binarize continuous treatment at median
if len(np.unique(T)) > 2:
    median_t = np.median(T)
    T_binary = (T > median_t).astype(int)
    print(f"Treatment binarized at median ({{median_t:.2f}})")
else:
    T_binary = T

if covariates:
    X = df_clean[covariates].values.astype(float)
    design = np.column_stack([np.ones(len(T_binary)), T_binary, X])
else:
    design = np.column_stack([np.ones(len(T_binary)), T_binary])

model = sm.OLS(Y, design)
results = model.fit()

print(f"\\nVerification OLS Results:")
print(f"  ATE:      {{results.params[1]:.4f}}")
print(f"  SE:       {{results.bse[1]:.4f}}")
print(f"  95% CI:   [{{results.conf_int()[1][0]:.4f}}, {{results.conf_int()[1][1]:.4f}}]")
print(f"  p-value:  {{results.pvalues[1]:.4f}}")
print(f"  R-squared: {{results.rsquared:.4f}}")
print(f"  N:        {{len(df_clean)}}")'''
            cells.append(new_code_cell(verify_code))

        return cells

    # ──────────────────── 12. Sensitivity Analysis ──────────────────────

    def _generate_sensitivity(self, state: AnalysisState) -> list:
        """Report sensitivity analysis results with executable verification code."""
        cells = []
        sensitivity_results = self._deduplicate_sensitivity(state.sensitivity_results)

        cells.append(new_markdown_cell(
            "## Sensitivity Analysis\n\n*Findings from the Sensitivity Analyst agent.*"
        ))

        # Results summary table
        if sensitivity_results:
            table_md = "### Results Summary\n\n"
            table_md += "| Analysis | Robustness Value | Interpretation |\n"
            table_md += "|----------|------------------|----------------|\n"
            for s in sensitivity_results:
                interp = s.interpretation[:100] + ("..." if len(s.interpretation) > 100 else "")
                table_md += f"| {s.method} | {s.robustness_value:.2f} | {interp} |\n"
            table_md += "\n"
            cells.append(new_markdown_cell(table_md))

        # Per-method details
        for s in sensitivity_results:
            if s.details:
                detail_md = f"#### {s.method}\n\n"
                detail_md += f"**Interpretation**: {s.interpretation}\n\n"
                detail_md += "**Details:**\n"
                for k, v in s.details.items():
                    if k == "subgroup_effects" and isinstance(v, list):
                        detail_md += "\n**Per-subgroup Effects:**\n\n"
                        detail_md += "| Subgroup | Effect |\n|----------|--------|\n"
                        for sg in v:
                            label = sg.get("label", "?")
                            effect = sg.get("effect", 0)
                            detail_md += f"| {label} | {effect:.4f} |\n"
                        detail_md += "\n"
                    elif isinstance(v, float):
                        detail_md += f"- {k}: {v:.4f}\n"
                    elif isinstance(v, (str, int, bool)):
                        detail_md += f"- {k}: {v}\n"
                detail_md += "\n"
                cells.append(new_markdown_cell(detail_md))

        # Executable: E-value computation
        cells.append(new_markdown_cell(
            "### Verification: E-value Computation\n\n"
            "The E-value quantifies how strong unmeasured confounding "
            "would need to be to explain away the observed effect."
        ))

        evalue_code = '''# E-value computation
def compute_e_value(ate, se, y_std):
    """Compute E-value from treatment effect estimate."""
    if abs(ate) < 1e-10 or y_std < 1e-10:
        return 1.0, 1.0

    # Approximate risk ratio
    rr = np.exp(0.91 * ate / y_std)
    if rr < 1:
        rr = 1 / rr

    e_val = rr + np.sqrt(rr * (rr - 1))

    # E-value for CI bound
    ci_bound = abs(ate) - 1.96 * se
    if ci_bound > 0:
        rr_ci = np.exp(0.91 * ci_bound / y_std)
        if rr_ci < 1:
            rr_ci = 1 / rr_ci
        e_val_ci = rr_ci + np.sqrt(rr_ci * (rr_ci - 1))
    else:
        e_val_ci = 1.0

    return e_val, e_val_ci

# Use the verification OLS results
e_val, e_val_ci = compute_e_value(results.params[1], results.bse[1], np.std(Y))
print(f"E-value (point estimate): {e_val:.2f}")
print(f"E-value (CI bound):       {e_val_ci:.2f}")
print()
if e_val > 3:
    print("STRONG robustness: A very strong unmeasured confounder would be needed.")
elif e_val > 2:
    print("MODERATE robustness: A moderately strong confounder could explain this.")
else:
    print("WEAK robustness: A relatively weak confounder could explain the effect.")'''
        cells.append(new_code_cell(evalue_code))

        # Executable: Placebo test
        cells.append(new_markdown_cell(
            "### Verification: Placebo Test\n\n"
            "Checks whether the treatment effect disappears under random treatment assignment."
        ))

        placebo_code = '''# Placebo test: permutation-based
n_permutations = 500
placebo_effects = []

for _ in range(n_permutations):
    T_placebo = np.random.permutation(T_binary)
    if covariates:
        X_p = df_clean[covariates].values.astype(float)
        design_p = np.column_stack([np.ones(len(T_placebo)), T_placebo, X_p])
    else:
        design_p = np.column_stack([np.ones(len(T_placebo)), T_placebo])

    try:
        res_p = sm.OLS(Y, design_p).fit()
        placebo_effects.append(res_p.params[1])
    except Exception:
        pass

real_ate = results.params[1]
p_value_perm = np.mean(np.abs(placebo_effects) >= np.abs(real_ate))

print(f"Real ATE:             {real_ate:.4f}")
print(f"Mean placebo effect:  {np.mean(placebo_effects):.4f}")
print(f"Permutation p-value:  {p_value_perm:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(placebo_effects, bins=50, alpha=0.7, color='gray', edgecolor='black', label='Placebo effects')
ax.axvline(x=real_ate, color='red', linewidth=2, label=f'Real ATE: {real_ate:.4f}')
ax.set_xlabel('Effect Estimate')
ax.set_ylabel('Frequency')
ax.set_title('Placebo Test: Real Effect vs Permuted Effects')
ax.legend()
plt.tight_layout()
plt.show()'''
        cells.append(new_code_cell(placebo_code))

        return cells

    # ──────────────────── 13. Critique Section ──────────────────────────

    def _generate_critique_section(self, state: AnalysisState) -> list:
        """Report critique agent findings."""
        cells = []

        md = "## Analysis Quality & Critique\n\n"
        md += "*Assessment from the automated Critique agent.*\n\n"

        for critique in state.critique_history:
            md += f"### Iteration {critique.iteration}\n\n"
            md += f"**Decision**: **{critique.decision.value}**\n\n"

            # Quality scores
            if critique.scores:
                md += "**Quality Scores:**\n\n"
                md += "| Dimension | Score |\n|-----------|-------|\n"
                for dim, score in critique.scores.items():
                    bar = "█" * score + "░" * (5 - score)
                    md += f"| {dim.replace('_', ' ').title()} | {bar} {score}/5 |\n"
                md += "\n"

            # Issues
            if critique.issues:
                md += "**Issues Identified:**\n"
                for issue in critique.issues:
                    md += f"- {issue}\n"
                md += "\n"

            # Improvements
            if critique.improvements:
                md += "**Improvements:**\n"
                for imp in critique.improvements:
                    md += f"- {imp}\n"
                md += "\n"

            # Reasoning
            if critique.reasoning:
                md += f"**Reasoning**: {critique.reasoning[:500]}\n\n"

            md += "---\n\n"

        cells.append(new_markdown_cell(md))
        return cells

    # ──────────────────── 14. Conclusions ───────────────────────────────

    async def _generate_conclusions(self, state: AnalysisState) -> list:
        """Generate conclusions section with LLM narrative."""
        cells = []

        effects = self._deduplicate_effects(state.treatment_effects)
        avg_effect = np.mean([e.estimate for e in effects]) if effects else 0
        all_positive = all(e.estimate > 0 for e in effects) if effects else False
        all_negative = all(e.estimate < 0 for e in effects) if effects else False

        # Try LLM-generated conclusion
        llm_context = {
            "treatment": state.treatment_variable,
            "outcome": state.outcome_variable,
            "n_methods": len(effects),
            "avg_effect": f"{avg_effect:.4f}",
            "direction": "positive" if all_positive else "negative" if all_negative else "mixed",
        }

        if state.sensitivity_results:
            llm_context["sensitivity"] = "; ".join(
                f"{s.method}: {s.robustness_value:.2f}" for s in state.sensitivity_results
            )

        if state.critique_history:
            latest = state.critique_history[-1]
            llm_context["critique_decision"] = latest.decision.value
            if latest.issues:
                llm_context["key_issues"] = "; ".join(latest.issues[:3])

        if state.domain_knowledge and state.domain_knowledge.get("uncertainties"):
            uncerts = state.domain_knowledge["uncertainties"]
            if uncerts:
                first = uncerts[0]
                if isinstance(first, dict):
                    llm_context["uncertainty"] = first.get("issue", str(first))
                else:
                    llm_context["uncertainty"] = str(first)

        llm_conclusion = await self._generate_llm_narrative("conclusions", llm_context)

        # Build conclusions section
        conclusion_md = "## Conclusions\n\n"

        if llm_conclusion:
            conclusion_md += llm_conclusion + "\n\n"
        else:
            # Fallback: template conclusion
            conclusion_md += "### Key Findings\n\n"
            conclusion_md += f"The analysis estimated the effect of **{state.treatment_variable}** "
            conclusion_md += f"on **{state.outcome_variable}** using {len(effects)} method(s).\n\n"
            conclusion_md += f"- **Average Treatment Effect**: {avg_effect:.4f}\n"
            direction = "positive" if all_positive else "negative" if all_negative else "mixed"
            conclusion_md += f"- **Direction consistency**: All methods agree on a **{direction}** effect.\n\n"

        # Recommendations (from orchestrator)
        if state.recommendations:
            conclusion_md += "### Recommendations\n\n"
            for i, rec in enumerate(state.recommendations, 1):
                conclusion_md += f"{i}. {rec}\n"
            conclusion_md += "\n"

        # Standard limitations
        conclusion_md += "### Limitations\n\n"
        conclusion_md += "- **Observational data**: Cannot rule out unmeasured confounding\n"
        conclusion_md += "- **Model assumptions**: Each method relies on specific assumptions\n"
        conclusion_md += "- **External validity**: Results may not generalize to other populations\n\n"

        cells.append(new_markdown_cell(conclusion_md))

        # Reproducibility footer
        repro_md = f"""---

### Reproducibility Information

- **Analysis Date**: {datetime.utcnow().strftime('%Y-%m-%d')}
- **Job ID**: {state.job_id}
- **Dataset**: {state.dataset_info.name or state.dataset_info.url}
- **Treatment Variable**: {state.treatment_variable}
- **Outcome Variable**: {state.outcome_variable}

This notebook was automatically generated by the Causal Inference Orchestrator.
"""
        cells.append(new_markdown_cell(repro_md))

        return cells

    # ──────────────────── Save Notebook ─────────────────────────────────

    def _save_notebook(self, nb: nbformat.NotebookNode, job_id: str) -> str:
        """Save the notebook to disk."""
        output_dir = Path(tempfile.gettempdir()) / "causal_orchestrator" / "notebooks"
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"causal_analysis_{job_id}.ipynb"
        filepath = output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        return str(filepath)
