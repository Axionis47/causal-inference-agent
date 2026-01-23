"""Notebook Generator Agent - Creates reproducible Jupyter notebooks."""

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
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


class NotebookGeneratorAgent(BaseAgent):
    """Agent that generates reproducible Jupyter notebooks.

    This agent compiles all analysis results into a well-documented,
    executable Jupyter notebook that includes:

    1. Introduction and dataset description
    2. Data loading and preprocessing code
    3. Exploratory data analysis
    4. Causal graph visualization
    5. Treatment effect estimation code and results
    6. Sensitivity analysis
    7. Conclusions and recommendations

    Uses LLM reasoning to:
    1. Generate clear explanations
    2. Add appropriate commentary
    3. Structure the narrative

    The notebook is fully reproducible with all code executable.
    """

    AGENT_NAME = "notebook_generator"

    SYSTEM_PROMPT = """You are an expert at creating clear, reproducible Jupyter notebooks
for causal inference analysis.

When generating notebooks:
1. Start with a clear introduction explaining the research question
2. Document all data preprocessing steps
3. Explain the choice of methods
4. Present results with proper interpretation
5. Discuss limitations and caveats
6. End with actionable conclusions

Write in a clear, educational style that helps readers understand both
the methods and the findings. Include comments in code cells to explain
what each step does.

The notebook should be fully executable - readers should be able to
reproduce the analysis by running all cells in order."""

    TOOLS = [
        {
            "name": "generate_narrative",
            "description": "Generate narrative text for a notebook section",
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "enum": [
                            "introduction",
                            "data_description",
                            "methodology",
                            "results",
                            "sensitivity",
                            "conclusions",
                        ],
                        "description": "The section to generate narrative for",
                    },
                    "content": {
                        "type": "string",
                        "description": "The narrative content for this section",
                    },
                },
                "required": ["section", "content"],
            },
        },
    ]

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Generate the analysis notebook.

        Args:
            state: Current analysis state with all results

        Returns:
            Updated state with notebook path
        """
        self.logger.info(
            "notebook_generation_start",
            job_id=state.job_id,
        )

        state.status = JobStatus.GENERATING_NOTEBOOK
        start_time = time.time()

        try:
            # Create the notebook
            nb = new_notebook()
            nb.metadata = {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "version": "3.11",
                },
            }

            # Generate sections
            cells = []

            # Title and Introduction
            cells.extend(await self._generate_introduction(state))

            # Setup and Imports
            cells.extend(self._generate_setup_cells())

            # Data Loading
            cells.extend(await self._generate_data_loading(state))

            # Exploratory Data Analysis (comprehensive)
            cells.extend(await self._generate_comprehensive_eda(state))

            # Causal Structure
            if state.proposed_dag:
                cells.extend(await self._generate_causal_structure(state))

            # Treatment Effect Estimation
            cells.extend(await self._generate_treatment_effects(state))

            # Sensitivity Analysis
            if state.sensitivity_results:
                cells.extend(await self._generate_sensitivity(state))

            # Conclusions
            cells.extend(await self._generate_conclusions(state))

            # Add all cells to notebook
            nb.cells = cells

            # Save notebook
            notebook_path = self._save_notebook(nb, state.job_id)
            state.notebook_path = notebook_path

            # Create trace
            duration_ms = int((time.time() - start_time) * 1000)
            trace = self.create_trace(
                action="notebook_generated",
                reasoning="Generated reproducible Jupyter notebook",
                outputs={
                    "path": notebook_path,
                    "n_cells": len(cells),
                },
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

    async def _generate_introduction(self, state: AnalysisState) -> list:
        """Generate introduction section."""
        cells = []

        # Title
        title = f"# Causal Inference Analysis\n\n"
        title += f"**Dataset**: {state.dataset_info.name or state.dataset_info.url}\n\n"
        title += f"**Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        title += f"**Job ID**: {state.job_id}\n"
        cells.append(new_markdown_cell(title))

        # Introduction narrative
        intro = "## Introduction\n\n"
        intro += "This notebook presents a causal inference analysis aimed at estimating "
        intro += f"the effect of **{state.treatment_variable}** on **{state.outcome_variable}**.\n\n"

        if state.data_profile:
            intro += f"The dataset contains {state.data_profile.n_samples:,} observations "
            intro += f"and {state.data_profile.n_features} variables.\n\n"

        intro += "### Analysis Overview\n\n"
        intro += "1. Data loading and preprocessing\n"
        intro += "2. Exploratory data analysis\n"
        if state.proposed_dag:
            intro += "3. Causal structure discovery\n"
        intro += f"4. Treatment effect estimation ({len(state.treatment_effects)} methods)\n"
        if state.sensitivity_results:
            intro += f"5. Sensitivity analysis ({len(state.sensitivity_results)} tests)\n"
        intro += "6. Conclusions and recommendations\n"

        cells.append(new_markdown_cell(intro))

        return cells

    def _generate_setup_cells(self) -> list:
        """Generate setup and import cells."""
        cells = []

        # Imports markdown
        cells.append(new_markdown_cell("## Setup\n\nFirst, we import the required libraries."))

        # Import code
        imports = """# Standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical libraries
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors

# Causal inference libraries (optional - install if needed)
try:
    from econml.dml import CausalForestDML, LinearDML
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    print("econml not available - some methods will be skipped")

try:
    from causallearn.search.ConstraintBased.PC import pc
    CAUSALLEARN_AVAILABLE = True
except ImportError:
    CAUSALLEARN_AVAILABLE = False
    print("causal-learn not available - causal discovery will be skipped")

# Visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

print("Setup complete!")"""

        cells.append(new_code_cell(imports))

        return cells

    async def _generate_data_loading(self, state: AnalysisState) -> list:
        """Generate data loading section."""
        cells = []

        cells.append(new_markdown_cell("## Data Loading\n\nLoad and preview the dataset."))

        # Data loading code
        load_code = f'''# Load the dataset
# Note: Update the path to your local file or use the Kaggle API
DATA_PATH = "{state.dataset_info.local_path or state.dataset_info.url}"

try:
    df = pd.read_csv(DATA_PATH)
except (FileNotFoundError, pd.errors.ParserError) as e:
    # If direct loading fails, you may need to download from Kaggle
    # !pip install kaggle
    # !kaggle datasets download -d <dataset_id>
    raise FileNotFoundError(f"Could not load data from {{DATA_PATH}}")

print(f"Dataset shape: {{df.shape}}")
print(f"\\nColumns: {{list(df.columns)}}")'''

        cells.append(new_code_cell(load_code))

        # Preview code
        preview_code = """# Preview the data
df.head(10)"""
        cells.append(new_code_cell(preview_code))

        # Data info
        info_code = """# Data types and missing values
print("Data Types:")
print(df.dtypes)
print(f"\\nMissing Values:")
print(df.isnull().sum())"""
        cells.append(new_code_cell(info_code))

        return cells

    async def _generate_eda(self, state: AnalysisState) -> list:
        """Generate EDA section."""
        cells = []

        cells.append(new_markdown_cell("## Exploratory Data Analysis\n\nExamine the key variables."))

        # Treatment variable analysis
        treatment_code = f'''# Treatment Variable: {state.treatment_variable}
treatment_var = "{state.treatment_variable}"

print(f"Treatment variable: {{treatment_var}}")
print(f"\\nValue counts:")
print(df[treatment_var].value_counts())
print(f"\\nBasic statistics:")
print(df[treatment_var].describe())

# Visualize treatment distribution
fig, ax = plt.subplots(figsize=(8, 5))
if df[treatment_var].nunique() <= 10:
    df[treatment_var].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f'Distribution of {{treatment_var}}')
else:
    df[treatment_var].hist(bins=30, ax=ax)
    ax.set_title(f'Distribution of {{treatment_var}}')
ax.set_xlabel(treatment_var)
ax.set_ylabel('Count')
plt.tight_layout()
plt.show()'''
        cells.append(new_code_cell(treatment_code))

        # Outcome variable analysis
        outcome_code = f'''# Outcome Variable: {state.outcome_variable}
outcome_var = "{state.outcome_variable}"

print(f"Outcome variable: {{outcome_var}}")
print(f"\\nBasic statistics:")
print(df[outcome_var].describe())

# Visualize outcome distribution
fig, ax = plt.subplots(figsize=(8, 5))
df[outcome_var].hist(bins=30, ax=ax)
ax.set_title(f'Distribution of {{outcome_var}}')
ax.set_xlabel(outcome_var)
ax.set_ylabel('Frequency')
plt.tight_layout()
plt.show()'''
        cells.append(new_code_cell(outcome_code))

        # Treatment vs outcome
        comparison_code = f'''# Treatment vs Outcome
fig, ax = plt.subplots(figsize=(10, 6))

if df[treatment_var].nunique() <= 5:
    # Box plot for categorical treatment
    df.boxplot(column=outcome_var, by=treatment_var, ax=ax)
    ax.set_title(f'{{outcome_var}} by {{treatment_var}}')
else:
    # Scatter plot for continuous treatment
    ax.scatter(df[treatment_var], df[outcome_var], alpha=0.5)
    ax.set_xlabel(treatment_var)
    ax.set_ylabel(outcome_var)
    ax.set_title(f'{{outcome_var}} vs {{treatment_var}}')

plt.tight_layout()
plt.show()

# Simple correlation
if df[treatment_var].dtype in ['int64', 'float64']:
    corr = df[[treatment_var, outcome_var]].corr().iloc[0, 1]
    print(f"\\nCorrelation between {{treatment_var}} and {{outcome_var}}: {{corr:.3f}}")'''
        cells.append(new_code_cell(comparison_code))

        return cells

    async def _generate_comprehensive_eda(self, state: AnalysisState) -> list:
        """Generate comprehensive EDA section with visualizations."""
        cells = []

        cells.append(new_markdown_cell("""## Exploratory Data Analysis

This section provides comprehensive exploratory analysis to assess data quality
and readiness for causal inference."""))

        # Data Quality Summary
        if state.eda_result:
            quality_md = f"""### Data Quality Summary

**Overall Quality Score**: {state.eda_result.data_quality_score:.1f}/100

"""
            if state.eda_result.data_quality_issues:
                quality_md += "**Issues Identified:**\n"
                for issue in state.eda_result.data_quality_issues:
                    quality_md += f"- {issue}\n"
            else:
                quality_md += "No significant data quality issues detected.\n"

            cells.append(new_markdown_cell(quality_md))

        # Distribution Analysis
        cells.append(new_markdown_cell("### Distribution Analysis"))
        dist_code = f'''# Distribution analysis for key variables
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Select numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Treatment and outcome
treatment_var = "{state.treatment_variable}"
outcome_var = "{state.outcome_variable}"

# Plot distributions for treatment and outcome
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Treatment distribution
ax = axes[0, 0]
if df[treatment_var].nunique() <= 10:
    df[treatment_var].value_counts().plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
else:
    df[treatment_var].hist(bins=30, ax=ax, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_title(f'Distribution of {{treatment_var}} (Treatment)')
ax.set_xlabel(treatment_var)
ax.set_ylabel('Count')

# Outcome distribution
ax = axes[0, 1]
df[outcome_var].hist(bins=30, ax=ax, color='coral', alpha=0.7, edgecolor='black')
ax.set_title(f'Distribution of {{outcome_var}} (Outcome)')
ax.set_xlabel(outcome_var)
ax.set_ylabel('Count')

# QQ plot for outcome (normality check)
ax = axes[1, 0]
stats.probplot(df[outcome_var].dropna(), dist="norm", plot=ax)
ax.set_title(f'Q-Q Plot: {{outcome_var}}')

# Box plots by treatment group
ax = axes[1, 1]
if df[treatment_var].nunique() <= 5:
    df.boxplot(column=outcome_var, by=treatment_var, ax=ax)
    ax.set_title(f'{{outcome_var}} by {{treatment_var}}')
else:
    # Scatter for continuous treatment
    ax.scatter(df[treatment_var], df[outcome_var], alpha=0.3)
    ax.set_xlabel(treatment_var)
    ax.set_ylabel(outcome_var)
    ax.set_title(f'{{outcome_var}} vs {{treatment_var}}')

plt.suptitle('Key Variable Distributions', y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

# Print distribution statistics
print("\\nDistribution Statistics:")
print(f"{{outcome_var}}:")
print(f"  Skewness: {{df[outcome_var].skew():.3f}}")
print(f"  Kurtosis: {{df[outcome_var].kurtosis():.3f}}")'''
        cells.append(new_code_cell(dist_code))

        # Correlation Matrix
        cells.append(new_markdown_cell("### Correlation Analysis"))
        corr_code = '''# Correlation matrix heatmap
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Limit to top 15 variables if too many
if len(numeric_df.columns) > 15:
    # Select most correlated with outcome
    correlations = numeric_df.corr()[outcome_var].abs().sort_values(ascending=False)
    top_cols = correlations.head(15).index.tolist()
    numeric_df = numeric_df[top_cols]

# Compute correlation matrix
corr_matrix = numeric_df.corr()

# Plot heatmap
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Matrix (Lower Triangle)', fontsize=14)
plt.tight_layout()
plt.show()

# Identify high correlations
print("\\nHigh Correlations (|r| > 0.7):")
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.7:
            print(f"  {corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {r:.3f}")'''
        cells.append(new_code_cell(corr_code))

        # Outlier Detection
        cells.append(new_markdown_cell("### Outlier Detection"))
        outlier_code = '''# Outlier detection using IQR method
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (series < lower) | (series > upper)

# Check outliers in key numeric variables
outlier_summary = {}
for col in numeric_df.columns[:10]:  # Limit to first 10
    mask = detect_outliers_iqr(df[col].dropna())
    outlier_pct = mask.sum() / len(mask) * 100
    if outlier_pct > 0:
        outlier_summary[col] = outlier_pct

# Visualize
if outlier_summary:
    fig, ax = plt.subplots(figsize=(10, 5))
    cols = list(outlier_summary.keys())
    pcts = list(outlier_summary.values())
    bars = ax.barh(cols, pcts, color='coral', alpha=0.7)
    ax.set_xlabel('Outlier Percentage (%)')
    ax.set_title('Outlier Prevalence by Variable')
    ax.axvline(x=5, color='red', linestyle='--', label='5% threshold')
    ax.legend()
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=9)
    plt.tight_layout()
    plt.show()
else:
    print("No significant outliers detected.")

# Box plots for variables with outliers
if outlier_summary:
    n_plots = min(4, len(outlier_summary))
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    for i, col in enumerate(list(outlier_summary.keys())[:n_plots]):
        axes[i].boxplot(df[col].dropna())
        axes[i].set_title(f'{col}')
    plt.suptitle('Box Plots of Variables with Outliers', y=1.02)
    plt.tight_layout()
    plt.show()'''
        cells.append(new_code_cell(outlier_code))

        # Covariate Balance (if treatment is binary)
        cells.append(new_markdown_cell("### Covariate Balance Assessment"))
        balance_code = f'''# Covariate balance between treatment groups
treatment_var = "{state.treatment_variable}"

if df[treatment_var].nunique() == 2:
    # Split into treatment and control
    treated = df[df[treatment_var] == df[treatment_var].max()]
    control = df[df[treatment_var] == df[treatment_var].min()]

    print(f"Treatment Group: n={{len(treated)}}")
    print(f"Control Group: n={{len(control)}}")

    # Calculate Standardized Mean Difference (SMD) for covariates
    balance_results = []
    covariates = [c for c in numeric_df.columns if c not in [treatment_var, outcome_var]][:10]

    for cov in covariates:
        t_mean = treated[cov].mean()
        c_mean = control[cov].mean()
        pooled_std = np.sqrt((treated[cov].std()**2 + control[cov].std()**2) / 2)

        if pooled_std > 0:
            smd = abs(t_mean - c_mean) / pooled_std
        else:
            smd = 0

        balance_results.append({{
            'Covariate': cov,
            'Treated Mean': t_mean,
            'Control Mean': c_mean,
            'SMD': smd,
            'Balanced': 'Yes' if smd < 0.1 else 'No'
        }})

    balance_df = pd.DataFrame(balance_results)
    print("\\nCovariate Balance Summary:")
    print(balance_df.to_string(index=False))

    # Love plot
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(balance_df))
    colors = ['green' if b == 'Yes' else 'red' for b in balance_df['Balanced']]
    ax.barh(y_pos, balance_df['SMD'], color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(balance_df['Covariate'])
    ax.axvline(x=0.1, color='red', linestyle='--', label='SMD = 0.1 threshold')
    ax.set_xlabel('Standardized Mean Difference (SMD)')
    ax.set_title('Covariate Balance (Love Plot)')
    ax.legend()
    plt.tight_layout()
    plt.show()

    n_imbalanced = (balance_df['Balanced'] == 'No').sum()
    if n_imbalanced > 0:
        print(f"\\n⚠️ {{n_imbalanced}} covariates are imbalanced (SMD >= 0.1)")
    else:
        print("\\n✓ All covariates are well-balanced (SMD < 0.1)")
else:
    print("Treatment is not binary. Skipping balance assessment.")'''
        cells.append(new_code_cell(balance_code))

        # Multicollinearity Check
        cells.append(new_markdown_cell("### Multicollinearity Check (VIF)"))
        vif_code = '''# Variance Inflation Factor (VIF) for multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Prepare covariates matrix
covariates = [c for c in numeric_df.columns if c not in [treatment_var, outcome_var]][:10]
X = df[covariates].dropna()

if len(X) > 10 and len(covariates) >= 2:
    # Add constant
    X_with_const = np.column_stack([np.ones(len(X)), X.values])

    vif_data = []
    for i, col in enumerate(covariates):
        try:
            vif = variance_inflation_factor(X_with_const, i + 1)
            vif_data.append({'Variable': col, 'VIF': vif})
        except (ValueError, np.linalg.LinAlgError):
            pass  # Skip columns that cause numerical issues

    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

    print("Variance Inflation Factors:")
    print(vif_df.to_string(index=False))

    # Flag high VIF
    high_vif = vif_df[vif_df['VIF'] > 5]
    if len(high_vif) > 0:
        print(f"\\n⚠️ Variables with VIF > 5 (potential multicollinearity):")
        for _, row in high_vif.iterrows():
            severity = "SEVERE" if row['VIF'] > 10 else "moderate"
            print(f"  - {row['Variable']}: VIF = {row['VIF']:.2f} ({severity})")
    else:
        print("\\n✓ No severe multicollinearity detected (all VIF < 5)")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['red' if v > 10 else 'orange' if v > 5 else 'green' for v in vif_df['VIF']]
    ax.barh(vif_df['Variable'], vif_df['VIF'], color=colors, alpha=0.7)
    ax.axvline(x=5, color='orange', linestyle='--', label='VIF = 5')
    ax.axvline(x=10, color='red', linestyle='--', label='VIF = 10')
    ax.set_xlabel('Variance Inflation Factor')
    ax.set_title('Multicollinearity Assessment')
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Insufficient data for VIF calculation.")'''
        cells.append(new_code_cell(vif_code))

        # Missing Data Analysis
        cells.append(new_markdown_cell("### Missing Data Analysis"))
        missing_code = '''# Missing data analysis
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Variable': missing.index,
    'Missing Count': missing.values,
    'Missing %': missing_pct.values
}).sort_values('Missing %', ascending=False)

# Only show variables with missing data
missing_df = missing_df[missing_df['Missing Count'] > 0]

if len(missing_df) > 0:
    print("Variables with Missing Data:")
    print(missing_df.to_string(index=False))

    # Visualization
    if len(missing_df) <= 20:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if p > 30 else 'orange' if p > 10 else 'green' for p in missing_df['Missing %']]
        ax.barh(missing_df['Variable'], missing_df['Missing %'], color=colors, alpha=0.7)
        ax.set_xlabel('Missing Percentage (%)')
        ax.set_title('Missing Data by Variable')
        ax.axvline(x=10, color='orange', linestyle='--', alpha=0.7)
        ax.axvline(x=30, color='red', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
else:
    print("✓ No missing data in the dataset!")

# Overall summary
total_missing = df.isnull().sum().sum()
total_cells = df.shape[0] * df.shape[1]
print(f"\\nOverall: {total_missing:,} missing values out of {total_cells:,} total ({100*total_missing/total_cells:.2f}%)")'''
        cells.append(new_code_cell(missing_code))

        # EDA Summary
        if state.eda_result and state.eda_result.summary_table:
            summary_md = "### EDA Summary\n\n"

            if "key_findings" in state.eda_result.summary_table:
                summary_md += "**Key Findings:**\n"
                for finding in state.eda_result.summary_table["key_findings"]:
                    summary_md += f"- {finding}\n"
                summary_md += "\n"

            if "recommendations" in state.eda_result.summary_table:
                summary_md += "**Recommendations for Causal Analysis:**\n"
                for rec in state.eda_result.summary_table["recommendations"]:
                    summary_md += f"- {rec}\n"
                summary_md += "\n"

            if "causal_readiness" in state.eda_result.summary_table:
                readiness = state.eda_result.summary_table["causal_readiness"]
                emoji = "✅" if readiness == "ready" else "⚠️" if readiness == "needs_attention" else "❌"
                summary_md += f"**Causal Inference Readiness**: {emoji} {readiness.replace('_', ' ').title()}\n"

            cells.append(new_markdown_cell(summary_md))

        return cells

    async def _generate_causal_structure(self, state: AnalysisState) -> list:
        """Generate causal structure section."""
        cells = []

        dag = state.proposed_dag

        cells.append(new_markdown_cell(f"""## Causal Structure

The causal graph was discovered using the **{dag.discovery_method}** algorithm.

This graph represents our hypothesis about the causal relationships between variables.
**Important**: The discovered structure is a hypothesis based on statistical associations
and should be validated with domain knowledge.
"""))

        # DAG visualization code
        dag_code = f'''# Causal Graph Visualization
import networkx as nx

# Create directed graph
G = nx.DiGraph()

# Add nodes
nodes = {json.dumps(dag.nodes)}
G.add_nodes_from(nodes)

# Add edges
edges = {json.dumps([(e.source, e.target) for e in dag.edges])}
G.add_edges_from(edges)

# Draw the graph
fig, ax = plt.subplots(figsize=(12, 8))
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

# Draw nodes
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

ax.set_title('Discovered Causal Graph\\n(Green=Treatment, Red=Outcome, Blue=Other)')
ax.axis('off')
plt.tight_layout()
plt.show()

print(f"Number of nodes: {{len(G.nodes())}}")
print(f"Number of edges: {{len(G.edges())}}")'''
        cells.append(new_code_cell(dag_code))

        return cells

    async def _generate_treatment_effects(self, state: AnalysisState) -> list:
        """Generate treatment effect estimation section."""
        cells = []

        cells.append(new_markdown_cell(f"""## Treatment Effect Estimation

We estimate the causal effect of **{state.treatment_variable}** on **{state.outcome_variable}**
using multiple methods for robustness.

**Treatment**: {state.treatment_variable}
**Outcome**: {state.outcome_variable}
"""))

        # Setup variables
        setup_code = f'''# Define treatment and outcome
treatment_var = "{state.treatment_variable}"
outcome_var = "{state.outcome_variable}"

# Prepare data
df_analysis = df[[treatment_var, outcome_var]].dropna()

# Get covariates (potential confounders)
covariates = {json.dumps(state.data_profile.potential_confounders[:10] if state.data_profile else [])}
covariates = [c for c in covariates if c in df.columns and c not in [treatment_var, outcome_var]]

if covariates:
    df_analysis = df[[treatment_var, outcome_var] + covariates].dropna()
    X = df_analysis[covariates].values
else:
    X = None

T = df_analysis[treatment_var].values
Y = df_analysis[outcome_var].values

# Binarize treatment if needed
if len(np.unique(T)) > 2:
    T_binary = (T > np.median(T)).astype(int)
    print(f"Treatment binarized at median ({{np.median(T):.2f}})")
else:
    T_binary = T

print(f"Sample size: {{len(T)}}")
print(f"Treated: {{np.sum(T_binary == 1)}}")
print(f"Control: {{np.sum(T_binary == 0)}}")'''
        cells.append(new_code_cell(setup_code))

        # Results summary from actual analysis
        results_md = "### Results Summary\n\n"
        results_md += "| Method | Estimand | Estimate | 95% CI | p-value |\n"
        results_md += "|--------|----------|----------|--------|--------|\n"

        for effect in state.treatment_effects:
            pval = f"{effect.p_value:.4f}" if effect.p_value else "N/A"
            results_md += f"| {effect.method} | {effect.estimand} | "
            results_md += f"{effect.estimate:.4f} | [{effect.ci_lower:.4f}, {effect.ci_upper:.4f}] | {pval} |\n"

        cells.append(new_markdown_cell(results_md))

        # OLS Regression code
        cells.append(new_markdown_cell("### Method 1: OLS Regression"))
        ols_code = '''# OLS Regression with covariates
if X is not None and len(X) > 0:
    design = np.column_stack([np.ones(len(T_binary)), T_binary, X])
else:
    design = np.column_stack([np.ones(len(T_binary)), T_binary])

model = sm.OLS(Y, design)
results = model.fit()

ate_ols = results.params[1]
se_ols = results.bse[1]
ci_ols = results.conf_int()[1]
pval_ols = results.pvalues[1]

print("OLS Regression Results:")
print(f"  ATE: {ate_ols:.4f}")
print(f"  SE: {se_ols:.4f}")
print(f"  95% CI: [{ci_ols[0]:.4f}, {ci_ols[1]:.4f}]")
print(f"  p-value: {pval_ols:.4f}")
print(f"  R-squared: {results.rsquared:.4f}")'''
        cells.append(new_code_cell(ols_code))

        # IPW code
        cells.append(new_markdown_cell("### Method 2: Inverse Probability Weighting"))
        ipw_code = '''# Inverse Probability Weighting
if X is not None and len(X) > 0:
    # Estimate propensity scores
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X, T_binary)
    ps = ps_model.predict_proba(X)[:, 1]
    ps = np.clip(ps, 0.01, 0.99)

    # IPW estimator
    weights_treated = T_binary / ps
    weights_control = (1 - T_binary) / (1 - ps)

    ate_ipw = np.mean(weights_treated * Y) - np.mean(weights_control * Y)

    # Bootstrap for SE
    n_bootstrap = 500
    bootstrap_estimates = []
    n = len(Y)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        w_t = weights_treated[idx]
        w_c = weights_control[idx]
        y_b = Y[idx]
        bootstrap_estimates.append(np.mean(w_t * y_b) - np.mean(w_c * y_b))

    se_ipw = np.std(bootstrap_estimates)
    ci_ipw = (np.percentile(bootstrap_estimates, 2.5), np.percentile(bootstrap_estimates, 97.5))

    print("IPW Results:")
    print(f"  ATE: {ate_ipw:.4f}")
    print(f"  SE: {se_ipw:.4f}")
    print(f"  95% CI: [{ci_ipw[0]:.4f}, {ci_ipw[1]:.4f}]")
else:
    print("IPW requires covariates for propensity score estimation")'''
        cells.append(new_code_cell(ipw_code))

        # Results comparison visualization
        cells.append(new_markdown_cell("### Results Comparison"))
        viz_code = f'''# Visualize results comparison
results_data = {json.dumps([
    {"method": e.method, "estimate": e.estimate, "ci_lower": e.ci_lower, "ci_upper": e.ci_upper}
    for e in state.treatment_effects
])}

fig, ax = plt.subplots(figsize=(10, 6))

methods = [r['method'] for r in results_data]
estimates = [r['estimate'] for r in results_data]
ci_lower = [r['ci_lower'] for r in results_data]
ci_upper = [r['ci_upper'] for r in results_data]

# Calculate error bars
yerr_lower = [e - l for e, l in zip(estimates, ci_lower)]
yerr_upper = [u - e for e, u in zip(estimates, ci_upper)]

y_pos = range(len(methods))
ax.barh(y_pos, estimates, xerr=[yerr_lower, yerr_upper], capsize=5, alpha=0.7)
ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(methods)
ax.set_xlabel('Treatment Effect Estimate')
ax.set_title('Treatment Effect Estimates Across Methods')
plt.tight_layout()
plt.show()'''
        cells.append(new_code_cell(viz_code))

        return cells

    async def _generate_sensitivity(self, state: AnalysisState) -> list:
        """Generate sensitivity analysis section."""
        cells = []

        cells.append(new_markdown_cell("""## Sensitivity Analysis

We assess the robustness of our findings to potential violations of causal assumptions.
"""))

        # Sensitivity results summary
        sens_md = "### Sensitivity Results\n\n"
        sens_md += "| Analysis | Robustness Value | Interpretation |\n"
        sens_md += "|----------|------------------|----------------|\n"

        for sens in state.sensitivity_results:
            sens_md += f"| {sens.method} | {sens.robustness_value:.2f} | {sens.interpretation[:50]}... |\n"

        cells.append(new_markdown_cell(sens_md))

        # E-value interpretation
        cells.append(new_markdown_cell("""### Understanding Sensitivity Metrics

**E-value**: The minimum strength of association (on the risk ratio scale) that an unmeasured
confounder would need to have with both treatment and outcome to explain away the observed effect.
- E-value > 2: Moderate robustness
- E-value > 3: Good robustness

**Specification Curve**: Shows how the estimate varies across reasonable analytical choices.
Higher stability (less variation) indicates more robust findings.
"""))

        return cells

    async def _generate_conclusions(self, state: AnalysisState) -> list:
        """Generate conclusions section."""
        cells = []

        # Build conclusions based on results
        avg_effect = np.mean([e.estimate for e in state.treatment_effects]) if state.treatment_effects else 0
        all_positive = all(e.estimate > 0 for e in state.treatment_effects) if state.treatment_effects else False
        all_negative = all(e.estimate < 0 for e in state.treatment_effects) if state.treatment_effects else False

        conclusion_text = f"""## Conclusions

### Key Findings

Based on our analysis of the effect of **{state.treatment_variable}** on **{state.outcome_variable}**:

1. **Average Treatment Effect**: The estimated effect across {len(state.treatment_effects)} methods
   is approximately **{avg_effect:.4f}**.

2. **Consistency**: """

        if all_positive:
            conclusion_text += "All methods agree on a **positive** effect."
        elif all_negative:
            conclusion_text += "All methods agree on a **negative** effect."
        else:
            conclusion_text += "Methods show **mixed** results regarding the direction of effect."

        conclusion_text += """

### Limitations

- **Observational data**: We cannot rule out unmeasured confounding
- **Model assumptions**: Each method relies on specific assumptions
- **External validity**: Results may not generalize to other populations

### Recommendations

"""
        for i, rec in enumerate(state.recommendations, 1):
            conclusion_text += f"{i}. {rec}\n"

        if not state.recommendations:
            conclusion_text += """1. Validate findings with domain experts
2. Consider collecting additional data to address unmeasured confounding
3. Replicate analysis on independent datasets
"""

        cells.append(new_markdown_cell(conclusion_text))

        # Reproducibility info
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

    def _save_notebook(self, nb: nbformat.NotebookNode, job_id: str) -> str:
        """Save the notebook to disk.

        Args:
            nb: The notebook object
            job_id: Job ID for filename

        Returns:
            Path to saved notebook
        """
        # Create directory
        output_dir = Path(tempfile.gettempdir()) / "causal_orchestrator" / "notebooks"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save notebook
        filename = f"causal_analysis_{job_id}.ipynb"
        filepath = output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        return str(filepath)
