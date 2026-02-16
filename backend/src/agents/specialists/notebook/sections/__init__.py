"""Notebook section renderers — one file per pipeline section."""

from .causal_structure import render_causal_structure
from .conclusions import render_conclusions
from .confounder_analysis import render_confounder_analysis
from .critique import render_critique_section
from .data_loading import render_data_loading
from .data_profile import render_data_profile_report
from .data_repairs import render_data_repairs
from .domain_knowledge import render_domain_knowledge
from .eda import render_eda_report
from .introduction import render_introduction
from .ps_diagnostics import render_ps_diagnostics
from .sensitivity import render_sensitivity
from .setup import render_setup_cells
from .treatment_effects import render_treatment_effects

__all__ = [
    "render_introduction",
    "render_domain_knowledge",
    "render_setup_cells",
    "render_data_loading",
    "render_data_profile_report",
    "render_data_repairs",
    "render_eda_report",
    "render_causal_structure",
    "render_confounder_analysis",
    "render_ps_diagnostics",
    "render_treatment_effects",
    "render_sensitivity",
    "render_critique_section",
    "render_conclusions",
]
