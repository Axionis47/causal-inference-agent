"""Method selection utilities for effect estimation.

Contains sample size thresholds and column matching helpers
used by the EffectEstimatorAgent to choose appropriate methods.
"""


class SampleSizeThresholds:
    """Minimum sample sizes for reliable estimation by method type."""

    # Minimum samples per treatment arm
    MIN_SAMPLES_BASIC = 30  # OLS, IPW - simple parametric methods
    MIN_SAMPLES_MATCHING = 50  # PSM - needs enough for good matches
    MIN_SAMPLES_ML = 100  # T/X/S-Learner - ML methods prone to overfitting
    MIN_SAMPLES_FOREST = 200  # Causal Forest, Double ML - complex ML

    # Total sample thresholds
    MIN_TOTAL_BASIC = 50
    MIN_TOTAL_ML = 200
    MIN_TOTAL_FOREST = 500

    @classmethod
    def get_method_requirements(cls, method: str) -> dict:
        """Get sample size requirements for a method."""
        requirements = {
            "ols": {"min_per_arm": cls.MIN_SAMPLES_BASIC, "min_total": cls.MIN_TOTAL_BASIC, "complexity": "low"},
            "ipw": {"min_per_arm": cls.MIN_SAMPLES_BASIC, "min_total": cls.MIN_TOTAL_BASIC, "complexity": "low"},
            "aipw": {"min_per_arm": cls.MIN_SAMPLES_BASIC, "min_total": cls.MIN_TOTAL_BASIC, "complexity": "medium"},
            "matching": {"min_per_arm": cls.MIN_SAMPLES_MATCHING, "min_total": 100, "complexity": "medium"},
            "s_learner": {"min_per_arm": cls.MIN_SAMPLES_ML, "min_total": cls.MIN_TOTAL_ML, "complexity": "high"},
            "t_learner": {"min_per_arm": cls.MIN_SAMPLES_ML, "min_total": cls.MIN_TOTAL_ML, "complexity": "high"},
            "x_learner": {"min_per_arm": cls.MIN_SAMPLES_ML, "min_total": cls.MIN_TOTAL_ML, "complexity": "high"},
            "causal_forest": {"min_per_arm": cls.MIN_SAMPLES_FOREST, "min_total": cls.MIN_TOTAL_FOREST, "complexity": "very_high"},
            "double_ml": {"min_per_arm": cls.MIN_SAMPLES_FOREST, "min_total": cls.MIN_TOTAL_FOREST, "complexity": "very_high"},
        }
        return requirements.get(method, {"min_per_arm": cls.MIN_SAMPLES_BASIC, "min_total": cls.MIN_TOTAL_BASIC, "complexity": "unknown"})

    @classmethod
    def check_method_viability(cls, method: str, n_treated: int, n_control: int) -> tuple[bool, str]:
        """Check if a method is viable given sample sizes.

        Returns:
            Tuple of (is_viable, reason_if_not_viable)
        """
        reqs = cls.get_method_requirements(method)
        min_arm = min(n_treated, n_control)
        total = n_treated + n_control

        if min_arm < reqs["min_per_arm"]:
            return False, f"Smallest arm has {min_arm} samples, but {method} requires {reqs['min_per_arm']}+ per arm"
        if total < reqs["min_total"]:
            return False, f"Total sample size {total} is below minimum {reqs['min_total']} for {method}"
        return True, ""

    @classmethod
    def get_recommended_methods(cls, n_treated: int, n_control: int) -> list[str]:
        """Get list of methods recommended for the given sample sizes."""
        recommended = []
        methods_priority = ["ols", "ipw", "aipw", "matching", "s_learner", "t_learner", "x_learner", "causal_forest", "double_ml"]

        for method in methods_priority:
            viable, _ = cls.check_method_viability(method, n_treated, n_control)
            if viable:
                recommended.append(method)

        return recommended

    @classmethod
    def get_sample_size_warning(cls, n_treated: int, n_control: int) -> str | None:
        """Generate a warning message if sample size is concerning."""
        min_arm = min(n_treated, n_control)
        total = n_treated + n_control

        warnings = []

        if min_arm < 50:
            warnings.append(f"⚠️ SMALL SAMPLE: Only {min_arm} samples in smallest arm. ML methods will likely overfit.")
        elif min_arm < 100:
            warnings.append(f"⚠️ MODERATE SAMPLE: {min_arm} samples in smallest arm. Use regularized methods.")

        if total < 200:
            warnings.append(f"⚠️ Limited total sample ({total}). Prefer OLS/IPW over complex ML methods.")

        if n_treated < 100 and n_control > 3 * n_treated:
            warnings.append(f"⚠️ IMBALANCED: {n_treated} treated vs {n_control} control. Consider matching methods carefully.")

        if warnings:
            return "\n".join(warnings)
        return None


def find_closest_column(name: str, columns: list[str]) -> str | None:
    """Find the closest matching column name using fuzzy matching.

    Handles cases where LLM returns "treatment" but column is "treat".
    """
    if name in columns:
        return name

    name_lower = name.lower()

    # Try exact lowercase match
    for col in columns:
        if col.lower() == name_lower:
            return col

    # Try prefix matching (treatment -> treat)
    for col in columns:
        col_lower = col.lower()
        if name_lower.startswith(col_lower) or col_lower.startswith(name_lower):
            return col

    # Try substring matching
    for col in columns:
        col_lower = col.lower()
        if name_lower in col_lower or col_lower in name_lower:
            return col

    return None
