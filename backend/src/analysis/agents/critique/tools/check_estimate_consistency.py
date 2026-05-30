"""check_estimate_consistency - cross-method agreement on the treatment effect."""

import numpy as np

SCHEMA = {
    "name": "check_estimate_consistency",
    "description": "Check consistency of treatment effect estimates across different methods.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def handle(agent, **kwargs) -> str:
    state = agent._state

    if len(state.treatment_effects) < 2:
        return "Only 1 method used - cannot check consistency across methods."

    output = "Estimate Consistency Check:\n"
    output += "=" * 50 + "\n"

    estimates = np.array([effect.estimate for effect in state.treatment_effects])
    methods = [effect.method for effect in state.treatment_effects]

    mean_est = np.mean(estimates)
    std_est = np.std(estimates)
    cv = std_est / (abs(mean_est) + 1e-10) * 100

    output += f"Mean estimate: {mean_est:.4f}\n"
    output += f"Std across methods: {std_est:.4f}\n"
    output += f"Coefficient of variation: {cv:.1f}%\n\n"

    output += "Individual estimates:\n"
    for method, est in zip(methods, estimates, strict=False):
        deviation = abs(est - mean_est) / (std_est + 1e-10)
        flag = " (outlier)" if deviation > 2 else ""
        output += f"  {method}: {est:.4f}{flag}\n"

    if cv < 10:
        output += "\nConsistency: EXCELLENT - estimates agree within 10%\n"
        agent._investigation_evidence.append("Estimates highly consistent across methods")
    elif cv < 25:
        output += "\nConsistency: GOOD - reasonable agreement across methods\n"
        agent._investigation_evidence.append("Estimates moderately consistent")
    elif cv < 50:
        output += "\nConsistency: MODERATE - some variation across methods\n"
        agent._investigation_evidence.append("Estimates show notable variation")
    else:
        output += "\nConsistency: POOR - estimates vary significantly\n"
        agent._investigation_evidence.append("CONCERN: Estimates inconsistent across methods")

    positive = sum(1 for e in estimates if e > 0)
    if positive == len(estimates) or positive == 0:
        output += "Sign: All estimates have same sign (good)\n"
    else:
        output += f"Sign: WARNING - {positive}/{len(estimates)} positive (inconsistent)\n"
        agent._investigation_evidence.append("CONCERN: Estimates have inconsistent signs")

    return output
