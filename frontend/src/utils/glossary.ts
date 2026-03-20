/** Tooltip glossary: short codes to plain-English definitions. */
const glossary: Record<string, string> = {
  ATE: 'Average Treatment Effect — the average difference in the outcome if everyone received the treatment versus if no one did.',
  ATT: 'Average Treatment Effect on the Treated — the average effect specifically among those who actually received the treatment.',
  CATE: 'Conditional Average Treatment Effect — the treatment effect for a specific subgroup defined by certain characteristics.',
  LATE: 'Local Average Treatment Effect — the effect among people whose treatment status was changed by an instrument or encouragement.',
  OLS: 'Ordinary Least Squares — fits a straight-line relationship between the treatment and outcome while adjusting for other variables.',
  IPW: 'Inverse Probability Weighting — reweights observations so treated and untreated groups look comparable.',
  AIPW: 'Augmented IPW — combines outcome modeling with probability weighting for a more reliable estimate.',
  PSM: 'Propensity Score Matching — pairs each treated unit with an untreated unit that had a similar probability of being treated.',
  DiD: 'Difference-in-Differences — compares outcome changes over time between treated and untreated groups, removing shared trends.',
  'IV/2SLS': 'Instrumental Variables — uses an external factor that affects treatment but not the outcome directly.',
  RDD: 'Regression Discontinuity Design — estimates the effect by comparing units just above and below a treatment cutoff.',
  'S-Learner': 'Trains a single model on both groups, then compares predictions with and without treatment.',
  'T-Learner': 'Trains separate models for treated and untreated groups, then takes the difference.',
  'X-Learner': 'A two-stage approach using cross-group predictions and propensity scores; good when group sizes differ.',
  'Causal Forest': 'A machine-learning method using many decision trees to estimate how the treatment effect varies across individuals.',
  'Double ML': 'Uses ML to control for confounders in a first stage, then estimates the causal effect in a second stage.',
  'p-value': 'The probability of seeing a result this extreme if the treatment truly had no effect. Below 0.05 suggests statistical significance.',
  CI: 'Confidence Interval — a range likely to contain the true treatment effect. A 95% CI means ~95% of such intervals would contain the true value.',
  '95% CI': 'A range likely to contain the true treatment effect. If repeated many times, about 95% of such intervals would contain the true value.',
  'E-value': 'The minimum strength an unmeasured confounder would need to fully explain away the observed effect. Larger = more robust.',
  'Std. Error': 'Standard Error — how much the estimate would vary across different samples. Smaller = more precise.',
  'Rosenbaum Bounds': 'A sensitivity check: how strong would hidden bias have to be to change the conclusion?',
  'Specification Curve': 'Re-runs analysis under many alternative model choices to check if the effect holds consistently.',
  'Placebo Test': 'Applies the analysis where no effect should exist. A "significant" placebo result casts doubt on the original finding.',
};

export default glossary;
